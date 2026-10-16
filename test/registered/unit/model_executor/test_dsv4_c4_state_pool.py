"""DeepSeek V4 C4 compress-state pool sized by the page it is addressed by.

The compressor addresses the C4 state ring as
swa_page * ring_size + swa_loc % ring_size with swa_page = swa_loc // page_size,
one 8-slot ring per 256-token SWA page. The pool configurator counted one ring per
128-token sliding window, which allocated twice the rows that the compressor can
address. These tests pin the pool sizes of a DeepSeek-V4-Flash launch (TP 8,
--context-length 262144, --max-total-tokens 262144, --page-size 256,
--max-running-requests 8, swa_full_tokens_ratio 0.1) and check that the smaller
pool still covers every row the compressor can address. CPU only.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.mem_cache import deepseek_v4_memory_pool
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.model_executor import pool_configurator
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE_SIZE = 256
MAX_TOTAL_TOKENS = 262144
MAX_RUNNING_REQUESTS = 8
# The fixture has 43 layers: two zero ratios, then 21 C4 and 20 C128 ratios.
# Three trailing zero ratios lie outside the layer slice.
COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4, 0, 0, 0]
NUM_LAYERS = 43


def _make_kvc() -> SimpleNamespace:
    model_config = SimpleNamespace(
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        index_head_dim=128,
        context_len=MAX_TOTAL_TOKENS,
        compress_ratios=COMPRESS_RATIOS,
        window_size=128,
    )
    server_args = SimpleNamespace(
        swa_full_tokens_ratio=0.1,
        speculative_algorithm=None,
        max_speculative_num_draft_tokens=None,
        max_running_requests=MAX_RUNNING_REQUESTS,
        disaggregation_mode="null",
        disaggregation_decode_extra_slots=0,
        enable_hisparse=False,
    )
    return SimpleNamespace(
        kv_cache_dtype_str="fp8_e4m3",
        model_config=model_config,
        layer_info=SimpleNamespace(start_layer=0, end_layer=NUM_LAYERS),
        ps=ParallelState.trivial(),
        server_args=server_args,
    )


def _make_configurator() -> DSV4PoolConfigurator:
    # create=True: the name is imported by the fix, so the patch also works on the
    # code before the fix, where the tests then fail on the pool sizes.
    with patch.object(
        pool_configurator,
        "get_schedule",
        create=True,
        return_value=SimpleNamespace(page_size=PAGE_SIZE),
    ):
        return DSV4PoolConfigurator(_make_kvc())


class TestDSV4C4StatePool(CustomTestCase):
    def setUp(self) -> None:
        # Pin the settings that shape the sizes asserted here, so that a value in the
        # environment of the test host cannot change them. ONLINE_C128 is read from
        # the environment when its module loads, so the test patches the constant.
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(envs.SGLANG_OPT_USE_ONLINE_COMPRESS.override(False))
        stack.enter_context(envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.override("float32"))
        stack.enter_context(patch.object(deepseek_v4_memory_pool, "ONLINE_C128", False))
        self.cfg = _make_configurator()
        self.config = self.cfg.calculate_pool_sizes_from_max_tokens(
            MAX_TOTAL_TOKENS, PAGE_SIZE
        )
        self.config.max_running_requests = MAX_RUNNING_REQUESTS
        self.config = self.cfg.finalize_with_max_running_requests(self.config)

    def test_layer_counts(self) -> None:
        self.assertEqual(self.cfg.num_layers_ca4, 21)
        self.assertEqual(self.cfg.num_layers_ca128, 20)
        self.assertEqual(self.cfg.c4_ring_size, 8)

    def test_pool_sizes_for_deployed_flags(self) -> None:
        # The c4 state pool is 26112 // 256 * 8 = 816 rows. Counting one ring per
        # 128-token window gave 26112 // 128 * 8 = 1632. The other pools do not change.
        c = self.config
        self.assertEqual(c.full_max_total_num_tokens, 262144)
        self.assertEqual(c.swa_max_total_num_tokens, 26112)
        self.assertEqual(c.c4_max_total_num_tokens, 65536)
        self.assertEqual(c.c128_max_total_num_tokens, 2048)
        self.assertEqual(c.c4_state_pool_size, 816)
        self.assertEqual(c.c128_state_pool_size, (MAX_RUNNING_REQUESTS + 1) * 128)

    def test_bytes_per_full_token(self) -> None:
        # 0.1 * 584 * 43 + 0.25 * 584 * 21 + 584 / 128 * 20 + 0.25 * 132 * 21
        # + 0.1 * (8 / 256) * (8192 + 2048) * 21 = 7033.45. The ring counted per
        # 128-token window gave 7705.45.
        self.assertAlmostEqual(self.cfg.bytes_per_full_token, 7033.45, places=6)

    def test_pool_covers_every_addressed_row(self) -> None:
        """SWA pages 1..N (page 0 is padding) with N = swa_tokens // page_size;
        the highest state row the compressor can address is N * ring + ring - 1."""
        swa_pages = self.config.swa_max_total_num_tokens // PAGE_SIZE
        ring = self.cfg.c4_ring_size
        swa_loc = torch.arange(PAGE_SIZE, (swa_pages + 1) * PAGE_SIZE)
        addressed = (swa_loc // PAGE_SIZE) * ring + swa_loc % ring
        pool = CompressStatePool(
            size=self.config.c4_state_pool_size,
            ring_size=ring,
            overlap=True,
            head_dim=512,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=4,
            swa_page_size=PAGE_SIZE,
        )
        rows = pool.kv_score_buffer.kv_score.shape[0]
        self.assertEqual(int(addressed.max()), 823)
        self.assertLess(int(addressed.max()), rows - 1)  # last row is the sentinel
        self.assertEqual(rows, 828)
        # Same addressing through the pool's own translation.
        self.assertTrue(
            torch.equal(pool.translate_from_swa_loc_to_state_loc(swa_loc), addressed)
        )

    def test_freed_bytes(self) -> None:
        """Per GPU: (1644 - 828) rows x (8192 B attention + 2048 B indexer state)
        x 21 C4 layers, with rows = ceil4(size + ring + 1)."""

        def rows(size: int) -> int:
            return (size + 8 + 1 + 3) // 4 * 4

        old = rows(26112 // 128 * 8)
        new = rows(self.config.c4_state_pool_size)
        self.assertEqual((old, new), (1644, 828))
        self.assertEqual((old - new) * (8192 + 2048) * 21, 175_472_640)


if __name__ == "__main__":
    unittest.main()
