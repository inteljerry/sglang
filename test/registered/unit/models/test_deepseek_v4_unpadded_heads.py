"""DeepSeek V4 on XPU: sparse attention calls that take the heads of a rank unpadded.

With attention TP, MQALayer pads the query heads of each rank to 64 when they fit,
else to n_heads, for CUDA FlashMLA. The XPU switches
SGLANG_OPT_XPU_DSV4_UNPADDED_HEADS_DECODE and _PREFILL skip the padding in decode
and in extend batches without speculative decoding. The unpadded call then needs
the attn_sink entries of the rank, one per local head.
"""

import unittest
from types import SimpleNamespace

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PREFILL_MODES = (
    ForwardMode.EXTEND,
    ForwardMode.MIXED,
    ForwardMode.SPLIT_PREFILL,
    ForwardMode.DLLM_EXTEND,
)
PADDED_MODES = (
    ForwardMode.TARGET_VERIFY,
    ForwardMode.DRAFT_EXTEND_V2,
    ForwardMode.IDLE,
    ForwardMode.PREBUILT,
)


def unpadded(mode: ForwardMode, **case: bool | int) -> bool:
    """_use_unpadded_heads for 8 attention TP ranks of 8 heads, both switches on."""
    case = {"tp": 8, "heads": 8, "decode": True, "prefill": True, **case}
    return deepseek_v4._use_unpadded_heads(
        mode, case["tp"], case["heads"], decode=case["decode"], prefill=case["prefill"]
    )


class TestUseUnpaddedHeads(CustomTestCase):
    def test_decode_follows_the_decode_switch(self) -> None:
        self.assertTrue(unpadded(ForwardMode.DECODE, prefill=False))
        self.assertFalse(unpadded(ForwardMode.DECODE, decode=False))

    def test_extend_without_speculative_follows_the_prefill_switch(self) -> None:
        for mode in PREFILL_MODES:
            with self.subTest(mode=mode):
                self.assertTrue(unpadded(mode, decode=False))
                self.assertFalse(unpadded(mode, prefill=False))

    def test_other_modes_keep_the_padding(self) -> None:
        for mode in PADDED_MODES:
            with self.subTest(mode=mode):
                self.assertFalse(unpadded(mode))

    def test_one_rank_or_an_unsupported_head_count_keeps_the_padding(self) -> None:
        # one rank pads nothing, more than 64 heads pad to n_heads, not to 64, and the
        # XPU kernels take only multiples of 8 heads
        for mode in (ForwardMode.DECODE, ForwardMode.EXTEND):
            with self.subTest(mode=mode):
                self.assertFalse(unpadded(mode, tp=1))
                self.assertFalse(unpadded(mode, heads=72))
                self.assertFalse(unpadded(mode, heads=4))
                self.assertTrue(unpadded(mode, heads=64))
                self.assertTrue(unpadded(mode, heads=24))

    def test_the_switches_are_off_by_default(self) -> None:
        self.assertIs(envs.SGLANG_OPT_XPU_DSV4_UNPADDED_HEADS_DECODE.default, False)
        self.assertIs(envs.SGLANG_OPT_XPU_DSV4_UNPADDED_HEADS_PREFILL.default, False)


class TestLocalAttnSinkUnpadded(CustomTestCase):
    def _layer(self, rank: int, heads: int) -> SimpleNamespace:
        return SimpleNamespace(
            attn_sink=torch.arange(64, dtype=torch.float32),
            attn_tp_rank=rank,
            n_local_heads=heads,
        )

    def test_the_rank_gets_its_own_heads_as_a_view(self) -> None:
        layer = self._layer(rank=3, heads=8)
        sink = deepseek_v4.MqaAttentionBase._local_attn_sink_unpadded(layer)
        self.assertEqual(sink.tolist(), list(range(24, 32)))
        self.assertTrue(sink.is_contiguous())
        self.assertEqual(sink.data_ptr(), layer.attn_sink[24:].data_ptr())

    def test_a_changed_head_split_gets_its_own_slice(self) -> None:
        # maybe_use_decode_attn_tp sets another rank and head count for one forward
        layer = self._layer(rank=3, heads=8)
        deepseek_v4.MqaAttentionBase._local_attn_sink_unpadded(layer)
        layer.attn_tp_rank, layer.n_local_heads = 1, 16
        sink = deepseek_v4.MqaAttentionBase._local_attn_sink_unpadded(layer)
        self.assertEqual(sink.tolist(), list(range(16, 32)))


if __name__ == "__main__":
    unittest.main()
