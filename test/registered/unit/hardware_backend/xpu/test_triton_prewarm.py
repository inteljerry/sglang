"""Shape plans of the XPU Triton pre-warm (SGLANG_OPT_XPU_TRITON_PREWARM).

The pre-warm launches one dummy call per Triton specialization that a DeepSeek V4
server can reach, so that no kernel compiles or loads after serving starts. These
tests check without a device that the planned launches cover every reachable
specialization key, that the key model agrees with Triton and with the launcher of
the logits kernel, and how the runner hook selects the warmers. The sample server
has 8 running requests, a 262,144-token context, page size 256, chunked prefill of
8,192 tokens and decode graphs at batch sizes 1, 2, 4 and 8.
"""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.xpu import triton_prewarm as tp
from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MAX_RUNNING = 8
CHUNKED_PREFILL = 8192
MAX_PREFILL = 16384
FULL_WIDTH = 262148  # req_to_token width: context 262144 plus the pad
PAGE_SIZE = 256
MAX_PAGES = -(-FULL_WIDTH // PAGE_SIZE)  # 1025
CAPTURE_BS = [1, 2, 4, 8]
CAPTURE_TIERS = [2048, 8192, 32768, FULL_WIDTH]


def bounds(**change: int) -> tp.PrewarmBounds:
    """The bounds of the sample server, with some changed."""
    values = {
        "max_rows": MAX_PREFILL,
        "max_pages": MAX_PAGES,
        "max_reqs": MAX_RUNNING,
        "page_size": PAGE_SIZE,
        "full_width": FULL_WIDTH,
        **change,
    }
    return tp.PrewarmBounds(**values)


def all_logits_keys(b: tp.PrewarmBounds) -> set:
    """Every (PAGES_PER_PROG, int_class(pages)) over the reachable rows x pages grid."""
    rows = torch.arange(1, b.max_rows + 1, dtype=torch.int64)[:, None]
    pages = torch.arange(1, b.max_pages + 1, dtype=torch.int64)[None, :]
    prod = rows * pages
    ok = rows <= b.max_reqs * pages * b.page_size
    q = torch.clamp(prod // tp.MIN_PROGRAMS, min=1)
    ppp = torch.pow(2, torch.floor(torch.log2(q.double()))).to(torch.int64)
    ppp = torch.where(
        prod < tp.MIN_PROGRAMS * tp.MAX_PAGES_PER_PROG, ppp, tp.MAX_PAGES_PER_PROG
    )
    cls = torch.where(pages == 1, 0, torch.where(pages % 16 == 0, 1, 2))
    code = (ppp * 4 + cls.expand_as(ppp))[ok]
    names = {0: "1", 1: "D", 2: "N"}
    return {(int(c) // 4, names[int(c) % 4]) for c in torch.unique(code)}


class TestPagedMqaLogitsPlans(CustomTestCase):
    def test_the_key_model_matches_brute_force(self) -> None:
        for rows in range(1, 70):
            for pages in range(1, 1100, 7):
                ppp, _ = tp.paged_mqa_logits_key(rows, pages)
                prod = rows * pages
                want = tp.MAX_PAGES_PER_PROG
                if prod < tp.MIN_PROGRAMS * want:
                    want = 1 << (max(1, prod // tp.MIN_PROGRAMS).bit_length() - 1)
                self.assertEqual(ppp, want)

    def test_the_plans_cover_every_reachable_key_once(self) -> None:
        for change in (
            {},
            {"max_reqs": 1 << 20},
            {"max_rows": CHUNKED_PREFILL},
            {"max_reqs": 16},
            {"max_rows": MAX_RUNNING},
            {"max_rows": 300, "max_pages": 70, "max_reqs": 1},
            {"max_rows": 1, "max_pages": 1, "max_reqs": 1},
        ):
            with self.subTest(**change):
                b = bounds(**change)
                plans = tp.paged_mqa_logits_plans(b)
                got = {tp.paged_mqa_logits_key(r, p) for r, p in plans}
                self.assertEqual(got, all_logits_keys(b))
                self.assertEqual(len(got), len(plans))
                for r, p in plans:
                    self.assertTrue(1 <= p <= b.max_pages)
                    self.assertTrue(1 <= r <= tp.max_logits_rows(p, b))

    def test_the_decode_graph_shapes_are_covered(self) -> None:
        plans = tp.paged_mqa_logits_plans(bounds())
        covered = {tp.paged_mqa_logits_key(r, p) for r, p in plans}
        for bs in CAPTURE_BS:
            for tier in CAPTURE_TIERS:
                with self.subTest(bs=bs, tier=tier):
                    key = tp.paged_mqa_logits_key(bs, -(-tier // PAGE_SIZE))
                    self.assertIn(key, covered)

    def test_the_sample_server_needs_few_small_launches(self) -> None:
        plans = tp.paged_mqa_logits_plans(bounds())
        # 6 PAGES_PER_PROG values x the classes "D" and "N", and pages == 1, which
        # holds at most 8 x 256 rows and so only PAGES_PER_PROG 1
        self.assertEqual(len(plans), 13)
        biggest = max(tp.paged_mqa_logits_launch_bytes(r, p) for r, p in plans)
        self.assertLess(biggest, 24 << 20)

    def test_pages_per_prog_matches_the_launcher(self) -> None:
        # the package exports a function of the same name, so import the module
        try:
            launcher = importlib.import_module("sgl_kernel.fp8_paged_mqa_logits")
        except ImportError as error:
            self.skipTest(f"sgl_kernel is not installed: {error}")
        self.assertEqual(launcher._MIN_PROGRAMS, tp.MIN_PROGRAMS)
        self.assertEqual(launcher._MAX_PAGES_PER_PROG, tp.MAX_PAGES_PER_PROG)
        for rows in (1, 2, 3, 4, 7, 8, 64, 1280, 8192, 16384):
            for pages in (1, 2, 5, 16, 17, 32, 128, 512, 1025):
                self.assertEqual(
                    launcher._pages_per_prog(rows, pages),
                    tp.pages_per_prog(rows, pages),
                )


class TestTritonIntSpecialization(CustomTestCase):
    def test_int_class_matches_triton(self) -> None:
        # native_specialize_impl is private. A change of its signature raises
        # TypeError, a failure that asks to check the key model again.
        try:
            from triton._C.libtriton import native_specialize_impl
            from triton.backends.compiler import BaseBackend
        except ImportError as error:
            self.skipTest(f"the triton specialization API is not available: {error}")
        for v in (1, 2, 3, 15, 16, 17, 32, 64, 1024, 1025, 8192, 65600):
            kind, key = native_specialize_impl(BaseBackend, v, False, True, True)[:2]
            if v == 1:
                self.assertEqual((kind, tp.int_class(v)), ("constexpr", "1"))
            else:
                self.assertEqual("D" if key == "D" else "N", tp.int_class(v))


class TestOtherKernelPlans(CustomTestCase):
    def test_alloc_bs_upper(self) -> None:
        self.assertEqual(tp.alloc_bs_upper_plans(MAX_RUNNING), [1, 2, 4, 8])
        self.assertEqual(tp.alloc_bs_upper_plans(16), [1, 2, 4, 8, 16])
        self.assertEqual(tp.alloc_bs_upper_plans(5), [1, 2, 4, 8])

    def test_page_table_positions(self) -> None:
        self.assertEqual(tp.page_table_positions_plans(MAX_PAGES), [1, 2, 16])
        self.assertEqual(tp.page_table_positions_plans(1), [1])

    def test_combine_c128_covers_every_key_once(self) -> None:
        max_c128 = FULL_WIDTH // 128
        plans = tp.combine_c128_plans(max_c128)
        covered = {(tp.next_power_of_2(m), tp.int_class(m)) for m in plans}
        for m in range(1, max_c128 + 1):
            self.assertIn((tp.next_power_of_2(m), tp.int_class(m)), covered)
        self.assertEqual(len(plans), len(covered))
        self.assertEqual(len(plans), 20)

    def test_the_first_c4_layer(self) -> None:
        self.assertEqual(tp.first_c4_layer_id([0, 0, 4, 128, 4], 5), 2)
        self.assertIsNone(tp.first_c4_layer_id([0, 128, 4], 2))
        self.assertIsNone(tp.first_c4_layer_id([], 43))


class TestPrewarmBounds(CustomTestCase):
    def test_chunked_prefill_bounds_the_rows(self) -> None:
        b = tp.prewarm_bounds(
            chunked_prefill_size=CHUNKED_PREFILL,
            max_prefill_tokens=MAX_PREFILL,
            max_running_requests=MAX_RUNNING,
            full_width=FULL_WIDTH,
            page_size=PAGE_SIZE,
        )
        self.assertEqual(b, bounds())

    def test_without_chunked_prefill_one_forward_holds_a_prompt(self) -> None:
        for chunked in (None, -1):
            with self.subTest(chunked=chunked):
                b = tp.prewarm_bounds(
                    chunked_prefill_size=chunked,
                    max_prefill_tokens=MAX_PREFILL,
                    max_running_requests=MAX_RUNNING,
                    full_width=FULL_WIDTH,
                    page_size=PAGE_SIZE,
                )
                self.assertEqual(b.max_rows, FULL_WIDTH)


class _RecordingLauncher:
    """Stands in for a triton.jit kernel and records each launch.

    A launch is (name, grid, argument dtypes, arguments, keyword arguments).
    """

    def __init__(self, calls: list, name: str) -> None:
        self.calls = calls
        self.name = name

    def __getitem__(self, grid: tuple):
        def launch(*args: object, **kwargs: object) -> None:
            dtypes = tuple(
                a.dtype if isinstance(a, torch.Tensor) else None for a in args
            )
            self.calls.append((self.name, grid, dtypes, args, kwargs))

        return launch


class TestWarmAllocator(CustomTestCase):
    def test_the_last_loc_dtypes(self) -> None:
        self.assertEqual(
            tp.decode_last_loc_dtypes(torch.int32), [torch.int32, torch.int64]
        )
        # an int64 req_to_token gives no second variant
        self.assertEqual(tp.decode_last_loc_dtypes(torch.int64), [torch.int64])

    def test_both_last_loc_dtypes_run_at_every_bs_upper(self) -> None:
        from sglang.kernels.ops.memory import allocator

        calls = []
        with (
            patch.object(
                allocator, "alloc_decode_kernel", _RecordingLauncher(calls, "decode")
            ),
            patch.object(
                allocator, "alloc_extend_kernel", _RecordingLauncher(calls, "extend")
            ),
        ):
            n = tp._warm_allocator(
                bounds(), tp.decode_last_loc_dtypes(torch.int32), torch.device("cpu")
            )
        bs_uppers = tp.alloc_bs_upper_plans(MAX_RUNNING)
        decode = [(g, d) for name, g, d, _, _ in calls if name == "decode"]
        self.assertEqual(len(decode), 2 * len(bs_uppers))
        for bs_upper in bs_uppers:
            dtypes = {d[1] for g, d in decode if g == (bs_upper,)}
            self.assertEqual(dtypes, {torch.int32, torch.int64})
        extend = [(g, d) for name, g, d, _, _ in calls if name == "extend"]
        self.assertEqual(len(extend), len(bs_uppers))
        for _, d in extend:
            self.assertEqual(d[2], torch.int64)  # last_loc is argument 2
        self.assertEqual(n, 3 * len(bs_uppers))


class TestWarmMetadataKernels(CustomTestCase):
    def test_the_page_table_launches_cover_the_plans_and_both_index_dtypes(
        self,
    ) -> None:
        from sglang.kernels.ops.attention import dsv4_attn_metadata_kernels as kernels

        calls = []
        with patch.object(
            kernels, "_page_table_positions_kernel", _RecordingLauncher(calls, "pt")
        ):
            n = tp._warm_page_table_positions(
                torch.zeros((2, FULL_WIDTH), dtype=torch.int32), bounds()
            )
        self.assertEqual(n, 6)
        self.assertEqual(len(calls), 6)
        self.assertEqual({args[8] for *_, args, _ in calls}, {1, 2, 16})  # num_pages
        self.assertEqual({d[1] for _, _, d, _, _ in calls}, {torch.int32, torch.int64})

    def test_the_combine_launches_cover_swa_c4_and_every_c128_plan(self) -> None:
        from sglang.srt.layers.attention.dsv4 import sparse_prefill_utils

        calls = []
        with patch.object(
            sparse_prefill_utils,
            "_combine_topk_swa_indices_kernel",
            _RecordingLauncher(calls, "combine"),
        ):
            n = tp._warm_combine(FULL_WIDTH // 128, 1024, torch.device("cpu"))
        plans = tp.combine_c128_plans(FULL_WIDTH // 128)
        self.assertEqual(n, 2 + len(plans))
        launches = [(k["COMPRESS_RATIO"], k["top_k"]) for *_, k in calls]
        self.assertEqual(launches[:2], [(1, 0), (4, 1024)])
        self.assertEqual(launches[2:], [(128, m) for m in plans])


class _DsV4Backend:
    """Stands in for DeepseekV4AttnBackend in the isinstance check."""

    c4_topk = 1024  # not the default 512, so the hook must pass the value through


def _runner(**change: object) -> SimpleNamespace:
    runner = SimpleNamespace(
        device="xpu",
        is_draft_worker=False,
        attn_backend=_DsV4Backend(),
        server_args=SimpleNamespace(
            chunked_prefill_size=CHUNKED_PREFILL,
            max_prefill_tokens=MAX_PREFILL,
            enable_deepseek_v4_fp4_indexer=False,
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.empty((MAX_RUNNING + 1, FULL_WIDTH), dtype=torch.int32)
        ),
        token_to_kv_pool=SimpleNamespace(
            get_index_k_with_scale_buffer=lambda layer_id: f"index_k {layer_id}"
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                compress_ratios=[0, 0, 4, 128], num_hidden_layers=4
            )
        ),
        page_size=PAGE_SIZE,
        max_running_requests=MAX_RUNNING,
    )
    for key, value in change.items():
        setattr(runner, key, value)
    return runner


class TestRunnerHook(CustomTestCase):
    def _prewarm(self, runner: SimpleNamespace, on: bool = True) -> dict:
        """Run the hook with recording warmers. Returns {warmer: arguments}."""
        calls = {}

        def record(name: str):
            def warm(*args: object) -> int:
                calls[name] = args
                return 1

            return warm

        with (
            envs.SGLANG_OPT_XPU_TRITON_PREWARM.override(on),
            patch.object(deepseek_v4_backend, "DeepseekV4AttnBackend", _DsV4Backend),
            patch.object(tp, "_warm_allocator", record("alloc")),
            patch.object(tp, "_warm_paged_mqa_logits", record("logits")),
            patch.object(tp, "_warm_page_table_positions", record("page_table")),
            patch.object(tp, "_warm_combine", record("combine")),
            patch.object(torch.xpu, "synchronize", lambda *args: None),
            patch.object(torch.xpu, "empty_cache", lambda *args: None),
        ):
            tp.maybe_prewarm_triton_kernels(runner)
        return calls

    def test_off_by_default(self) -> None:
        self.assertIs(envs.SGLANG_OPT_XPU_TRITON_PREWARM.default, False)
        self.assertEqual(self._prewarm(_runner(), on=False), {})

    def test_a_dsv4_runner_runs_every_warmer_with_its_bounds(self) -> None:
        calls = self._prewarm(_runner())
        self.assertEqual(set(calls), {"alloc", "logits", "page_table", "combine"})
        self.assertEqual(calls["alloc"][0], bounds())
        self.assertEqual(calls["alloc"][1], [torch.int32, torch.int64])
        self.assertEqual(calls["logits"], ("index_k 2", bounds()))
        self.assertEqual(calls["page_table"][1], bounds())
        self.assertEqual(calls["combine"][:2], (FULL_WIDTH // 128, 1024))

    def test_other_backends_and_indexers_skip_their_warmers(self) -> None:
        self.assertEqual(
            set(self._prewarm(_runner(attn_backend=SimpleNamespace(c4_topk=1024)))),
            {"alloc"},
        )
        fp4 = _runner()
        fp4.server_args.enable_deepseek_v4_fp4_indexer = True
        self.assertEqual(set(self._prewarm(fp4)), {"alloc", "page_table", "combine"})

    def test_draft_workers_and_other_devices_skip_the_prewarm(self) -> None:
        for change in ({"is_draft_worker": True}, {"device": "cuda"}):
            with self.subTest(**change):
                self.assertEqual(self._prewarm(_runner(**change)), {})


if __name__ == "__main__":
    unittest.main()
