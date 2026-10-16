"""DeepSeek V4 C4 indexer on XPU: prefill chunks at a long context run in row blocks.

The XPU logits kernel allocates fp32 logits of rows x C4 length. At a long context one
prefill chunk allocates hundreds of MiB, and every further chunk allocates more. Row
blocks keep one allocation near a budget. The tests check the block arithmetic, the
calls that run in blocks, that blocked logits and top-k assemble the same output rows
as one call, that the top-k of a block reads and writes the rows of its block, and the
dispatch in forward_c4_indexer. They use fake kernels that follow the per-row contract
of the real ones.
"""

import sys
import types
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsv4 import indexer
from sglang.srt.layers.attention.dsv4.indexer import (
    C4IndexerBackendMixin,
    PagedIndexerMetadata,
    _c4_indexer_blocked,
    _c4_indexer_min_rows,
    _c4_indexer_row_block,
    _c4_indexer_topk_rows,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20
TOPK = 8
_INDEXER = "sglang.srt.layers.attention.dsv4.indexer"


def fake_logits(calls: list[tuple[int, int]], metadata: object = "meta"):
    """A logits kernel that computes each row on its own, as the XPU Triton kernel does.

    logits[i, j] depends on row i and on the page that row i maps column j to. It is
    zero past the length of the row and past the page table. Each call appends
    (rows, width) to calls.
    """

    def fn(q, kv, weights, seq_lens_2d, page_table, meta, max_len, clean):
        assert meta == metadata and clean is False
        rows = q.shape[0]
        calls.append((rows, max_len))
        cols = torch.arange(max_len)
        covered = cols < page_table.shape[1] * 64
        # the physical page of each column, per row
        page = page_table[:, (cols // 64).clamp(max=page_table.shape[1] - 1)].float()
        scores = q[:, 0, 0, :1].float() * torch.sin(
            cols.float() + weights[:, :1] + page * 0.37
        )
        lens = seq_lens_2d.reshape(-1, 1)
        return torch.where(
            (cols[None, :] < lens) & covered[None, :], scores, torch.zeros(())
        )

    return fn


def fake_topk_kernel(name: str, calls: list[str]):
    """A top-k with the argument list of topk_transform_512 ("v1") or _v2 ("v2").

    Per row, it writes the TOPK largest valid logits: a page index made from each
    column to out, and for v1 the column to the raw output when there is one.
    """

    def run(scores, seq_lens, page_table, out, page_size, extra=None):
        calls.append(name)
        raw = extra if name == "v1" else None
        for i in range(scores.shape[0]):
            n = int(seq_lens[i])
            idx = (
                torch.topk(scores[i, :n], min(TOPK, n)).indices
                if n
                else torch.empty(0, dtype=torch.long)
            )
            out[i].fill_(-1)
            out[i, : idx.numel()] = idx.to(torch.int32) * 3 + 1
            if raw is not None:
                raw[i].fill_(-1)
                raw[i, : idx.numel()] = idx.to(torch.int32)

    return run


def fake_topk(out: torch.Tensor, raw: torch.Tensor, seq_lens: torch.Tensor):
    """The v1 fake top-k on rows [r0, r1), as _c4_indexer_blocked calls topk_fn."""
    kernel = fake_topk_kernel("v1", [])

    def topk(scores, r0=None, r1=None):
        rows = slice(r0, r1)
        kernel(scores, seq_lens[rows], None, out[rows], 64, raw[rows])

    return topk


def make_case(
    rows: int, ctx: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    g = torch.Generator().manual_seed(seed)
    n_c4 = ctx // 4
    pages = (n_c4 + 63) // 64
    q = torch.randn(rows, 1, 4, 8, generator=g)
    weights = torch.randn(rows, 4, generator=g)
    pos = torch.arange(ctx - rows, ctx, dtype=torch.int32)
    seq_lens = (pos + 1) // 4
    page_table = torch.stack(
        [torch.randperm(pages * 2, generator=g)[:pages] for _ in range(rows)]
    ).to(torch.int32)
    return q, weights, seq_lens, page_table, n_c4


def pages_of(max_len: int) -> int:
    return -(-max_len // 64)


def block_of(rows: int, max_len: int, budget: int) -> int | None:
    return _c4_indexer_row_block(
        query_rows=rows,
        max_seq_len=max_len,
        pages=pages_of(max_len),
        budget_bytes=budget,
    )


class TestRowBlock(CustomTestCase):
    def test_short_context_is_not_blocked(self) -> None:
        # 8,192 rows x 4,096 C4 entries x 4 B is exactly the 128 MiB budget: one call, as before
        self.assertIsNone(block_of(8192, 4096, 128 * MIB))
        self.assertIsNone(block_of(1, 65536, 128 * MIB))
        self.assertIsNone(block_of(8192, 65536, 0))
        # logits of exactly the budget run as one call, one byte more run in blocks
        self.assertIsNone(block_of(100, 65536, 100 * 65536 * 4))
        self.assertEqual(block_of(100, 65536, 100 * 65536 * 4 - 1), 64)
        self.assertIsNone(block_of(8192, 0, 128 * MIB))  # no C4 entries at all
        # rows at the floor cannot split: 128 rows keep 32-page programs at 1,008 pages
        self.assertIsNone(block_of(128, 64512, 1 * MIB))

    def test_long_context_blocks_fit_the_budget(self) -> None:
        for rows, max_len in (
            (8192, 18432),
            (8192, 32768),
            (8192, 64512),
            (4096, 64512),
            (3000, 60000),
        ):
            block = block_of(rows, max_len, 128 * MIB)
            self.assertIsNotNone(block)
            self.assertEqual(block % 64, 0)
            self.assertLess(block, rows)
            self.assertLessEqual(block * max_len * 4, 128 * MIB)
        self.assertEqual(block_of(8192, 64512, 128 * MIB), 512)
        self.assertEqual(block_of(8192, 32768, 128 * MIB), 1024)

    def test_a_block_keeps_the_kernels_32_page_programs(self) -> None:
        # the logits kernel runs 32 pages per program only while rows x pages >= 65,536 (_pages_per_prog)
        for rows, max_len, budget in (
            (8192, 64512, 1 * MIB),
            (8192, 16384, 1 * MIB),
            (8192, 10240, 16 * MIB),
        ):
            block = block_of(rows, max_len, budget)
            self.assertIsNotNone(block)
            self.assertGreaterEqual(block * pages_of(max_len), 65536)
        # 1,008 pages: 128 rows keep 32-page programs
        self.assertEqual(block_of(8192, 64512, 1 * MIB), 128)
        self.assertEqual(_c4_indexer_min_rows(1008), 128)
        self.assertEqual(_c4_indexer_min_rows(192), 384)


class TestRowBlockPlan(CustomTestCase):
    """_get_xpu_indexer_row_blocks picks the calls that run in row blocks."""

    def _plan(self, **overrides) -> tuple[int | None, int | None]:
        rows = overrides.get("rows", 8192)
        max_len = overrides.get("max_len", 64512)
        backend = SimpleNamespace(
            dsa_topk_backend=overrides.get("topk", DSATopKBackend.SGL_KERNEL),
            xpu_indexer_logits_budget_bytes=overrides.get("budget", 128 * MIB),
        )
        metadata = SimpleNamespace(
            max_c4_seq_len=max_len,
            c4_page_size=64,
            use_prefill_cuda_graph=overrides.get("prefill_graph", False),
        )
        page_table = torch.zeros(
            rows, overrides.get("width", pages_of(max_len)), dtype=torch.int32
        )
        with (
            envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(
                overrides.get("torch_topk", False)
            ),
            mock.patch(
                f"{_INDEXER}.is_in_tc_piecewise_cuda_graph",
                return_value=overrides.get("piecewise_graph", False),
            ),
            mock.patch(
                f"{_INDEXER}.is_in_breakable_cuda_graph",
                return_value=overrides.get("breakable_graph", False),
            ),
        ):
            return C4IndexerBackendMixin._get_xpu_indexer_row_blocks(
                backend,
                forward_batch=SimpleNamespace(
                    forward_mode=overrides.get("mode", ForwardMode.EXTEND)
                ),
                indexer_metadata=metadata,
                page_table=page_table,
                query_rows=rows,
            )

    def test_prefill_above_the_budget_runs_in_blocks(self) -> None:
        self.assertEqual(self._plan(), (512, 128))
        self.assertEqual(self._plan(mode=ForwardMode.MIXED), (512, 128))

    def test_other_modes_keep_one_call(self) -> None:
        # 1,024 decode rows at 65,536 entries are above the budget and run as one call
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.IDLE,
            ForwardMode.SPLIT_PREFILL,
        ):
            with self.subTest(mode=mode):
                self.assertEqual(
                    self._plan(mode=mode, rows=1024, max_len=65536), (None, None)
                )

    def test_graphs_and_other_top_k_keep_one_call(self) -> None:
        for case in (
            {"prefill_graph": True},
            {"piecewise_graph": True},
            {"breakable_graph": True},
            {"torch_topk": True},
            {"topk": DSATopKBackend.TORCH},
            {"topk": DSATopKBackend.FLASHINFER},
        ):
            with self.subTest(case=case):
                self.assertEqual(self._plan(**case), (None, None))

    def test_small_or_disabled_calls_keep_one_call(self) -> None:
        self.assertEqual(self._plan(max_len=4096), (None, None))
        self.assertEqual(self._plan(budget=0), (None, None))
        self.assertEqual(self._plan(max_len=0), (None, None))

    def test_pages_follow_a_narrow_page_table(self) -> None:
        # the kernel counts min(page table width, ceil(max_len / 64)) pages, so a
        # narrower table raises the floor
        self.assertEqual(self._plan(budget=1 * MIB), (128, 128))
        self.assertEqual(self._plan(budget=1 * MIB, width=256), (256, 256))
        # a partial last page counts: 65,473 entries are 1,024 pages, so 64 rows suffice
        self.assertEqual(self._plan(budget=1 * MIB, max_len=65473), (64, 64))

    def test_the_budget_is_read_once_at_init(self) -> None:
        self.assertEqual(envs.SGLANG_OPT_XPU_INDEXER_LOGITS_BUDGET_MB.default, 128)
        with envs.SGLANG_OPT_XPU_INDEXER_LOGITS_BUDGET_MB.override(64):
            backend = C4IndexerBackendMixin()
        self.assertEqual(backend.xpu_indexer_logits_budget_bytes, 64 * MIB)


class TestBlockedAssembly(CustomTestCase):
    def _run(self, rows: int, ctx: int, budget: int) -> list[int]:
        q, weights, seq_lens, page_table, n_c4 = make_case(rows, ctx)
        ref_out = torch.full((rows, TOPK), -7, dtype=torch.int32)
        ref_raw = torch.full_like(ref_out, -7)
        calls = []
        fn = fake_logits(calls)
        fake_topk(ref_out, ref_raw, seq_lens)(
            fn(q, None, weights, seq_lens[:, None], page_table, "meta", n_c4, False)
        )
        pages = pages_of(n_c4)
        block = _c4_indexer_row_block(
            query_rows=rows, max_seq_len=n_c4, pages=pages, budget_bytes=budget
        )
        self.assertIsNotNone(block)
        floor = _c4_indexer_min_rows(pages)
        out, raw = torch.full_like(ref_out, -7), torch.full_like(ref_raw, -7)
        calls.clear()
        _c4_indexer_blocked(
            logits_fn=fn,
            topk_fn=fake_topk(out, raw, seq_lens),
            q=q,
            kv_cache=None,
            weights=weights,
            seq_lens_2d=seq_lens[:, None],
            page_table=page_table,
            metadata="meta",
            max_seq_len=n_c4,
            query_rows=rows,
            row_block=block,
            min_rows=floor,
        )
        self.assertTrue(torch.equal(out, ref_out))
        self.assertTrue(torch.equal(raw, ref_raw))
        # every row once, at the width of the chunk
        self.assertEqual(sum(r for r, _ in calls), rows)
        self.assertTrue(all(m == n_c4 for _, m in calls))
        # every block, the last too, keeps 32-page programs
        self.assertTrue(all(r * pages >= 65536 for r, _ in calls))
        # a merged tail adds fewer than floor rows
        self.assertTrue(all(r < block + floor for r, _ in calls))
        return [r for r, _ in calls]

    def _spans(self, rows: int, row_block: int, min_rows: int) -> list[tuple[int, int]]:
        """The (r0, r1) row span of each block that _c4_indexer_blocked runs."""
        q, weights, seq_lens, page_table, n_c4 = make_case(rows, 65536)
        spans = []
        _c4_indexer_blocked(
            logits_fn=fake_logits([]),
            topk_fn=lambda scores, r0, r1: spans.append((r0, r1)),
            q=q,
            kv_cache=None,
            weights=weights,
            seq_lens_2d=seq_lens[:, None],
            page_table=page_table,
            metadata="meta",
            max_seq_len=n_c4,
            query_rows=rows,
            row_block=row_block,
            min_rows=min_rows,
        )
        return spans

    def test_block_spans(self) -> None:
        cases = {
            # no floor: the last block ends at the last row
            (150, 64, 0): [(0, 64), (64, 128), (128, 150)],
            # a tail of exactly min_rows stays a block of its own
            (192, 64, 64): [(0, 64), (64, 128), (128, 192)],
            # a shorter tail joins the block before it
            (191, 64, 64): [(0, 64), (64, 191)],
        }
        for (rows, row_block, min_rows), expected in cases.items():
            with self.subTest(rows=rows, min_rows=min_rows):
                self.assertEqual(self._spans(rows, row_block, min_rows), expected)

    def test_a_block_of_logits_is_freed_before_the_next_block(self) -> None:
        q, weights, seq_lens, page_table, n_c4 = make_case(256, 65536)
        refs = []
        live_at_call = []
        logits_of = fake_logits([])

        def logits_fn(*args):
            live_at_call.append(sum(ref() is not None for ref in refs))
            logits = logits_of(*args)
            refs.append(weakref.ref(logits))
            return logits

        _c4_indexer_blocked(
            logits_fn=logits_fn,
            topk_fn=lambda scores, r0, r1: None,
            q=q,
            kv_cache=None,
            weights=weights,
            seq_lens_2d=seq_lens[:, None],
            page_table=page_table,
            metadata="meta",
            max_seq_len=n_c4,
            query_rows=256,
            row_block=64,
            min_rows=64,
        )
        self.assertEqual(live_at_call, [0, 0, 0, 0])

    def test_blocks_assemble_the_unblocked_rows(self) -> None:
        self.assertEqual(self._run(rows=512, ctx=65536, budget=16 * MIB), [256, 256])

    def test_a_short_tail_joins_the_block_before_it(self) -> None:
        # 1,024 pages: floor 64 rows; blocks of 64 leave a 44-row tail, which joins the last block
        self.assertEqual(
            self._run(rows=300, ctx=262144, budget=8 * MIB), [64, 64, 64, 108]
        )


class TestBlockTopk(CustomTestCase):
    """_c4_indexer_topk_rows hands each top-k kernel the rows of its block, in place."""

    ROWS, FIRST, END = 10, 4, 7

    def _run(self, use_v2: bool, with_raw: bool):
        """Return (kernel name, kernel arguments, output rows, raw rows) of one call."""
        seq_lens = torch.arange(self.ROWS, dtype=torch.int32)
        page_table = torch.arange(self.ROWS * 4, dtype=torch.int32).reshape(
            self.ROWS, 4
        )
        out = torch.zeros(self.ROWS, TOPK, dtype=torch.int32)
        raw = torch.zeros(self.ROWS, TOPK, dtype=torch.int32)
        calls = []

        def kernel(name: str):
            def run(scores, lens, table, rows_out, page_size, last):
                calls.append((name, scores, lens, table, page_size, last))
                rows_out.fill_(1)
                if name == "v1" and last is not None:
                    last.fill_(2)

            return run

        with mock.patch.object(
            indexer, "topk_transform_512", kernel("v1")
        ), mock.patch.object(indexer, "topk_transform_512_v2", kernel("v2")):
            _c4_indexer_topk_rows(
                scores := torch.zeros(self.END - self.FIRST, 8),
                self.FIRST,
                self.END,
                c4_seq_lens=seq_lens,
                page_table=page_table,
                c4_sparse_page_indices=out,
                page_size=64,
                topk_metadata="meta",
                raw_indices=raw if with_raw else None,
                use_v2=use_v2,
            )
        self.assertEqual(len(calls), 1)
        name, got_scores, lens, table, page_size, last = calls[0]
        self.assertIs(got_scores, scores)
        self.assertEqual(lens.tolist(), seq_lens[self.FIRST : self.END].tolist())
        self.assertTrue(torch.equal(table, page_table[self.FIRST : self.END]))
        self.assertEqual(page_size, 64)
        return name, last, out, raw

    def _rows_set(self, rows: torch.Tensor, value: int) -> None:
        """Only the rows of the block hold value."""
        expected = torch.zeros_like(rows)
        expected[self.FIRST : self.END] = value
        self.assertTrue(torch.equal(rows, expected))

    def test_v1_writes_its_rows_and_the_raw_rows(self) -> None:
        name, last, out, raw = self._run(use_v2=False, with_raw=True)
        self.assertEqual(name, "v1")
        self.assertIsInstance(last, torch.Tensor)
        self.assertEqual(last.shape, (self.END - self.FIRST, TOPK))
        self._rows_set(out, 1)
        self._rows_set(raw, 2)

    def test_v1_without_raw_indices_passes_none(self) -> None:
        name, last, out, raw = self._run(use_v2=False, with_raw=False)
        self.assertEqual(name, "v1")
        self.assertIsNone(last)
        self._rows_set(out, 1)
        self.assertEqual(int(raw.abs().sum()), 0)

    def test_v2_gets_the_shared_metadata(self) -> None:
        name, last, out, _ = self._run(use_v2=True, with_raw=False)
        self.assertEqual(name, "v2")
        self.assertEqual(last, "meta")
        self._rows_set(out, 1)


class _XpuIndexerBackend(C4IndexerBackendMixin):
    """The state that forward_c4_indexer reads, with fixed indexer projections."""

    def __init__(self, *, forward_metadata, kv_cache, q_indexer, weights) -> None:
        super().__init__()
        self.forward_metadata = forward_metadata
        self.token_to_kv_pool = SimpleNamespace(
            get_index_k_with_scale_buffer=lambda layer_id: kv_cache
        )
        self.hisparse_coordinator = None
        self._projections = (q_indexer, weights)

    def _forward_prepare_normal(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self._projections


class TestForwardC4IndexerDispatch(CustomTestCase):
    """forward_c4_indexer on XPU: the calls that run in blocks, with the right top-k."""

    ROWS, CTX = 512, 65536  # 256 pages of 64 C4 entries: 32 MiB of logits

    def _run(self, **case) -> SimpleNamespace:
        """One forward. Returns the rows of each logits call, the top-k calls, the outputs."""
        q, weights, seq_lens, page_table, _ = make_case(self.ROWS, self.CTX)
        triton_calls, native_calls, topk_calls = [], [], []
        kernels = types.ModuleType("sgl_kernel")
        kernels.fp8_paged_mqa_logits_triton = fake_logits(triton_calls, metadata=None)
        kernels.fp8_paged_mqa_logits = fake_logits(native_calls, metadata=None)
        out = torch.full((self.ROWS, TOPK), -7, dtype=torch.int32)
        raw = torch.full_like(out, -7) if case.get("raw") else None
        with (
            mock.patch(f"{_INDEXER}.is_xpu", return_value=True),
            mock.patch(
                "sglang.srt.layers.attention.dsv4.metadata.is_xpu", return_value=True
            ),
            mock.patch.dict(sys.modules, {"sgl_kernel": kernels}),
            mock.patch(
                f"{_INDEXER}.topk_transform_512", fake_topk_kernel("v1", topk_calls)
            ),
            mock.patch(
                f"{_INDEXER}.topk_transform_512_v2", fake_topk_kernel("v2", topk_calls)
            ),
            mock.patch(f"{_INDEXER}.get_global_indexer_capturer", return_value=None),
            envs.SGLANG_FP8_PAGED_MQA_XPU_KERNEL.override(case.get("native", False)),
            envs.SGLANG_OPT_USE_TOPK_V2.override(case.get("topk_v2", False)),
            envs.SGLANG_OPT_XPU_INDEXER_LOGITS_BUDGET_MB.override(case["budget_mib"]),
        ):
            indexer_metadata = PagedIndexerMetadata(
                page_size=256, page_table=page_table, c4_seq_lens=seq_lens
            )
            backend = _XpuIndexerBackend(
                forward_metadata=SimpleNamespace(
                    indexer_metadata=indexer_metadata,
                    core_metadata=SimpleNamespace(
                        positions=torch.arange(self.ROWS),
                        c4_sparse_page_indices=out,
                        c4_sparse_raw_indices=raw,
                        page_table=indexer_metadata.page_table,
                    ),
                ),
                kv_cache=torch.zeros(
                    2 * page_table.shape[1], 64 * 132, dtype=torch.uint8
                ),
                q_indexer=q[:, 0],
                weights=weights[:, :, None],
            )
            if case.get("nonpaged"):
                # a nonpaged plan computes the logits itself, in one call
                backend._get_nonpaged_indexer_plan = lambda **kwargs: "plan"
                backend._forward_nonpaged_indexer = lambda **kwargs: fake_logits(
                    [], metadata=None
                )(
                    q,
                    None,
                    weights,
                    seq_lens[:, None],
                    page_table,
                    None,
                    self.CTX // 4,
                    False,
                )
            backend.forward_c4_indexer(
                x=torch.zeros(self.ROWS, 1),
                q_lora=torch.zeros(self.ROWS, 1),
                c4_indexer=SimpleNamespace(use_fp4_indexer=False, layer_id=0),
                forward_batch=SimpleNamespace(
                    forward_mode=case.get("mode", ForwardMode.EXTEND)
                ),
            )
        return SimpleNamespace(
            triton=[rows for rows, _ in triton_calls],
            native=[rows for rows, _ in native_calls],
            topk=topk_calls,
            out=out,
            raw=raw,
        )

    def test_a_long_prefill_runs_in_blocks_and_gives_the_rows_of_one_call(self) -> None:
        one = self._run(budget_mib=0)
        blocks = self._run(budget_mib=16)
        self.assertEqual((one.triton, one.topk), ([512], ["v1"]))
        self.assertEqual((blocks.triton, blocks.topk), ([256, 256], ["v1", "v1"]))
        self.assertTrue(torch.equal(blocks.out, one.out))

    def test_excluded_calls_keep_one_call(self) -> None:
        for case, expected in (
            ({"native": True}, ([], [512])),
            ({"nonpaged": True}, ([], [])),
            ({"mode": ForwardMode.DECODE}, ([512], [])),
        ):
            with self.subTest(case=case):
                run = self._run(budget_mib=16, **case)
                self.assertEqual((run.triton, run.native), expected)
                self.assertEqual(run.topk, ["v1"])

    def test_raw_indices_keep_the_v1_top_k_in_blocks(self) -> None:
        # with top-k v2 on, only v1 writes the raw indices that a caller reads
        one = self._run(budget_mib=0, topk_v2=True, raw=True)
        blocks = self._run(budget_mib=16, topk_v2=True, raw=True)
        self.assertEqual(blocks.topk, ["v1", "v1"])
        self.assertTrue(torch.equal(blocks.raw, one.raw))
        self.assertTrue(torch.equal(blocks.out, one.out))
        self.assertEqual(self._run(budget_mib=16, topk_v2=True).topk, ["v2", "v2"])


if __name__ == "__main__":
    unittest.main()
