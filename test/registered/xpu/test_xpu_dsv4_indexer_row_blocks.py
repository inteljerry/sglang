"""DeepSeek V4 C4 indexer on XPU: row-blocked prefill logits and top-k are bit-identical.

A prefill chunk at a long context runs the XPU logits kernel and the top-k in row
blocks, so one fp32 logits allocation stays near a budget. Both kernels compute each
row on its own, and every block keeps the C4 width of the chunk. So the page and raw
indices must be torch.equal to one call over all rows. The test runs the kernels
(sgl_kernel fp8_paged_mqa_logits_triton, topk_transform_512 with the raw output as
prefill uses it, topk_transform_512_v2 without it) on one 8,192-row chunk at 32K, 48K
and 128K context, and checks the allocation of each blocked call.
"""

import unittest

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    _c4_indexer_blocked,
    _c4_indexer_min_rows,
    _c4_indexer_row_block,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=60, suite="stage-b-test-1-gpu-xpu")

# DeepSeek-V4-Flash indexer: 64 heads of 128, top 512, C4 pages of 64 entries
HEADS, HEAD_DIM, TOPK, PAGE = 64, 128, 512, 64
MIB = 1 << 20


def make_chunk(
    ctx: int, rows: int, device: str, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One prefill chunk of `rows` query rows ending at context `ctx`: row i sits at position ctx - rows + i."""
    g = torch.Generator().manual_seed(seed)
    n_c4 = ctx // 4
    pages = (n_c4 + PAGE - 1) // PAGE
    q = (
        torch.randn(rows, 1, HEADS, HEAD_DIM, generator=g)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    vals = (
        torch.randn(pages, PAGE * HEAD_DIM, generator=g)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    scales = (
        (torch.rand(pages, PAGE, generator=g) + 0.5)
        .view(torch.uint8)
        .reshape(pages, PAGE * 4)
    )
    cache = (
        torch.cat([vals, scales], dim=1)
        .reshape(pages, PAGE, 1, HEAD_DIM + 4)
        .to(device)
    )
    weights = torch.randn(rows, HEADS, generator=g).to(device)
    pos = torch.arange(ctx - rows, ctx, dtype=torch.int32)
    seq_lens = ((pos + 1) // 4).to(device)
    # a distinct page mapping per row, so a wrong row slice of the page table changes the output
    page_table = (
        torch.stack([torch.randperm(pages, generator=g) for _ in range(rows)])
        .to(torch.int32)
        .to(device)
    )
    return q, cache, weights, seq_lens, page_table, n_c4


@unittest.skipUnless(torch.xpu.is_available(), "needs an XPU device")
class TestDsv4IndexerRowBlocks(CustomTestCase):
    def _check(self, ctx: int, rows: int, budget_mib: int) -> None:
        from sgl_kernel import (
            fp8_paged_mqa_logits_triton,
            topk_transform_512,
            topk_transform_512_v2,
        )

        q, cache, weights, seq_lens, page_table, n_c4 = make_chunk(ctx, rows, "xpu")
        width = TOPK
        sizes, kept = [], []

        def logits_fn(*args):
            out = fp8_paged_mqa_logits_triton(*args)
            sizes.append(out.numel() * 4)
            kept.append(out.clone())
            return out

        def topk_closure(out, raw):
            meta = torch.empty(0, dtype=torch.int32, device="xpu")

            def run(scores, r0=None, r1=None):
                sl = slice(r0, r1)
                if raw is None:
                    topk_transform_512_v2(
                        scores, seq_lens[sl], page_table[sl], out[sl], PAGE, meta
                    )
                else:
                    topk_transform_512(
                        scores, seq_lens[sl], page_table[sl], out[sl], PAGE, raw[sl]
                    )

            return run

        pages = -(-n_c4 // PAGE)
        block = _c4_indexer_row_block(
            query_rows=rows,
            max_seq_len=n_c4,
            pages=pages,
            budget_bytes=budget_mib * MIB,
        )
        self.assertIsNotNone(block)
        floor = _c4_indexer_min_rows(pages)
        for with_raw in (True, False):
            ref = torch.full((rows, width), -7, dtype=torch.int32, device="xpu")
            ref_raw = torch.full_like(ref, -7) if with_raw else None
            logits = logits_fn(
                q, cache, weights, seq_lens[:, None], page_table, None, n_c4, False
            )
            topk_closure(ref, ref_raw)(logits)
            out = torch.full_like(ref, -7)
            raw = torch.full_like(ref, -7) if with_raw else None
            sizes.clear()
            kept.clear()
            _c4_indexer_blocked(
                logits_fn=logits_fn,
                topk_fn=topk_closure(out, raw),
                q=q,
                kv_cache=cache,
                weights=weights,
                seq_lens_2d=seq_lens[:, None],
                page_table=page_table,
                metadata=None,
                max_seq_len=n_c4,
                query_rows=rows,
                row_block=block,
                min_rows=floor,
            )
            torch.xpu.synchronize()
            # the logits of every block, in row order
            self.assertTrue(
                torch.equal(torch.cat(kept), logits), f"logits differ at ctx {ctx}"
            )
            del logits
            self.assertTrue(
                torch.equal(out, ref),
                f"page indices differ at ctx {ctx}, raw={with_raw}",
            )
            if with_raw:
                self.assertTrue(
                    torch.equal(raw, ref_raw), f"raw indices differ at ctx {ctx}"
                )
            rows_per_call = [sz // (n_c4 * 4) for sz in sizes]
            self.assertEqual(sum(rows_per_call), rows)
            # 32-page programs in every block
            self.assertTrue(
                all(r * pages >= 65536 for r in rows_per_call), rows_per_call
            )
            self.assertTrue(
                all(r < block + floor for r in rows_per_call), rows_per_call
            )

    def test_32k_context_small_budget(self) -> None:
        # 256 MiB of logits in blocks of 2,048 rows
        self._check(ctx=32768, rows=8192, budget_mib=64)

    def test_128k_context_default_budget(self) -> None:
        # 1 GiB of logits in blocks of 1,024 rows
        self._check(ctx=131072, rows=8192, budget_mib=128)

    def test_a_short_tail_joins_the_last_block(self) -> None:
        # blocks of 2,688 rows, and the 128-row tail joins the last block
        self._check(ctx=49152, rows=8192, budget_mib=128)


if __name__ == "__main__":
    unittest.main()
