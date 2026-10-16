"""DeepSeek V4 C4 indexer on XPU: the early-exit Triton logits equal the original.

fp8_paged_mqa_logits_triton_early_exit skips each page whose first position is at
or past the length of its row. sgl_kernel.fp8_paged_mqa_logits_triton stores zeros
there. So the two kernels must give torch.equal logits for every row length: 0, a
partial first page, a page boundary, a partial last page and the full width. The
cases cover the two PAGES_PER_PROG regimes of the launcher, a max_seq_len that is
not a multiple of the page, and a page table narrower than max_seq_len.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

# DeepSeek-V4-Flash indexer: 64 heads of 128, C4 pages of 64 entries
HEADS, HEAD_DIM, PAGE = 64, 128, 64


def make_inputs(
    seq_lens: list[int], width: int, num_pages: int, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    """q, the paged indexer cache, weights, seq_lens and a page table, on the XPU."""
    g = torch.Generator().manual_seed(seed)
    rows = len(seq_lens)
    q = torch.randn(rows, 1, HEADS, HEAD_DIM, generator=g).to(torch.float8_e4m3fn)
    vals = (
        torch.randn(num_pages, PAGE * HEAD_DIM, generator=g)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    scales = (
        (torch.rand(num_pages, PAGE, generator=g) + 0.5)
        .view(torch.uint8)
        .reshape(num_pages, PAGE * 4)
    )
    cache = torch.cat([vals, scales], dim=1).reshape(num_pages, PAGE, 1, HEAD_DIM + 4)
    weights = torch.randn(rows, HEADS, generator=g)
    lens = torch.tensor(seq_lens, dtype=torch.int32)[:, None]
    page_table = torch.randint(
        0, num_pages, (rows, width), generator=g, dtype=torch.int32
    )
    return tuple(x.to("xpu") for x in (q, cache, weights, lens, page_table))


@unittest.skipUnless(torch.xpu.is_available(), "needs an XPU device")
class TestDsv4IndexerEarlyExit(CustomTestCase):
    def _check(self, seq_lens: list[int], width: int, max_seq_len: int) -> None:
        from sgl_kernel import fp8_paged_mqa_logits_triton

        from sglang.kernels.ops.attention.dsv4.xpu_paged_mqa_logits import (
            fp8_paged_mqa_logits_triton_early_exit,
        )

        q, cache, weights, lens, page_table = make_inputs(
            seq_lens, width, num_pages=2 * width
        )
        args = (q, cache, weights, lens, page_table, None, max_seq_len, False)
        expected = fp8_paged_mqa_logits_triton(*args)
        actual = fp8_paged_mqa_logits_triton_early_exit(*args)
        torch.xpu.synchronize()
        self.assertTrue(torch.equal(actual, expected))
        # the pages past each row store +0.0, as the original kernel does
        positions = torch.arange(max_seq_len, device="xpu")
        past = positions[None, :] >= lens
        self.assertEqual(int(actual[past].ne(0).sum()), 0)

    def test_decode_rows_in_a_wide_graph(self) -> None:
        # 8 rows x 128 pages: fewer than 65,536 row-pages, so PAGES_PER_PROG is 1
        self._check([0, 1, 63, 64, 65, 2047, 4096, 8192], width=128, max_seq_len=8192)

    def test_prefill_chunk_with_32_page_programs(self) -> None:
        # 512 rows x 128 pages and 512 rows x 512 pages: at least 65,536
        # row-pages, so 32 pages per program
        lens = list(range(32768 - 512 + 1, 32768 + 1))
        self._check([n // 4 for n in lens], width=128, max_seq_len=8192)
        self._check(lens, width=512, max_seq_len=32768)

    def test_partial_last_page_and_narrow_page_table(self) -> None:
        # max_seq_len 8,001 ends one entry into page 125 (from 0), so the store of
        # that page is masked. A 100-page table leaves a zero tail.
        self._check([10, 6400, 8000, 8001], width=126, max_seq_len=8001)
        self._check([10, 6400, 8000, 8001], width=100, max_seq_len=8001)


if __name__ == "__main__":
    unittest.main()
