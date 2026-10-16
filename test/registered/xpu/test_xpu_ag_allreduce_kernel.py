"""XPU: the Triton rank-ordered sum of the all-gather all-reduce equals the torch order.

rank_ordered_sum_ runs one Triton kernel on an XPU tensor, and torch ops in the same
order on a CPU tensor: an fp32 sum in rank order, rounded once to the output dtype.
The two must be torch.equal for every world size, dtype and size, including a size
that is not a multiple of the kernel block.
"""

import itertools
import unittest

import torch

from sglang.srt.distributed.device_communicators import xpu_ag_allreduce
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")


@unittest.skipUnless(torch.xpu.is_available(), "needs an XPU device")
class TestXpuRankOrderedSum(CustomTestCase):
    def test_the_kernel_matches_the_torch_order(self) -> None:
        g = torch.Generator().manual_seed(0)
        cases = itertools.product(
            (2, 4, 8),
            (torch.bfloat16, torch.float16, torch.float32),
            ((1, 4096), (2, 4096), (1, 5000)),
        )
        for world, dtype, shape in cases:
            with self.subTest(world=world, dtype=dtype, shape=shape):
                gathered = torch.randn((world,) + shape, generator=g).to(dtype)
                expected = torch.empty(shape, dtype=dtype)
                xpu_ag_allreduce.rank_ordered_sum_(gathered, expected)
                out = torch.empty(shape, dtype=dtype, device="xpu")
                xpu_ag_allreduce.rank_ordered_sum_(gathered.to("xpu"), out)
                torch.xpu.synchronize()
                self.assertTrue(torch.equal(out.cpu(), expected))

    def test_the_kernel_rounds_once_after_fp32(self) -> None:
        gathered = torch.full((8, 1, 4), 2.0**-9, dtype=torch.bfloat16, device="xpu")
        gathered[0] = 1.0
        out = torch.empty(1, 4, dtype=torch.bfloat16, device="xpu")
        xpu_ag_allreduce.rank_ordered_sum_(gathered, out)
        torch.xpu.synchronize()
        self.assertTrue(torch.equal(out.cpu(), torch.full((1, 4), 1.015625).bfloat16()))


if __name__ == "__main__":
    unittest.main()
