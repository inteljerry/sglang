"""XPU TP all-reduce of small decode batches as one all-gather and a rank-ordered sum.

With SGLANG_OPT_XPU_AG_ALLREDUCE_MAX_TOKENS = N > 0, GroupCoordinator.all_reduce on
XPU runs the all-reduce of a contiguous float [T, ...] tensor with 1 <= T <= N as an
all-gather into [W, T, ...] and one fp32 sum in rank order. The tests check the
predicate, the order and the single rounding of the sum, the in-place result, and
the routing in GroupCoordinator.all_reduce. The routing test uses meta tensors,
which skip the CPU branch of all_reduce.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators import xpu_ag_allreduce
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

WORLD, HIDDEN = 8, 4096


class TestUseAllGatherSum(CustomTestCase):
    def setUp(self) -> None:
        patcher = patch.object(xpu_ag_allreduce, "MAX_TOKENS", 2)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_small_float_batches_take_the_all_gather(self) -> None:
        for tokens in (1, 2):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                with self.subTest(tokens=tokens, dtype=dtype):
                    x = torch.zeros(tokens, HIDDEN, dtype=dtype)
                    self.assertTrue(xpu_ag_allreduce.use_all_gather_sum(x, WORLD))
        # exactly 1 MiB per rank
        x = torch.zeros(1, 1 << 19, dtype=torch.bfloat16)
        self.assertTrue(xpu_ag_allreduce.use_all_gather_sum(x, WORLD))

    def test_other_calls_keep_the_all_reduce(self) -> None:
        cases = {
            "one rank": (torch.zeros(1, HIDDEN), 1),
            # 2 elements, within the token limit: only the dim check rejects it
            "1-D tensor": (torch.zeros(2), WORLD),
            "no tokens": (torch.zeros(0, HIDDEN), WORLD),
            "above the limit": (torch.zeros(3, HIDDEN), WORLD),
            "int tensor": (torch.zeros(1, HIDDEN, dtype=torch.int32), WORLD),
            "strided tensor": (torch.zeros(HIDDEN, 2).t(), WORLD),
            "above 1 MiB": (torch.zeros(1, (1 << 19) + 1, dtype=torch.bfloat16), WORLD),
        }
        for name, (x, world) in cases.items():
            with self.subTest(name):
                self.assertFalse(xpu_ag_allreduce.use_all_gather_sum(x, world))
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            x = torch.zeros(1, HIDDEN)
            self.assertFalse(xpu_ag_allreduce.use_all_gather_sum(x, WORLD))

    def test_the_path_is_off_by_default(self) -> None:
        self.assertEqual(envs.SGLANG_OPT_XPU_AG_ALLREDUCE_MAX_TOKENS.default, 0)
        with patch.object(xpu_ag_allreduce, "MAX_TOKENS", 0):
            x = torch.zeros(1, HIDDEN)
            self.assertFalse(xpu_ag_allreduce.use_all_gather_sum(x, WORLD))


class TestRankOrderedSum(CustomTestCase):
    def test_the_sum_rounds_once_after_fp32(self) -> None:
        # 1 + 7 * 2**-9 is 1.013671875, which rounds to the bf16 value 1.015625.
        # A bf16 sum would round 1 + 2**-9 back to 1 at every step.
        gathered = torch.full((WORLD, 1, 4), 2.0**-9, dtype=torch.bfloat16)
        gathered[0] = 1.0
        out = torch.empty(1, 4, dtype=torch.bfloat16)
        xpu_ag_allreduce.rank_ordered_sum_(gathered, out)
        self.assertTrue(torch.equal(out, torch.full_like(out, 1.015625)))

    def test_the_sum_adds_in_rank_order(self) -> None:
        # (1e8 + -1e8) + 1 is 1 in fp32, and (1 + -1e8) + 1e8 is 0
        gathered = torch.tensor([[[1e8]], [[-1e8]], [[1.0]]], dtype=torch.float32)
        out = torch.empty(1, 1)
        xpu_ag_allreduce.rank_ordered_sum_(gathered, out)
        self.assertEqual(out.item(), 1.0)


class TestAllGatherSum(CustomTestCase):
    def test_the_input_holds_the_sum_of_every_rank(self) -> None:
        g = torch.Generator().manual_seed(0)
        ranks = torch.randn(WORLD, 2, HIDDEN, generator=g).to(torch.bfloat16)
        x = ranks[3].clone()
        calls = []

        def all_gather_into_tensor(out: torch.Tensor, inp: torch.Tensor) -> None:
            calls.append((tuple(out.shape), inp))
            out.copy_(ranks.reshape(out.shape))

        result = xpu_ag_allreduce.all_gather_sum_(x, WORLD, all_gather_into_tensor)
        self.assertIs(result, x)
        self.assertEqual(calls[0][0], (WORLD * 2, HIDDEN))
        self.assertIs(calls[0][1], x)
        expected = ranks[0].float()
        for r in range(1, WORLD):
            expected = expected + ranks[r].float()
        self.assertTrue(torch.equal(x, expected.to(torch.bfloat16)))


class TestAllReduceRouting(CustomTestCase):
    def _all_reduce(self, tokens: int, max_tokens: int) -> list[str]:
        """GroupCoordinator.all_reduce on XPU with a meta tensor. Returns the path."""
        path = []
        group = SimpleNamespace(
            world_size=WORLD,
            hpu_communicator=None,
            xpu_communicator=SimpleNamespace(disabled=False),
            unique_name="tp",
            all_gather_into_tensor=lambda out, inp: path.append("all_gather"),
        )
        x = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device="meta")
        with (
            patch.object(xpu_ag_allreduce, "MAX_TOKENS", max_tokens),
            patch.object(
                parallel_state,
                "inplace_all_reduce",
                lambda tensor, group_name: path.append("all_reduce"),
            ),
        ):
            result = parallel_state.GroupCoordinator.all_reduce(group, x)
        self.assertIs(result, x)
        return path

    def test_small_batches_all_gather_and_others_all_reduce(self) -> None:
        self.assertEqual(self._all_reduce(tokens=1, max_tokens=2), ["all_gather"])
        self.assertEqual(self._all_reduce(tokens=2, max_tokens=2), ["all_gather"])
        self.assertEqual(self._all_reduce(tokens=3, max_tokens=2), ["all_reduce"])
        self.assertEqual(self._all_reduce(tokens=1, max_tokens=0), ["all_reduce"])


if __name__ == "__main__":
    unittest.main()
