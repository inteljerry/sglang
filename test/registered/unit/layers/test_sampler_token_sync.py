"""Unit tests for the token-id minimum across TP ranks in srt/layers/sampler.py.

On XPU the sampler takes the minimum of an all_gather, because all_reduce with
ReduceOp.MIN was observed to return the sum through xccl. The tests emulate eight
ranks, and an all_reduce that sums although MIN is requested. They check the
token ids that the sampler leaves in the tensor.
"""

import unittest
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.layers import sampler
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_xpu_ci(est_time=5, suite="stage-a-test-1-gpu-xpu")

WORLD_SIZE = 8
GROUP = object()  # the patched collectives never read the group
HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()

IdsOfRank = Callable[[int], torch.Tensor]


def distinct_ids(rank: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """Rank r holds [93 + r, 7 - r]: the minimum of each column sits on another rank."""
    return torch.tensor([93 + rank, 7 - rank], dtype=dtype)


@contextmanager
def emulated_group(
    ids_of_rank: IdsOfRank, *, reduce_sums: bool
) -> Generator[list[str], None, None]:
    """Patch the collectives of a group of eight ranks. The caller is rank 0.

    all_gather_into_tensor stacks the ids of all ranks. The emulated all_reduce
    returns the sum when reduce_sums is set, and the minimum when not. Yields the
    list of calls.
    """
    calls: list[str] = []

    def all_gather_into_tensor(
        output: torch.Tensor, tensor: torch.Tensor, group: object = None
    ) -> None:
        calls.append("all_gather")
        assert output.shape == (WORLD_SIZE, *tensor.shape)
        for rank in range(WORLD_SIZE):
            output[rank] = ids_of_rank(rank)

    def all_reduce(
        tensor: torch.Tensor, op: object = None, group: object = None
    ) -> None:
        calls.append("all_reduce")
        assert op == dist.ReduceOp.MIN
        stacked = torch.stack([ids_of_rank(r) for r in range(WORLD_SIZE)])
        tensor.copy_(stacked.sum(0) if reduce_sums else stacked.amin(0))

    with patch.multiple(
        dist,
        get_world_size=lambda group=None: WORLD_SIZE,
        all_gather_into_tensor=all_gather_into_tensor,
        all_reduce=all_reduce,
    ):
        yield calls


class TestAllReduceMin(CustomTestCase):
    def _reduce(
        self, ids_of_rank: IdsOfRank, *, xpu: bool, reduce_sums: bool
    ) -> tuple[torch.Tensor, list[str]]:
        ids = ids_of_rank(0).clone()
        with (
            emulated_group(ids_of_rank, reduce_sums=reduce_sums) as calls,
            patch.object(sampler, "_min_needs_all_gather", return_value=xpu),
        ):
            sampler._all_reduce_min_(ids, GROUP)
        return ids, calls

    def test_gather_path_returns_the_minimum_where_all_reduce_sums(self) -> None:
        ids, calls = self._reduce(distinct_ids, xpu=True, reduce_sums=True)
        self.assertEqual(ids.tolist(), [93, 0])
        self.assertEqual(calls, ["all_gather"])

    def test_the_sum_is_what_the_old_call_returns_on_xccl(self) -> None:
        """Negative control: the same ids through all_reduce on a backend that sums."""
        ids, calls = self._reduce(distinct_ids, xpu=False, reduce_sums=True)
        self.assertEqual(ids.tolist(), [772, 28])
        self.assertEqual(calls, ["all_reduce"])

    def test_other_devices_keep_all_reduce(self) -> None:
        ids, calls = self._reduce(distinct_ids, xpu=False, reduce_sums=False)
        self.assertEqual(ids.tolist(), [93, 0])
        self.assertEqual(calls, ["all_reduce"])

    def test_dtype_shape_and_identity_are_kept(self) -> None:
        cases = {
            "int64 pair": lambda r: distinct_ids(r, torch.int64),
            "batch of 4096": lambda r: torch.arange(4096, dtype=torch.int32) + 3 * r,
            "one id": lambda r: torch.tensor([500 + r], dtype=torch.int32),
        }
        for name, ids_of_rank in cases.items():
            with self.subTest(name):
                ids = ids_of_rank(0).clone()
                before = (ids.data_ptr(), ids.dtype, ids.shape)
                with (
                    emulated_group(ids_of_rank, reduce_sums=True),
                    patch.object(sampler, "_min_needs_all_gather", return_value=True),
                ):
                    sampler._all_reduce_min_(ids, GROUP)
                self.assertEqual((ids.data_ptr(), ids.dtype, ids.shape), before)
                want = torch.stack([ids_of_rank(r) for r in range(WORLD_SIZE)]).amin(0)
                self.assertTrue(torch.equal(ids, want))

    def test_only_xpu_uses_the_gather(self) -> None:
        for device_type in ("cpu", "cuda", "npu", "mps", "musa", "hpu", "meta"):
            with self.subTest(device_type):
                self.assertFalse(sampler._min_needs_all_gather(device_type))
        self.assertTrue(sampler._min_needs_all_gather("xpu"))

    def test_a_cpu_tensor_takes_the_all_reduce_path(self) -> None:
        """The path follows the device of the tensor, with no patch of the predicate."""
        ids = distinct_ids(0)
        with emulated_group(distinct_ids, reduce_sums=False) as calls:
            sampler._all_reduce_min_(ids, GROUP)
        self.assertEqual((ids.tolist(), calls), ([93, 0], ["all_reduce"]))

    @unittest.skipUnless(HAS_XPU, "needs an XPU device")
    def test_an_xpu_tensor_takes_the_gather_path(self) -> None:
        def ids_of_rank(rank: int) -> torch.Tensor:
            return distinct_ids(rank).to("xpu")

        ids = ids_of_rank(0).clone()
        with emulated_group(ids_of_rank, reduce_sums=True) as calls:
            sampler._all_reduce_min_(ids, GROUP)
        self.assertEqual((ids.cpu().tolist(), calls), ([93, 0], ["all_gather"]))


class TestSyncTokenIdsAcrossTp(CustomTestCase):
    """With SYNC_TOKEN_IDS_ACROSS_TP off, the sync runs for a grammar request only."""

    def _sync(self, grammars: list) -> tuple[torch.Tensor, list[str]]:
        sampler_like = SimpleNamespace(tp_sync_group=GROUP)
        ids = distinct_ids(0)
        with (
            emulated_group(distinct_ids, reduce_sums=True) as calls,
            patch.object(sampler, "_min_needs_all_gather", return_value=True),
            patch.object(sampler, "SYNC_TOKEN_IDS_ACROSS_TP", False),
        ):
            sampler.Sampler._sync_token_ids_across_tp(
                sampler_like, ids, SimpleNamespace(grammars=grammars)
            )
        return ids, calls

    def test_a_grammar_request_syncs_to_the_minimum(self) -> None:
        ids, calls = self._sync([object()])
        self.assertEqual((ids.tolist(), calls), ([93, 0], ["all_gather"]))

    def test_without_a_grammar_the_ids_are_left_alone(self) -> None:
        ids, calls = self._sync([])
        self.assertEqual((ids.tolist(), calls), ([93, 7], []))


if __name__ == "__main__":
    unittest.main()
