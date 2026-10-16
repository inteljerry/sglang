"""TP all-reduce as one all-gather and one local fp32 sum, for small XPU decode batches.

A ring all-reduce over W ranks takes 2 (W - 1) dependent steps, and a ring
all-gather takes W - 1. A decode batch of 1 or 2 tokens moves a few KiB per rank,
so the latency of the steps sets the time, not the bytes. With
SGLANG_OPT_XPU_AG_ALLREDUCE_MAX_TOKENS = N > 0, the all-reduce of a [T, ...] tensor
with 1 <= T <= N is an all-gather of the input of every rank into [W, T, ...] and
one Triton kernel that sums the W slices into the input tensor. use_all_gather_sum
also requires more than one rank, a contiguous bf16, fp16 or fp32 tensor of at
least two dimensions and at most 1 MiB per rank, and no torch.compile. Callers keep
the in-place contract of GroupCoordinator.all_reduce. A larger T keeps the
all-reduce, because the all-gather moves W times the bytes.

Summation order: for every element, the kernel adds the W gathered values in rank
order, ((x_0 + x_1) + x_2) + ... + x_{W-1}, in fp32, and rounds once to the input
dtype (round to nearest even). Every rank sums the same gathered bytes in the same
order, so all ranks hold bit-identical results, as after a ring all-reduce. The
result can differ in the last bits from the all-reduce of the collective library,
which has its own order and precision. So the switch changes numerics.

The path allocates the gathered buffer, calls all_gather_into_tensor of the
group and launches one local sum kernel. It reads no device value on the host.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch

from sglang.srt.environ import envs

# The all-gather path applies to 1 to this many tokens. 0 disables it.
MAX_TOKENS = envs.SGLANG_OPT_XPU_AG_ALLREDUCE_MAX_TOKENS.get()
# A cap on the input size, because the all-gather moves W times the bytes.
MAX_BYTES_PER_RANK = 1 << 20
_SUM_BLOCK = 1024
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def use_all_gather_sum(input_: torch.Tensor, world_size: int) -> bool:
    """True when this all-reduce takes the all-gather path (see the module doc)."""
    return (
        MAX_TOKENS > 0
        and world_size > 1
        and input_.dim() >= 2
        and 1 <= input_.shape[0] <= MAX_TOKENS
        and input_.dtype in _DTYPES
        and input_.is_contiguous()
        and input_.numel() * input_.element_size() <= MAX_BYTES_PER_RANK
        and not torch.compiler.is_compiling()
    )


@functools.cache
def _sum_kernel() -> Callable:
    """The Triton kernel of rank_ordered_sum_, built at its first use."""
    import triton
    import triton.language as tl

    @triton.jit
    def rank_ordered_sum_kernel(
        src_ptr, dst_ptr, n, WORLD: tl.constexpr, BLOCK: tl.constexpr
    ):
        # dst[i] = ((src[0, i] + src[1, i]) + ...) + src[WORLD - 1, i] in fp32
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        acc = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        for r in tl.static_range(1, WORLD):
            x = tl.load(src_ptr + r * n + offs, mask=mask, other=0.0)
            acc += x.to(tl.float32)
        tl.store(dst_ptr + offs, acc.to(dst_ptr.dtype.element_ty), mask=mask)

    return rank_ordered_sum_kernel


def rank_ordered_sum_(gathered: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out = the sum over dim 0 of gathered [W, *out.shape], in rank order in fp32.

    The sum is rounded once to out.dtype. On a GPU it is one Triton kernel, and on
    other devices the same order in torch ops.
    """
    assert gathered.shape[1:] == out.shape and gathered.is_contiguous()
    if gathered.device.type in ("xpu", "cuda"):
        n = out.numel()
        grid = ((n + _SUM_BLOCK - 1) // _SUM_BLOCK,)
        _sum_kernel()[grid](gathered, out, n, WORLD=gathered.shape[0], BLOCK=_SUM_BLOCK)
        return out
    acc = gathered[0].to(torch.float32)
    for r in range(1, gathered.shape[0]):
        acc = acc + gathered[r].to(torch.float32)
    out.copy_(acc)
    return out


def all_gather_sum_(
    input_: torch.Tensor,
    world_size: int,
    all_gather_into_tensor: Callable[[torch.Tensor, torch.Tensor], None],
) -> torch.Tensor:
    """In-place all-reduce (sum) of input_ as one all-gather and rank_ordered_sum_.

    all_gather_into_tensor(output [W * T, ...], input [T, ...]) gathers over the
    group of the caller.
    """
    gathered = torch.empty(
        (world_size,) + tuple(input_.shape), dtype=input_.dtype, device=input_.device
    )
    all_gather_into_tensor(
        gathered.view((world_size * input_.shape[0],) + tuple(input_.shape[1:])),
        input_,
    )
    return rank_ordered_sum_(gathered, input_)
