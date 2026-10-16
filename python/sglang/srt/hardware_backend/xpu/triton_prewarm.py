"""Pre-warm the Triton kernels that DeepSeek V4 on XPU compiles while it serves.

Triton compiles a kernel specialization at its first launch and loads it onto the
device at the same time. On an XPU server several first launches happen after
serving starts: the C4 indexer logits (_paged_mqa_logits_kernel) at the first
prefill or eager decode of a new shape class, and the allocator, page-table and
index-combine kernels at their first batch of a new size. Each compile stalls the
engine on every rank, and each device load needs free device memory. With
SGLANG_OPT_XPU_TRITON_PREWARM, the model runner launches once, on dummy inputs,
every specialization of these four kernel families that the server can reach. It
does so before it captures the decode graphs, and in eager mode too.

A specialization is keyed by the constexpr values, by the dtype of every tensor
argument and, for every int argument not in do_not_specialize, by whether it equals
1 or is a multiple of 16 (triton.backends.compiler.BaseBackend). Pointer alignment
also keys, and every buffer of the pre-warm is a whole allocation. The *_plans
functions list, for each kernel, the distinct keys that its call sites
can produce and one small launch per key. They need no device, so unit tests check
them on CPU.

Triton writes compiled kernels under TRITON_CACHE_DIR. A server that mounts a host
directory there loads the pre-warmed kernels from disk after a restart, and the
pre-warm then costs the device loads only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# fp8_paged_mqa_logits_triton (sgl_kernel/fp8_paged_mqa_logits.py) picks
# PAGES_PER_PROG from these two constants.
MIN_PROGRAMS = 2048
MAX_PAGES_PER_PROG = 32
# The indexer K cache holds C4 entries 64 to a page (page_size 256 // 4).
C4_BLOCK = 64
INDEXER_HEADS = 64
INDEXER_HEAD_DIM = 128


class PrewarmBounds(msgspec.Struct, frozen=True):
    """Bounds of the shapes that the server can launch."""

    max_rows: int  # most query rows of one forward
    max_pages: int  # widest page table, in pages
    max_reqs: int  # most running requests
    page_size: int
    full_width: int  # width of req_to_token, in tokens


def prewarm_bounds(
    *,
    chunked_prefill_size: int | None,
    max_prefill_tokens: int,
    max_running_requests: int,
    full_width: int,
    page_size: int,
) -> PrewarmBounds:
    """The bounds of a server from its arguments and the width of req_to_token."""
    candidates = [chunked_prefill_size or 0, max_prefill_tokens, max_running_requests]
    if candidates[0] <= 0:
        # Without chunked prefill, one forward can hold a whole prompt.
        candidates.append(full_width)
    return PrewarmBounds(
        max_rows=max(candidates),
        max_pages=-(-full_width // page_size),
        max_reqs=max_running_requests,
        page_size=page_size,
        full_width=full_width,
    )


def int_class(value: int) -> str:
    """How Triton specializes an int argument: "1" (a constexpr), "D" (a multiple of
    16) or "N" (neither)."""
    if value == 1:
        return "1"
    return "D" if value % 16 == 0 else "N"


def next_power_of_2(n: int) -> int:
    """The smallest power of 2 that is at least n, and 1 for n <= 0."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


# ---- _paged_mqa_logits_kernel ------------------------------------------------


def pages_per_prog(rows: int, pages: int) -> int:
    """PAGES_PER_PROG of fp8_paged_mqa_logits_triton for [rows, pages]."""
    ppp = MAX_PAGES_PER_PROG
    if rows * pages < MIN_PROGRAMS * ppp:
        ppp = max(1, (rows * pages) // MIN_PROGRAMS)
        ppp = 1 << (ppp.bit_length() - 1)
    return ppp


def paged_mqa_logits_key(rows: int, pages: int) -> tuple[int, str]:
    """Specialization key of one indexer logits launch.

    The indexer passes a contiguous int32 page table [rows, pages] and
    max_seq_len = pages * 64 (PagedIndexerMetadata.max_c4_seq_len). The kernel
    arguments that change between launches are then num_pages_req = pages,
    page_table_stride_b = pages, max_seq_len = logits_stride_b = 64 * pages
    (always a multiple of 16) and the constexpr PAGES_PER_PROG. So the key is
    (PAGES_PER_PROG, int_class(pages)). The other arguments (head count, the
    strides of q, weight and the K cache, the page count of the K cache) are the
    same for every launch of a server.
    """
    return pages_per_prog(rows, pages), int_class(pages)


def _ppp_product_range(ppp: int) -> tuple[int, int | None]:
    """Closed range of rows * pages that gives PAGES_PER_PROG == ppp."""
    if ppp == 1:
        return 1, 2 * MIN_PROGRAMS - 1
    if ppp == MAX_PAGES_PER_PROG:
        return MIN_PROGRAMS * MAX_PAGES_PER_PROG, None
    return MIN_PROGRAMS * ppp, 2 * MIN_PROGRAMS * ppp - 1


def paged_mqa_logits_launch_bytes(rows: int, pages: int) -> int:
    """Device bytes of one pre-warm launch.

    The launch allocates fp8 q [rows, 64, 128], fp32 weights [rows, 64], int32
    lengths [rows], an int32 page table [rows, pages] and fp32 logits
    [rows, 64 * pages].
    """
    per_row = INDEXER_HEADS * INDEXER_HEAD_DIM + 4 * INDEXER_HEADS + 4
    return rows * per_row + rows * pages * (4 + 4 * C4_BLOCK)


def max_logits_rows(pages: int, bounds: PrewarmBounds) -> int:
    """Most query rows that one launch can have at this page-table width.

    A forward holds at most max_rows tokens. Its max_reqs requests are each at most
    pages * page_size tokens long, so they have at most that many new tokens.
    """
    return min(bounds.max_rows, bounds.max_reqs * pages * bounds.page_size)


def paged_mqa_logits_plans(bounds: PrewarmBounds) -> list[tuple[int, int]]:
    """One (rows, pages) launch per key that the server can reach.

    The range is 1 <= pages <= max_pages and 1 <= rows <= max_logits_rows(pages).
    For each key, the plan takes the launch with the fewest device bytes
    (paged_mqa_logits_launch_bytes), and on a tie the one with fewer rows. rows
    is the count of query rows: the batch size of a decode graph or an eager
    decode, or the tokens of a prefill forward. pages is the page-table width,
    ceil(longest sequence / page_size), up to the full-context width.
    """
    best: dict[tuple[int, str], tuple[int, int, int]] = {}
    ppp = 1
    while ppp <= MAX_PAGES_PER_PROG:
        lo, hi = _ppp_product_range(ppp)
        for pages in range(1, bounds.max_pages + 1):
            rows = max(1, -(-lo // pages))
            if rows > max_logits_rows(pages, bounds):
                continue
            if hi is not None and rows * pages > hi:
                continue
            key = (ppp, int_class(pages))
            candidate = (paged_mqa_logits_launch_bytes(rows, pages), rows, pages)
            if key not in best or candidate < best[key]:
                best[key] = candidate
        ppp *= 2
    return sorted((rows, pages) for _, rows, pages in best.values())


# ---- allocator, page-table and combine kernels ------------------------------


def alloc_bs_upper_plans(max_running_requests: int) -> list[int]:
    """bs_upper values of alloc_decode_kernel and alloc_extend_kernel.

    bs_upper is next_power_of_2(bs) for every batch size up to max_running_requests.
    """
    return sorted({next_power_of_2(bs) for bs in range(1, max_running_requests + 1)})


def decode_last_loc_dtypes(req_to_token_dtype: torch.dtype) -> list[torch.dtype]:
    """The last_loc dtypes of alloc_decode_kernel at decode.

    SWATokenToKVPoolAllocator.alloc_decode passes the full-attention allocator the
    raw req_to_token slice, so the dtype of req_to_token, and the SWA allocator the
    output of translate_loc_from_full_to_swa, which is int64. Triton keys a
    specialization by the dtype of a pointer argument, so each dtype is its own
    variant. The last_loc of alloc_extend_kernel is always int64.
    """
    return list(dict.fromkeys([req_to_token_dtype, torch.int64]))


def page_table_positions_plans(max_pages: int) -> list[int]:
    """num_pages values that cover every int_class of _page_table_positions_kernel.

    num_pages is ceil(max_seq_len / page_size), from 1 to max_pages.
    """
    plans: dict[str, int] = {}
    for pages in (1, 2, 16):
        if pages <= max_pages:
            plans.setdefault(int_class(pages), pages)
    return sorted(plans.values())


def combine_c128_plans(max_c128: int) -> list[int]:
    """c128_max values that cover every key of the C128 call of the combine kernel.

    topk_indices of _combine_topk_swa_indices_kernel is [tokens, c128_max], so its
    stride is c128_max (int_class) and PADDED_TOP_K is next_power_of_2(c128_max).
    The output stride ceil_align(c128_max + 128, 128) is always a multiple of 16.
    c128_max = max(longest sequence // 128, 1) runs from 1 to max_c128.
    """
    best: dict[tuple[int, str], int] = {}
    for m in range(1, max_c128 + 1):
        best.setdefault((next_power_of_2(m), int_class(m)), m)
    return sorted(best.values())


def first_c4_layer_id(compress_ratios: Sequence[int], num_layers: int) -> int | None:
    """The first layer with a C4 indexer, or None."""
    for layer_id, ratio in enumerate(compress_ratios[:num_layers]):
        if ratio == 4:
            return layer_id
    return None


def indexer_uses_triton_logits(fp4_indexer: bool) -> bool:
    """True when the XPU C4 indexer calls fp8_paged_mqa_logits_triton."""
    return not (
        fp4_indexer
        or envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
        or envs.SGLANG_OPT_USE_AITER_INDEXER.get()
        or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        or envs.SGLANG_FP8_PAGED_MQA_XPU_KERNEL.get()
    )


# ---- launches ------------------------------------------------------------------


def _warm_paged_mqa_logits(index_k: torch.Tensor, bounds: PrewarmBounds) -> int:
    """One launch per plan, on the indexer K cache [pages, 64 * 132] of one layer."""
    from sgl_kernel import fp8_paged_mqa_logits_triton

    kv = index_k.view(index_k.shape[0], C4_BLOCK, 1, INDEXER_HEAD_DIM + 4)
    device = kv.device
    plans = paged_mqa_logits_plans(bounds)
    for rows, pages in plans:
        q = torch.zeros(
            (rows, 1, INDEXER_HEADS, INDEXER_HEAD_DIM),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        weights = torch.zeros((rows, INDEXER_HEADS), dtype=torch.float32, device=device)
        # The indexer passes C4 lengths as [rows, 1] int32. A zero-filled page table
        # points at page 0, the padding page.
        seq_lens = torch.zeros((rows, 1), dtype=torch.int32, device=device)
        page_table = torch.zeros((rows, pages), dtype=torch.int32, device=device)
        fp8_paged_mqa_logits_triton(
            q, kv, weights, seq_lens, page_table, None, pages * C4_BLOCK, False
        )
    return len(plans)


def _warm_allocator(
    bounds: PrewarmBounds, last_loc_dtypes: Sequence[torch.dtype], device: torch.device
) -> int:
    """Launch the two allocator kernels at every bs_upper and last_loc dtype."""
    from sglang.kernels.ops.memory.allocator import (
        alloc_decode_kernel,
        alloc_extend_kernel,
    )

    n = 0
    for bs_upper in alloc_bs_upper_plans(bounds.max_reqs):
        bs = bs_upper
        # Every request starts a new page: seq_len 1, no prefix.
        seq_lens = torch.ones(bs, dtype=torch.int64, device=device)
        prefix_lens = torch.zeros(bs, dtype=torch.int64, device=device)
        free_pages = torch.arange(1, bs + 1, dtype=torch.int64, device=device)
        out = torch.empty(bs, dtype=torch.int64, device=device)
        for last_loc_dtype in last_loc_dtypes:
            last_loc = torch.full((bs,), -1, dtype=last_loc_dtype, device=device)
            alloc_decode_kernel[(bs,)](
                seq_lens, last_loc, free_pages, out, bs_upper, bounds.page_size
            )
            n += 1
        last_loc = torch.full((bs,), -1, dtype=torch.int64, device=device)
        alloc_extend_kernel[(bs,)](
            prefix_lens, seq_lens, last_loc, free_pages, out, bs_upper, bounds.page_size
        )
        n += 1
    return n


def _warm_page_table_positions(
    req_to_token: torch.Tensor, bounds: PrewarmBounds
) -> int:
    """One page-table launch per num_pages plan and request-index dtype."""
    from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
        build_page_table_positions_triton,
    )

    device = req_to_token.device
    n = 0
    for pages in page_table_positions_plans(bounds.max_pages):
        # The scheduler passes int64 request indices, and graph buffers can be int32.
        for idx_dtype in (torch.int64, torch.int32):
            build_page_table_positions_triton(
                req_to_token=req_to_token,
                req_pool_indices_repeated=torch.zeros(
                    1, dtype=idx_dtype, device=device
                ),
                seq_lens_casual=torch.ones(1, dtype=torch.int32, device=device),
                max_seq_len=pages * bounds.page_size,
                page_size=bounds.page_size,
                swa_window=128,
            )
            n += 1
    return n


def _warm_combine(max_c128: int, c4_topk: int, device: torch.device) -> int:
    """The combine kernel for SWA-only, C4 and every C128 plan."""
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
        combine_topk_swa_indices,
    )

    def launch(width: int, compress_ratio: int, topk: int) -> None:
        i32 = {"dtype": torch.int32, "device": device}
        combine_topk_swa_indices(
            topk_indices=torch.zeros((1, width), **i32),
            query_start_loc=torch.tensor([0, 1], **i32),
            seq_lens=torch.ones(1, **i32),
            gather_lens=torch.ones(1, **i32),
            compressed_base=torch.zeros(1, **i32),
            swa_base=torch.zeros(1, **i32),
            window_size=128,
            compress_ratio=compress_ratio,
            topk=topk,
        )

    # SWA-only layers (width 1, topk 0) and C4 layers (the index_topk of the model)
    launch(1, 1, 0)
    launch(c4_topk, 4, c4_topk)
    plans = combine_c128_plans(max_c128)
    for m in plans:
        launch(m, 128, m)
    return 2 + len(plans)


def _triton_logits_index_k(
    *,
    token_to_kv_pool: object,
    compress_ratios: Sequence[int],
    num_layers: int,
    fp4_indexer: bool,
) -> torch.Tensor | None:
    """The indexer K cache of the first C4 layer, or None without Triton logits."""
    layer_id = first_c4_layer_id(compress_ratios, num_layers)
    if layer_id is None or not indexer_uses_triton_logits(fp4_indexer):
        return None
    return token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=layer_id)


def _warm_dsv4_kernels(
    *,
    index_k: torch.Tensor | None,
    req_to_token: torch.Tensor,
    bounds: PrewarmBounds,
    c4_topk: int,
) -> dict[str, int]:
    """The launches of the DeepSeek V4 kernels, as {kernel: count}.

    index_k is None when the indexer does not use the Triton logits kernel.
    """
    counts = {}
    if index_k is not None:
        counts["paged_mqa_logits"] = _warm_paged_mqa_logits(index_k, bounds)
    counts["page_table_positions"] = _warm_page_table_positions(req_to_token, bounds)
    counts["combine_topk_swa"] = _warm_combine(
        max(bounds.full_width // 128, 1), c4_topk, req_to_token.device
    )
    return counts


def maybe_prewarm_triton_kernels(model_runner: ModelRunner) -> None:
    """Launch each reachable specialization of the warmed kernels once.

    This is a no-op unless SGLANG_OPT_XPU_TRITON_PREWARM is on, the device is an
    XPU and the runner is not a draft worker. It is the one function of this module
    that takes the runner, because it needs the pool tensors, the attention backend
    and the sizes that exist only after init_memory_pool and init_attention_backends
    run. It reads the runner and never writes it.
    """
    if (
        not envs.SGLANG_OPT_XPU_TRITON_PREWARM.get()
        or model_runner.device != "xpu"
        or model_runner.is_draft_worker
    ):
        return
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    device = torch.device(model_runner.device)
    req_to_token = model_runner.req_to_token_pool.req_to_token
    server_args = model_runner.server_args
    bounds = prewarm_bounds(
        chunked_prefill_size=server_args.chunked_prefill_size,
        max_prefill_tokens=server_args.max_prefill_tokens,
        max_running_requests=model_runner.max_running_requests,
        full_width=req_to_token.shape[1],
        page_size=model_runner.page_size,
    )
    tic = time.perf_counter()
    counts = {
        "alloc": _warm_allocator(
            bounds, decode_last_loc_dtypes(req_to_token.dtype), device
        )
    }
    backend = model_runner.attn_backend
    if isinstance(backend, DeepseekV4AttnBackend):
        config = model_runner.model_config.hf_text_config
        index_k = _triton_logits_index_k(
            token_to_kv_pool=model_runner.token_to_kv_pool,
            compress_ratios=config.compress_ratios,
            num_layers=config.num_hidden_layers,
            fp4_indexer=server_args.enable_deepseek_v4_fp4_indexer,
        )
        counts.update(
            _warm_dsv4_kernels(
                index_k=index_k,
                req_to_token=req_to_token,
                bounds=bounds,
                c4_topk=backend.c4_topk,
            )
        )
    torch.xpu.synchronize(device)
    # Return the dummy buffers to the device before the graphs are captured.
    torch.xpu.empty_cache()
    logger.info(
        "XPU Triton pre-warm: %s launches in %.1f s (max_rows %d, max_pages %d)",
        counts,
        time.perf_counter() - tic,
        bounds.max_rows,
        bounds.max_pages,
    )
