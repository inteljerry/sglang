"""Keep a set amount of device memory outside the caching allocator of XPU."""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu

logger = logging.getLogger(__name__)


def xpu_reserve_device_headroom(headroom_mb: int, device: int) -> None:
    """Cap the XPU caching allocator so that headroom_mb of device memory stays outside it.

    A long chunked prefill can leave no device memory free outside the cache of
    torch. An allocation made outside the cache can then fail with
    UR_RESULT_ERROR_OUT_OF_RESOURCES. The cap is the memory that torch holds now
    plus the memory that the device reports free, less the headroom. Torch then
    raises OutOfMemoryError at the cap, and the headroom stays free. A headroom of
    0 or less, a device that is not an XPU, and a cap at or below what torch
    holds now apply no cap. The value must come from a measurement of the
    tightest rank.
    """
    if headroom_mb <= 0 or not is_xpu():
        return
    try:
        total = torch.xpu.get_device_properties(device).total_memory
        free, _ = torch.xpu.mem_get_info(device)
        reserved = torch.xpu.memory_reserved(device)
    except (RuntimeError, ValueError) as error:
        # The cap is optional. A failed device query must not stop the server.
        logger.warning(f"XPU device headroom not applied: {error!r}")
        return
    limit = reserved + free - (headroom_mb << 20)
    if limit <= reserved:
        logger.warning(
            f"XPU device headroom of {headroom_mb} MiB not applied: only "
            f"{free >> 20} MiB free after start-up"
        )
        return
    torch.xpu.set_per_process_memory_fraction(limit / total, device)
    logger.info(
        f"XPU caching allocator capped at {limit >> 20} MiB of {total >> 20} MiB "
        f"(holds {reserved >> 20}, {free >> 20} free): {headroom_mb} MiB stay outside it"
    )


def apply_xpu_device_headroom_from_env(device: int) -> None:
    """Apply the cap that SGLANG_OPT_XPU_DEVICE_HEADROOM_MB asks for."""
    xpu_reserve_device_headroom(envs.SGLANG_OPT_XPU_DEVICE_HEADROOM_MB.get(), device)
