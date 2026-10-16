"""XPU device headroom: with the caching allocator capped, torch runs out of memory
before the device does.

The test fills the device with torch allocations until torch raises OutOfMemoryError.
Then it checks that the device still reports at least the configured headroom free
(less a small allowance for the runtime), so an allocation made outside the cache of
torch can still succeed.
"""

import unittest

import torch

from sglang.srt.hardware_backend.xpu.device_headroom import xpu_reserve_device_headroom
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

HEADROOM_MB = 1024
ALLOWANCE_MB = 128  # runtime allocations made while filling


@unittest.skipUnless(torch.xpu.is_available(), "needs an XPU device")
class TestXpuDeviceHeadroom(CustomTestCase):
    def test_torch_stops_short_of_the_headroom(self) -> None:
        device = torch.xpu.current_device()
        before = torch.xpu.get_per_process_memory_fraction(device)
        blocks = []
        try:
            xpu_reserve_device_headroom(HEADROOM_MB, device)
            # the cap was applied
            self.assertLess(torch.xpu.get_per_process_memory_fraction(device), before)
            with self.assertRaises(torch.OutOfMemoryError):
                while True:
                    blocks.append(
                        torch.empty(256 << 20, dtype=torch.uint8, device="xpu")
                    )
            torch.xpu.synchronize()
            free, _ = torch.xpu.mem_get_info()
            self.assertGreaterEqual(
                free >> 20, HEADROOM_MB - ALLOWANCE_MB, f"free {free >> 20} MiB"
            )
        finally:
            del blocks
            torch.xpu.empty_cache()
            torch.xpu.set_per_process_memory_fraction(before, device)


if __name__ == "__main__":
    unittest.main()
