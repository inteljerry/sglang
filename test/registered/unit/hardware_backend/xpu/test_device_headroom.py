"""XPU device headroom: the cap on the caching allocator.

The tests check the arithmetic of the cap, the device that is passed in, the cases
without a cap, the warnings, the types of the errors that the code hides and the
types that it does not, and the read of the environment variable.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.xpu import device_headroom
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TOTAL = 32 * 1024 << 20  # a 32 GiB device
MIB = 1 << 20


class FakeXpu:
    """The calls of torch.xpu that the cap uses. `calls` records the cap."""

    def __init__(
        self,
        reserved_mb: int,
        free_mb: int,
        fail: Exception | None = None,
        fail_set: Exception | None = None,
    ) -> None:
        self.reserved = reserved_mb * MIB
        self.free = free_mb * MIB
        self.fail = fail
        self.fail_set = fail_set
        self.calls: list[tuple[float, int]] = []

    def get_device_properties(self, device: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=TOTAL)

    def mem_get_info(self, device: int | None = None) -> tuple[int, int]:
        if self.fail is not None:
            raise self.fail
        return self.free, TOTAL

    def memory_reserved(self, device: int | None = None) -> int:
        return self.reserved

    def set_per_process_memory_fraction(self, fraction: float, device: int) -> None:
        if self.fail_set is not None:
            raise self.fail_set
        self.calls.append((fraction, device))


class TestXpuDeviceHeadroom(CustomTestCase):
    def _cap(
        self, fake: FakeXpu, headroom_mb: int, xpu: bool = True
    ) -> list[tuple[float, int]]:
        with (
            patch.object(device_headroom, "is_xpu", return_value=xpu),
            patch.object(device_headroom.torch, "xpu", fake, create=True),
        ):
            device_headroom.xpu_reserve_device_headroom(headroom_mb, 3)
        return fake.calls

    def test_the_cap_counts_memory_held_outside_torch(self) -> None:
        # A rank after start-up: torch holds 23,976 MiB and 4,161 MiB are free. The
        # device holds the rest (about 4.5 GB) outside torch.
        calls = self._cap(FakeXpu(23976, 4161), 1024)
        self.assertEqual(len(calls), 1)
        fraction, device = calls[0]
        self.assertEqual(device, 3)  # the device that is passed in, not the current one
        self.assertAlmostEqual(fraction * TOTAL, (23976 + 4161 - 1024) * MIB, delta=1)

    def test_a_failed_device_query_does_not_stop_the_server(self) -> None:
        for error in (RuntimeError("sysman query failed"), ValueError("bad fraction")):
            with self.subTest(type(error).__name__):
                fake = FakeXpu(23976, 4161, fail=error)
                with self.assertLogs(device_headroom.logger, "WARNING") as logs:
                    calls = self._cap(fake, 1024)
                self.assertEqual(calls, [])
                self.assertIn("not applied", logs.output[0])

    def test_a_failed_cap_is_not_hidden(self) -> None:
        """The variable asks for the cap, so an error of the setter must reach the caller."""
        fake = FakeXpu(23976, 4161, fail_set=RuntimeError("cannot set the fraction"))
        with self.assertRaises(RuntimeError):
            self._cap(fake, 1024)

    def test_an_unexpected_error_is_not_hidden(self) -> None:
        with self.assertRaises(KeyError):
            self._cap(FakeXpu(23976, 4161, fail=KeyError("x")), 1024)

    def test_not_applied_without_room(self) -> None:
        with self.assertLogs(device_headroom.logger, "WARNING") as logs:
            calls = self._cap(FakeXpu(30000, 900), 1024)
        self.assertEqual(calls, [])
        self.assertIn("only 900 MiB free", logs.output[0])

    def test_the_environment_variable_sets_the_headroom(self) -> None:
        fake = FakeXpu(23976, 4161)
        with (
            envs.SGLANG_OPT_XPU_DEVICE_HEADROOM_MB.override(2048),
            patch.object(device_headroom, "is_xpu", return_value=True),
            patch.object(device_headroom.torch, "xpu", fake, create=True),
        ):
            device_headroom.apply_xpu_device_headroom_from_env(5)
        ((fraction, device),) = fake.calls
        self.assertEqual(device, 5)
        self.assertAlmostEqual(fraction * TOTAL, (23976 + 4161 - 2048) * MIB, delta=1)

    def test_off_at_zero_and_on_other_devices(self) -> None:
        self.assertEqual(self._cap(FakeXpu(23976, 4161), 0), [])
        self.assertEqual(self._cap(FakeXpu(23976, 4161), 1024, xpu=False), [])
        self.assertEqual(self._cap(FakeXpu(23976, 4161), -5), [])


if __name__ == "__main__":
    unittest.main()
