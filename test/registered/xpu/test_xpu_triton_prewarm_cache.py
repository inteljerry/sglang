"""XPU: after the Triton pre-warm, reachable logits launches compile nothing new.

_warm_paged_mqa_logits launches the C4 indexer logits kernel once per planned
specialization key. Launches at random reachable shapes after it must reuse a
compiled variant: the in-process cache of the kernel (JITFunction.device_caches)
must not grow.
"""

import importlib
import unittest

import torch

from sglang.srt.hardware_backend.xpu import triton_prewarm as tp
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=120, suite="stage-b-test-1-gpu-xpu")

MAX_ROWS, MAX_PAGES = 2048, 160


@unittest.skipUnless(torch.xpu.is_available(), "needs an XPU device")
class TestPrewarmLeavesNothingToCompile(CustomTestCase):
    def test_random_reachable_shapes_hit_the_cache(self) -> None:
        # the package exports a function of the same name, so import the module
        launcher = importlib.import_module("sgl_kernel.fp8_paged_mqa_logits")

        device = torch.device("xpu", 0)
        num_pages = 1024 + 1
        index_k = torch.zeros((num_pages, 64 * 132), dtype=torch.uint8, device=device)
        bounds = tp.PrewarmBounds(
            max_rows=MAX_ROWS,
            max_pages=MAX_PAGES,
            max_reqs=1 << 20,
            page_size=256,
            full_width=MAX_PAGES * 256,
        )
        tp._warm_paged_mqa_logits(index_k, bounds)
        kernel = launcher._paged_mqa_logits_kernel

        def variants() -> int:
            return sum(len(cache[0]) for cache in kernel.device_caches.values())

        before = variants()
        kv = index_k.view(num_pages, 64, 1, 132)
        g = torch.Generator().manual_seed(0)
        for _ in range(40):
            rows = int(torch.randint(1, MAX_ROWS + 1, (1,), generator=g))
            pages = int(torch.randint(1, MAX_PAGES + 1, (1,), generator=g))
            launcher.fp8_paged_mqa_logits_triton(
                torch.zeros(
                    (rows, 1, 64, 128), dtype=torch.float8_e4m3fn, device=device
                ),
                kv,
                torch.zeros((rows, 64), dtype=torch.float32, device=device),
                torch.ones((rows, 1), dtype=torch.int32, device=device),
                torch.zeros((rows, pages), dtype=torch.int32, device=device),
                None,
                pages * 64,
                False,
            )
        torch.xpu.synchronize()
        self.assertEqual(variants(), before)


if __name__ == "__main__":
    unittest.main()
