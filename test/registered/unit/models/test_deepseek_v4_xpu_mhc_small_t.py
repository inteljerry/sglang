"""DeepSeek V4 on XPU: mhc_fused_post_pre calls of 1 to 8 tokens run the small-T op.

With SGLANG_OPT_XPU_MHC_SMALL_T, _get_mhc_ops() on XPU wraps mhc_fused_post_pre of
sgl_kernel. A call with 1 to 8 tokens runs mhc_fused_post_pre_small_t with the same
arguments, and every other call runs mhc_fused_post_pre. With the switch off, or
with an sgl_kernel that has no mhc_fused_post_pre_small_t, the op object is not
changed. The tests replace sgl_kernel with a stub module.
"""

import sys
import types
import unittest
from typing import Any, Callable
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_XPU_OPS = (
    "fused_hc_head",
    "hc_post",
    "hc_split_sinkhorn",
    "mhc_fused_post_pre",
    "mhc_pre",
)


def _stub_sgl_kernel(
    with_small_t: bool,
) -> tuple[types.ModuleType, list[tuple[str, tuple, dict]]]:
    """A stub sgl_kernel whose ops record (name, args, kwargs) and return their name."""
    calls = []
    module = types.ModuleType("sgl_kernel")

    def make_op(name: str) -> Callable[..., str]:
        def op(*args: Any, **kwargs: Any) -> str:
            calls.append((name, args, kwargs))
            return name

        return op

    names = _XPU_OPS + (("mhc_fused_post_pre_small_t",) if with_small_t else ())
    for name in names:
        setattr(module, name, make_op(name))
    return module, calls


def _load_ops(
    small_t: bool, with_kernel: bool = True
) -> tuple[types.ModuleType, deepseek_v4.MhcOps, list[tuple[str, tuple, dict]]]:
    """_get_mhc_ops() on XPU with the stub sgl_kernel. Returns (module, ops, calls)."""
    module, calls = _stub_sgl_kernel(with_kernel)
    deepseek_v4._get_mhc_ops.cache_clear()
    try:
        with (
            patch.dict(sys.modules, {"sgl_kernel": module}),
            patch.object(deepseek_v4, "_is_xpu", True),
            envs.SGLANG_OPT_XPU_MHC_SMALL_T.override(small_t),
        ):
            ops = deepseek_v4._get_mhc_ops()
    finally:
        deepseek_v4._get_mhc_ops.cache_clear()
    return module, ops, calls


class TestDeepseekV4XpuMhcSmallT(CustomTestCase):
    def test_the_op_is_not_changed_with_the_switch_off(self) -> None:
        self.assertIs(envs.SGLANG_OPT_XPU_MHC_SMALL_T.default, False)
        module, ops, _ = _load_ops(small_t=False)
        self.assertIs(ops.mhc_fused_post_pre, module.mhc_fused_post_pre)
        self.assertIs(ops.mhc_pre, module.mhc_pre)
        self.assertIs(ops.mhc_post, module.hc_post)

    def test_the_op_is_not_changed_without_the_kernel(self) -> None:
        with self.assertLogs(deepseek_v4.logger, level="WARNING") as logs:
            module, ops, _ = _load_ops(small_t=True, with_kernel=False)
        self.assertIs(ops.mhc_fused_post_pre, module.mhc_fused_post_pre)
        self.assertIn("mhc_fused_post_pre_small_t", logs.output[0])

    def test_small_batches_run_small_t_with_the_same_arguments(self) -> None:
        _, ops, calls = _load_ops(small_t=True)
        residual = torch.empty(0)
        for num_tokens in (1, 2, 4, 8):
            with self.subTest(num_tokens=num_tokens):
                x = torch.empty(num_tokens, 16)
                result = ops.mhc_fused_post_pre(
                    x, residual, 1e-6, norm_weight="w", norm_eps=1e-6
                )
                self.assertEqual(result, "mhc_fused_post_pre_small_t")
                name, args, kwargs = calls[-1]
                self.assertEqual(name, "mhc_fused_post_pre_small_t")
                self.assertIs(args[0], x)
                self.assertIs(args[1], residual)
                self.assertEqual(args[2:], (1e-6,))
                self.assertEqual(kwargs, {"norm_weight": "w", "norm_eps": 1e-6})

    def test_other_batches_run_mhc_fused_post_pre(self) -> None:
        _, ops, calls = _load_ops(small_t=True)
        for num_tokens in (0, 9, 16, 512):
            with self.subTest(num_tokens=num_tokens):
                x = torch.empty(num_tokens, 16)
                result = ops.mhc_fused_post_pre(x, n_splits=0)
                self.assertEqual(result, "mhc_fused_post_pre")
                name, args, kwargs = calls[-1]
                self.assertEqual(name, "mhc_fused_post_pre")
                self.assertIs(args[0], x)
                self.assertEqual(kwargs, {"n_splits": 0})


if __name__ == "__main__":
    unittest.main()
