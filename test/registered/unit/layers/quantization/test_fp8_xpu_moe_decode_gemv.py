"""XPU MXFP4 MoE: decode batches that moe_w4a16_decode_gemv takes skip fused_experts.

With SGLANG_OPT_XPU_MOE_DECODE_GEMV on XPU, Fp8MoEMethod.apply runs
sgl_kernel.moe_w4a16_decode_gemv for an MXFP4 decode batch within the limits of the
op, and fused_experts for every other call. Without the op, the switch logs a
warning and fused_experts runs. The tests replace sgl_kernel with a stub module.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

HIDDEN, INTER, EXPERTS = 512, 256, 4


def make_call(**change: Any) -> dict:
    """The arguments of an eligible decode call of 2 tokens, with some changed."""
    tokens, topk = change.pop("tokens", 2), change.pop("topk", 8)
    hidden, inter = change.pop("hidden", HIDDEN), change.pop("inter", INTER)
    layer = SimpleNamespace(
        w13_weight=torch.zeros(EXPERTS, 2 * inter, hidden // 2, dtype=torch.int8),
        w2_weight=torch.zeros(EXPERTS, hidden, inter // 2, dtype=torch.int8),
        w13_weight_scale_inv=torch.ones(1),
        w2_weight_scale_inv=torch.ones(1),
    )
    config = SimpleNamespace(
        activation="silu",
        swiglu_limit=10.0,
        gemm1_alpha=None,
        no_combine=False,
        routed_scaling_factor=1.5,
        gemm1_clamp_limit=None,
        apply_router_weight_on_input=False,
    )
    call = {
        "layer": layer,
        "x": torch.zeros(tokens, hidden, dtype=torch.bfloat16),
        "topk_weights": torch.ones(tokens, topk),
        "topk_ids": torch.zeros(tokens, topk, dtype=torch.int32),
        "config": config,
        "max_tokens": 8,
    }
    for key, value in change.items():
        target = call if key in call else (layer if hasattr(layer, key) else config)
        if isinstance(target, dict):
            target[key] = value
        else:
            setattr(target, key, value)
    return call


def eligible(call: dict) -> bool:
    return fp8._xpu_moe_decode_gemv_eligible(
        call["layer"],
        call["x"],
        (call["topk_weights"], call["topk_ids"], None),
        call["config"],
        call["max_tokens"],
    )


def misaligned(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """A contiguous tensor whose data starts 2 bytes past a 16-byte boundary."""
    size = torch.Size(shape).numel() * torch.tensor([], dtype=dtype).element_size()
    flat = torch.zeros(size + 32, dtype=torch.int8)
    start = (2 - flat.data_ptr()) % 16
    return flat[start : start + size].view(dtype).view(shape)


class TestDecodeGemvEligible(CustomTestCase):
    def test_a_decode_batch_within_the_limits_is_eligible(self) -> None:
        for tokens, topk in ((1, 8), (8, 8), (8, 6)):
            with self.subTest(tokens=tokens, topk=topk):
                self.assertTrue(eligible(make_call(tokens=tokens, topk=topk)))
        self.assertTrue(eligible(make_call(inter=512)))

    def test_each_limit_of_the_op_keeps_fused_experts(self) -> None:
        bias = torch.zeros(EXPERTS, 2 * INTER)
        cases = {
            "no tokens": {"tokens": 0},
            # 40 routes: only the token limit rejects it
            "above max_tokens": {"tokens": 5, "max_tokens": 4},
            "topk 4": {"topk": 4},
            "T * topk above 64": {"tokens": 9, "max_tokens": 16},
            "fp16 activations": {"x": torch.zeros(2, HIDDEN, dtype=torch.float16)},
            "strided x": {"x": torch.zeros(HIDDEN, 2, dtype=torch.bfloat16).t()},
            "unaligned x": {"x": misaligned((2, HIDDEN), torch.bfloat16)},
            "float topk ids": {"topk_ids": torch.zeros(2, 8)},
            "fp16 topk weights": {"topk_weights": torch.ones(2, 8).half()},
            "hidden 768": {"hidden": 768},
            "intermediate 384": {"inter": 384},
            "gelu": {"activation": "gelu"},
            "no swiglu limit": {"swiglu_limit": None},
            "zero swiglu limit": {"swiglu_limit": 0.0},
            "gemm1 alpha": {"gemm1_alpha": 1.702},
            "no combine": {"no_combine": True},
        }
        for name, change in cases.items():
            with self.subTest(name):
                self.assertFalse(eligible(make_call(**change)))
        for weight in ("w13_weight", "w2_weight"):
            with self.subTest(f"unaligned {weight}"):
                call = make_call()
                old = getattr(call["layer"], weight)
                setattr(call["layer"], weight, misaligned(old.shape, old.dtype))
                self.assertFalse(eligible(call))
        for name in ("w13_weight_bias", "w2_weight_bias"):
            with self.subTest(name):
                call = make_call()
                setattr(call["layer"], name, bias)
                self.assertFalse(eligible(call))


def _stub_sgl_kernel(
    with_gemv: bool,
) -> tuple[types.ModuleType, list[tuple[str, tuple, dict[str, Any]]]]:
    """A stub sgl_kernel whose fused_experts and GEMV record their calls."""
    calls = []
    module = types.ModuleType("sgl_kernel")

    def make_op(name: str) -> Callable[..., torch.Tensor]:
        def op(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            calls.append((name, args, kwargs))
            return torch.full_like(x, 7.0 if name == "fused_experts" else 3.0)

        return op

    module.fused_experts = make_op("fused_experts")
    if with_gemv:
        module.moe_w4a16_decode_gemv = make_op("moe_w4a16_decode_gemv")
    return module, calls


class TestDecodeGemvOp(CustomTestCase):
    def setUp(self) -> None:
        fp8._xpu_moe_decode_gemv_op.cache_clear()
        self.addCleanup(fp8._xpu_moe_decode_gemv_op.cache_clear)

    def test_the_op_comes_from_sgl_kernel(self) -> None:
        module, _ = _stub_sgl_kernel(with_gemv=True)
        with patch.dict(sys.modules, {"sgl_kernel": module}):
            self.assertIs(fp8._xpu_moe_decode_gemv_op(), module.moe_w4a16_decode_gemv)

    def test_a_missing_op_logs_a_warning_and_gives_none(self) -> None:
        module, _ = _stub_sgl_kernel(with_gemv=False)
        with (
            patch.dict(sys.modules, {"sgl_kernel": module}),
            self.assertLogs(fp8.logger, level="WARNING") as logs,
        ):
            self.assertIsNone(fp8._xpu_moe_decode_gemv_op())
        self.assertIn("moe_w4a16_decode_gemv", logs.output[0])

    def test_the_method_reads_the_switches_once_on_xpu(self) -> None:
        module, _ = _stub_sgl_kernel(with_gemv=True)
        config = SimpleNamespace(
            weight_block_size=[128, 128],
            is_fp4_experts=True,
            dequant_fp4_to_fp8=False,
        )
        self.assertIs(envs.SGLANG_OPT_XPU_MOE_DECODE_GEMV.default, False)
        self.assertEqual(envs.SGLANG_OPT_XPU_MOE_DECODE_GEMV_MAX_TOKENS.default, 8)
        for on_xpu, switch, expected in (
            (True, True, module.moe_w4a16_decode_gemv),
            (True, False, None),
            (False, True, None),
        ):
            with (
                self.subTest(on_xpu=on_xpu, switch=switch),
                patch.dict(sys.modules, {"sgl_kernel": module}),
                patch.object(fp8, "_is_xpu", on_xpu),
                envs.SGLANG_OPT_XPU_MOE_DECODE_GEMV.override(switch),
                envs.SGLANG_OPT_XPU_MOE_DECODE_GEMV_MAX_TOKENS.override(4),
            ):
                method = fp8.Fp8MoEMethod(config)
                self.assertIs(method.xpu_moe_decode_gemv, expected)
                self.assertEqual(method.xpu_moe_decode_gemv_max_tokens, 4)


class TestApplyRouting(CustomTestCase):
    def _apply(self, call: dict, with_op: bool) -> tuple[torch.Tensor, list]:
        """Fp8MoEMethod.apply on the XPU backend. Returns the output and op calls."""
        module, calls = _stub_sgl_kernel(with_gemv=True)
        method = SimpleNamespace(
            moe_runner_config=call["config"],
            is_fp4_expert=True,
            block_quant=True,
            xpu_moe_decode_gemv=module.moe_w4a16_decode_gemv if with_op else None,
            xpu_moe_decode_gemv_max_tokens=call["max_tokens"],
        )
        dispatch_output = SimpleNamespace(
            hidden_states=call["x"],
            topk_output=(call["topk_weights"], call["topk_ids"], None),
        )
        with (
            patch.dict(sys.modules, {"sgl_kernel": module}),
            patch.object(fp8, "use_intel_amx_backend", return_value=False),
            patch.object(fp8, "use_intel_xpu_backend", return_value=True),
        ):
            combine = fp8.Fp8MoEMethod.apply(method, call["layer"], dispatch_output)
        return combine.hidden_states, calls

    def test_an_eligible_decode_batch_runs_the_gemv(self) -> None:
        call = make_call()
        output, calls = self._apply(call, with_op=True)
        self.assertEqual([name for name, _, _ in calls], ["moe_w4a16_decode_gemv"])
        self.assertTrue(torch.equal(output, torch.full_like(call["x"], 3.0)))
        _, args, kwargs = calls[0]
        self.assertEqual(len(args), 4)
        self.assertIs(args[0], call["layer"].w13_weight)
        self.assertIs(args[1], call["layer"].w2_weight)
        self.assertIs(args[2], call["topk_weights"])
        self.assertIs(args[3], call["topk_ids"])
        self.assertIs(kwargs["w1_scale"], call["layer"].w13_weight_scale_inv)
        self.assertIs(kwargs["w2_scale"], call["layer"].w2_weight_scale_inv)
        self.assertEqual(kwargs["routed_scaling_factor"], 1.5)
        self.assertEqual(kwargs["swiglu_limit"], 10.0)

    def test_other_calls_run_fused_experts(self) -> None:
        for name, call, with_op in (
            ("no op", make_call(), False),
            ("prefill", make_call(tokens=64, max_tokens=8), True),
        ):
            with self.subTest(name):
                output, calls = self._apply(call, with_op)
                self.assertEqual([op for op, _, _ in calls], ["fused_experts"])
                self.assertTrue(torch.equal(output, torch.full_like(call["x"], 7.0)))


if __name__ == "__main__":
    unittest.main()
