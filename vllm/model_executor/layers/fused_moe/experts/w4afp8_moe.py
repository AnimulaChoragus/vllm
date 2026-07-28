# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUTLASS experts for the GLM W4AFP8 checkpoint format."""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform


def _run_w4afp8(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    strides: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    topk_weights: torch.Tensor,
    group_size: int,
) -> None:
    if activation not in (
        MoEActivation.SILU,
        MoEActivation.GELU,
        MoEActivation.SWIGLUOAI,
    ):
        raise NotImplementedError(f"W4AFP8 does not support {activation}")
    if w1.dtype != torch.int8 or w2.dtype != torch.int8:
        raise ValueError("W4AFP8 expert weights must be packed int8 tensors")
    if group_size != 128:
        raise ValueError("W4AFP8 CUTLASS kernel supports group_size=128 only")

    tokens, hidden = hidden_states.shape
    intermediate = w2.size(2) * 2
    topk = topk_ids.size(1)
    rows = tokens * topk
    if w1.size(2) * 2 != hidden or w1.size(1) != intermediate * 2:
        raise ValueError(
            "W4AFP8 w13 shape is incompatible with routed-expert dimensions"
        )
    if w2.size(1) != hidden or intermediate % group_size or hidden % group_size:
        raise ValueError("W4AFP8 dimensions must be divisible by group_size")

    a1q_storage = _resize_cache(
        workspace2.view(torch.float8_e4m3fn),
        (rows, hidden),
    )
    mm1_out = _resize_cache(workspace13, (rows, intermediate * 2))
    act_out = _resize_cache(workspace2, (rows, intermediate))
    mm1_numel = rows * intermediate * 2
    quant_storage = workspace13.flatten()[mm1_numel:].view(torch.float8_e4m3fn)
    quant_out = _resize_cache(quant_storage, (rows, intermediate))
    mm2_out = _resize_cache(workspace2, (rows, hidden))

    unpermuted = torch.empty_like(
        hidden_states,
        dtype=torch.float8_e4m3fn,
    )
    ops.scaled_fp8_quant(hidden_states, a1_scale, output=unpermuted)
    num_experts = global_num_experts if expert_map is None else expert_map.size(0)
    a1q, _, expert_offsets_all, inv_perm, _ = moe_permute(
        unpermuted,
        None,
        topk_ids,
        num_experts,
        w1.size(0),
        expert_map,
        permuted_hidden_states=a1q_storage,
    )
    problem_sizes1 = torch.empty(
        (w1.size(0), 3),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    problem_sizes2 = torch.empty_like(problem_sizes1)
    ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
        expert_offsets_all,
        problem_sizes1,
        problem_sizes2,
        intermediate,
        hidden,
        True,
    )
    a_s1, b_s1, d_s1, a_s2, b_s2, d_s2 = strides
    expert_offsets = expert_offsets_all[:-1].to(torch.int32)
    ops.cutlass_w4afp8_moe_mm(
        mm1_out,
        a1q,
        w1,
        a1_scale.float(),
        w1_scale,
        expert_offsets,
        problem_sizes1,
        a_s1,
        b_s1,
        d_s1,
        d_s1,
        group_size,
        topk,
    )
    if activation == MoEActivation.SILU:
        torch.ops._C.silu_and_mul_quant(
            quant_out,
            mm1_out,
            a2_scale.float(),
        )
        a2q = quant_out
    else:
        apply_moe_activation(activation, act_out, mm1_out)
        a2q, _ = ops.scaled_fp8_quant(
            act_out,
            a2_scale,
            output=quant_out,
        )
    ops.cutlass_w4afp8_moe_mm(
        mm2_out,
        a2q,
        w2,
        a2_scale.float(),
        w2_scale,
        expert_offsets,
        problem_sizes2,
        a_s2,
        b_s2,
        d_s2,
        d_s2,
        group_size,
        topk,
    )
    moe_unpermute(
        output,
        mm2_out,
        topk_weights,
        inv_perm,
        expert_offsets_all,
    )


class CutlassExpertsW4AFP8(mk.FusedMoEExpertsModular):
    def __init__(self, method, layer) -> None:
        assert method.moe_quant_config is not None
        super().__init__(
            moe_config=method.moe,
            quant_config=method.moe_quant_config,
        )
        self.out_dtype = method.moe.in_dtype
        self.strides = (
            method.a_strides1,
            method.b_strides1,
            method.d_strides1,
            method.a_strides2,
            method.b_strides2,
            method.d_strides2,
        )
        self.static_a1_scale = layer.w13_input_scale
        self.static_a2_scale = layer.w2_input_scale
        self.group_size = method.group_size

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda() and (
            current_platform.is_device_capability(90)
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return moe_parallel_config.ep_size == 1

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if expert_tokens_meta is not None:
            raise NotImplementedError("W4AFP8 batched-expert format is not implemented")
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        return (
            (M * topk, max(K, N + (N + 3) // 4)),
            (M * topk, max(activation_out_dim, K)),
            (M, K),
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        if expert_tokens_meta is not None or workspace13 is None or workspace2 is None:
            raise NotImplementedError("W4AFP8 requires standard expert dispatch")
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "W4AFP8 does not support input-side router weighting"
            )
        _run_w4afp8(
            output,
            hidden_states,
            w1,
            w2,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            self.w1_scale,
            self.w2_scale,
            self.static_a1_scale,
            self.static_a2_scale,
            self.strides,
            workspace13,
            workspace2,
            topk_weights,
            self.group_size,
        )


def make_w4afp8_moe_kernel(method, layer) -> mk.FusedMoEKernel:
    supported, reason = CutlassExpertsW4AFP8.is_supported_config(
        CutlassExpertsW4AFP8,
        method.moe,
        None,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )
    if not supported:
        raise NotImplementedError(
            f"W4AFP8 CUTLASS kernel does not support this configuration: {reason}"
        )
    return mk.FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(),
        CutlassExpertsW4AFP8(method, layer),
    )
