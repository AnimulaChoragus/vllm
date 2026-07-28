# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization support for GLM W4AFP8 checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4afp8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.w4afp8 import (
    interleave_w4afp8_scales,
    make_w4afp8_moe_strides,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )


class _W4AFP8LinearConfig(Fp8Config):
    keep_deepgemm_weight_scale_fp32_layout = True
    force_deepgemm_e8m0: bool | None = False


class W4AFP8Config(QuantizationConfig):
    """Configuration for GLM W4AFP8 checkpoint serialization."""

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        if group_size != 128:
            raise ValueError("W4AFP8 currently supports group_size=128 only")
        self.ignored_layers = ignored_layers or []
        self.group_size = group_size
        self.linear_quant_config = _W4AFP8LinearConfig(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=self.ignored_layers,
            weight_block_size=[128, 128],
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "w4afp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> W4AFP8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        if quant_method != "w4afp8":
            raise ValueError(f"Unsupported W4AFP8 quant_method: {quant_method}")
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            ignored_layers=ignored_layers,
            group_size=cls.get_from_keys_or(config, ["group_size"], 128),
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)
        self.linear_quant_config.ignored_layers = self.ignored_layers

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self.linear_quant_config)
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return W4AFP8MoEMethod(self, layer.moe_config)
        return None


class W4AFP8MoEMethod(FusedMoEMethodBase):
    """Routed-expert adapter for static-FP8 and packed-INT4 tensors."""

    def __init__(
        self,
        quant_config: W4AFP8Config,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.group_size = quant_config.group_size

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if (
            hidden_size % self.group_size
            or intermediate_size_per_partition % self.group_size
        ):
            raise ValueError("W4AFP8 expert dimensions must be divisible by group_size")
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        weight_specs = {
            "w13_weight": (
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
            ),
            "w2_weight": (
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
            ),
        }
        for name, shape in weight_specs.items():
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=torch.int8),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

        scale_attrs = {
            **extra_weight_attrs,
            "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
        }
        scale_specs = {
            "w13_weight_scale_inv": (
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
            ),
            "w2_weight_scale_inv": (
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
            ),
        }
        for name, scale_shape in scale_specs.items():
            param = torch.nn.Parameter(
                torch.empty(scale_shape, dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, scale_attrs)

        input_scale_specs = {
            "w13_input_scale": (num_experts, 2),
            "w2_input_scale": (num_experts,),
        }
        for name, input_scale_shape in input_scale_specs.items():
            param = torch.nn.Parameter(
                torch.ones(input_scale_shape, dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
            param.is_w4afp8_input_scale = True

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        device = layer.w13_weight.device
        (
            self.a_strides1,
            self.b_strides1,
            self.d_strides1,
            self.a_strides2,
            self.b_strides2,
            self.d_strides2,
        ) = make_w4afp8_moe_strides(
            layer.local_num_experts,
            layer.hidden_size,
            layer.intermediate_size_per_partition,
            device,
        )
        self.s_strides1 = self.d_strides1
        self.s_strides2 = self.d_strides2
        replace_parameter(
            layer,
            "w13_weight_scale_inv",
            torch.nn.Parameter(
                interleave_w4afp8_scales(layer.w13_weight_scale_inv),
                requires_grad=False,
            ),
        )
        replace_parameter(
            layer,
            "w2_weight_scale_inv",
            torch.nn.Parameter(
                interleave_w4afp8_scales(layer.w2_weight_scale_inv),
                requires_grad=False,
            ),
        )
        for name in ("w13_input_scale", "w2_input_scale"):
            value = getattr(layer, name).max().to(torch.float32).reshape(1)
            replace_parameter(
                layer,
                name,
                torch.nn.Parameter(value, requires_grad=False),
            )
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        from vllm.model_executor.layers.fused_moe.experts.w4afp8_moe import (
            make_w4afp8_moe_kernel,
        )

        self.moe_kernel = make_w4afp8_moe_kernel(self, layer)

    def get_fused_moe_quant_config(
        self,
        layer: RoutedExperts,
    ) -> FusedMoEQuantConfig:
        return int4_w4afp8_moe_quant_config(
            w1_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            g1_alphas=torch.empty(0, device=layer.w13_weight.device),
            g2_alphas=torch.empty(0, device=layer.w2_weight.device),
            per_act_token_quant=False,
            per_out_ch_quant=False,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
