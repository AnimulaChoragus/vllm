# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.w4afp8 import (
    interleave_w4afp8_scales,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
)
from vllm.model_executor.layers.quantization.w4afp8 import (
    W4AFP8Config,
    W4AFP8MoEMethod,
)


def test_w4afp8_config_resolves_checkpoint_quant_method() -> None:
    config_cls = get_quantization_config("w4afp8")
    config = config_cls.from_config({"quant_method": "w4afp8"})

    assert config_cls is W4AFP8Config
    assert config.group_size == 128
    assert config.linear_quant_config.is_checkpoint_fp8_serialized
    assert config.linear_quant_config.weight_block_size == [128, 128]
    assert config.linear_quant_config.keep_deepgemm_weight_scale_fp32_layout
    assert config.linear_quant_config.force_deepgemm_e8m0 is False
    assert config.get_supported_act_dtypes() == [torch.bfloat16]


def test_w4afp8_config_rejects_unsupported_group_size() -> None:
    with pytest.raises(ValueError, match="group_size=128"):
        W4AFP8Config.from_config({"quant_method": "w4afp8", "group_size": 64})


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        (
            8,
            torch.tensor(
                [
                    [
                        [0, 1, 2, 3, 8, 9, 10, 11],
                        [4, 5, 6, 7, 12, 13, 14, 15],
                    ]
                ],
                dtype=torch.bfloat16,
            ),
        ),
        (
            3,
            torch.tensor(
                [[[0, 3], [1, 4], [2, 5]]],
                dtype=torch.bfloat16,
            ),
        ),
    ],
)
def test_interleave_w4afp8_scales_matches_cutlass_layout(
    groups: int,
    expected: torch.Tensor,
) -> None:
    scales = torch.arange(
        2 * groups,
        dtype=torch.bfloat16,
    ).reshape(1, 2, groups)

    actual = interleave_w4afp8_scales(scales)

    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()


def test_interleave_w4afp8_scales_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="experts, out_features, groups"):
        interleave_w4afp8_scales(torch.ones(2, 4))


def test_w4afp8_expert_parameter_shapes_match_checkpoint_layout() -> None:
    method = object.__new__(W4AFP8MoEMethod)
    method.group_size = 128
    layer = torch.nn.Module()

    method.create_weights(
        layer=layer,
        num_experts=3,
        hidden_size=512,
        intermediate_size_per_partition=256,
        params_dtype=torch.bfloat16,
    )

    assert layer.w13_weight.shape == (3, 512, 256)
    assert layer.w2_weight.shape == (3, 512, 128)
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w2_weight.dtype == torch.int8
    assert layer.w13_weight_scale_inv.shape == (3, 512, 4)
    assert layer.w2_weight_scale_inv.shape == (3, 512, 2)
    assert layer.w13_input_scale.shape == (3, 2)
    assert layer.w2_input_scale.shape == (3,)


def test_w4afp8_dense_policy_preserves_checkpoint_scale_layout() -> None:
    weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
    scale = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    processed_weight, processed_scale = deepgemm_post_process_fp8_weight_block(
        weight,
        scale,
        quant_block_shape=(128, 128),
        use_e8m0=False,
        keep_weight_scale_fp32_layout=True,
    )

    assert processed_weight is weight
    assert processed_scale is scale


def test_w4afp8_post_load_connects_scales_to_moe_kernel(monkeypatch) -> None:
    from vllm.model_executor.layers.fused_moe.experts import w4afp8_moe

    method = object.__new__(W4AFP8MoEMethod)
    method.group_size = 128
    method.moe = SimpleNamespace()
    layer = torch.nn.Module()
    method.create_weights(
        layer=layer,
        num_experts=3,
        hidden_size=512,
        intermediate_size_per_partition=256,
        params_dtype=torch.bfloat16,
    )
    layer.local_num_experts = 3
    layer.hidden_size = 512
    layer.intermediate_size_per_partition = 256
    layer.w13_input_scale.data.copy_(
        torch.tensor(
            [[0.25, 0.5], [0.75, 0.125], [0.375, 0.625]],
            dtype=torch.bfloat16,
        )
    )
    layer.w2_input_scale.data.copy_(
        torch.tensor([0.25, 0.5, 0.375], dtype=torch.bfloat16)
    )
    sentinel = object()
    monkeypatch.setattr(
        w4afp8_moe,
        "make_w4afp8_moe_kernel",
        lambda method, layer: sentinel,
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight_scale_inv.shape == (3, 1, 2048)
    assert layer.w2_weight_scale_inv.shape == (3, 2, 512)
    assert layer.w13_input_scale.dtype == torch.float32
    assert layer.w2_input_scale.dtype == torch.float32
    torch.testing.assert_close(
        layer.w13_input_scale,
        torch.tensor([0.75]),
    )
    torch.testing.assert_close(
        layer.w2_input_scale,
        torch.tensor([0.5]),
    )
    assert method.moe_kernel is sentinel
    assert method.moe_quant_config.w1_scale is layer.w13_weight_scale_inv
    assert method.moe_quant_config.w2_scale is layer.w2_weight_scale_inv


def _make_input_scale_param(shape: tuple[int, ...]) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.ones(shape), requires_grad=False)
    param.is_w4afp8_input_scale = True
    return param


def test_w4afp8_mapping_adds_checkpoint_input_scales() -> None:
    model = torch.nn.Module()
    model.register_parameter(
        "w13_input_scale",
        _make_input_scale_param((2, 2)),
    )

    mapping = RoutedExperts.make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=2,
    )

    assert (
        "experts.routed_experts.w13_",
        "experts.0.w1.",
        0,
        "w1",
    ) in mapping
    assert (
        "experts.routed_experts.w2_",
        "experts.1.w2.",
        1,
        "w2",
    ) in mapping
    assert (
        "experts.routed_experts.w13_",
        "experts.1.w3.",
        1,
        "w3",
    ) in mapping


def test_w4afp8_w1_w3_input_scales_do_not_overwrite_each_other() -> None:
    loader = object.__new__(RoutedExperts)
    loader.quant_config = W4AFP8Config()
    loader.quant_method = SimpleNamespace()
    loader._map_global_expert_id_to_local_expert_id = lambda expert_id: expert_id
    param = _make_input_scale_param((2, 2))

    for shard_id, value in (("w1", 0.25), ("w3", 0.75)):
        loaded = loader.weight_loader(
            param=param,
            loaded_weight=torch.tensor(value),
            weight_name=f"experts.1.{shard_id}.input_scale",
            shard_id=shard_id,
            expert_id=1,
            return_success=True,
        )
        assert loaded

    torch.testing.assert_close(param[1], torch.tensor([0.25, 0.75]))
