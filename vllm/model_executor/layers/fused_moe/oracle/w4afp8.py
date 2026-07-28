# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout helpers for GLM W4AFP8 routed experts."""

from __future__ import annotations

import torch


def interleave_w4afp8_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert ``[E, N, K / 128]`` BF16 scales to the kernel layout."""
    if scales.dim() != 3:
        raise ValueError(
            "W4AFP8 scales must have [experts, out_features, groups] layout, "
            f"got {tuple(scales.shape)}"
        )
    experts, out_features, groups = scales.shape
    pack = 4 if groups % 4 == 0 else 1
    return (
        scales.reshape(experts, out_features, groups // pack, pack)
        .permute(0, 2, 1, 3)
        .reshape(experts, groups // pack, out_features * pack)
        .contiguous()
    )


def make_w4afp8_moe_strides(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device | str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build CUTLASS pointer-array stride descriptors."""
    a_strides1 = torch.full(
        (num_experts, 3), hidden_size, device=device, dtype=torch.int64
    )
    d_strides1 = torch.full(
        (num_experts, 3),
        2 * intermediate_size,
        device=device,
        dtype=torch.int64,
    )
    a_strides2 = torch.full(
        (num_experts, 3),
        intermediate_size,
        device=device,
        dtype=torch.int64,
    )
    d_strides2 = torch.full(
        (num_experts, 3), hidden_size, device=device, dtype=torch.int64
    )
    return (
        a_strides1,
        a_strides1,
        d_strides1,
        a_strides2,
        a_strides2,
        d_strides2,
    )
