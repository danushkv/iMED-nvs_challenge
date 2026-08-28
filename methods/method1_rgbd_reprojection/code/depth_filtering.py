"""Source-only depth filters for the Phase-11 ablation.

Filtering never invents values at invalid center pixels. Both implementations
are vectorized with PyTorch and operate on a single metric Z-depth image.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _patches(depth: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H,W], got {tuple(depth.shape)}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    radius = kernel_size // 2
    padded = F.pad(depth[None, None], (radius, radius, radius, radius), mode="replicate")
    return F.unfold(padded, kernel_size=kernel_size)[0].transpose(0, 1)


def median_depth_3x3(depth: torch.Tensor) -> torch.Tensor:
    """Median-filter valid source depths while preserving invalid centers."""

    center_valid = torch.isfinite(depth) & (depth > 0)
    center = depth.reshape(-1, 1)
    patches = _patches(depth, 3)
    neighbor_valid = torch.isfinite(patches) & (patches > 0)
    # Invalid neighbors cannot drag the median toward the invalid encoding.
    safe_patches = torch.where(neighbor_valid, patches, center.expand_as(patches))
    filtered = safe_patches.median(dim=1).values.reshape_as(depth)
    return torch.where(center_valid, filtered, depth)


def bilateral_depth(
    depth: torch.Tensor,
    kernel_size: int = 5,
    sigma_spatial_pixels: float = 2.0,
    sigma_depth_mm: float = 3.0,
) -> torch.Tensor:
    """Edge-preserving bilateral filtering of metric depth."""

    if sigma_spatial_pixels <= 0 or sigma_depth_mm <= 0:
        raise ValueError("bilateral sigmas must be positive")
    center_valid = torch.isfinite(depth) & (depth > 0)
    center = depth.reshape(-1, 1)
    patches = _patches(depth, kernel_size)
    neighbor_valid = torch.isfinite(patches) & (patches > 0)
    safe_patches = torch.where(neighbor_valid, patches, center.expand_as(patches))

    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, dtype=depth.dtype, device=depth.device)
    dy, dx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    spatial_weight = torch.exp(
        -(dx.square() + dy.square()) / (2.0 * sigma_spatial_pixels**2)
    ).reshape(1, -1)
    range_weight = torch.exp(
        -(safe_patches - center).square() / (2.0 * sigma_depth_mm**2)
    )
    weights = spatial_weight * range_weight * neighbor_valid.to(depth.dtype)
    weight_sum = weights.sum(dim=1).clamp_min(1e-12)
    filtered = (safe_patches * weights).sum(dim=1) / weight_sum
    filtered = filtered.reshape_as(depth)
    return torch.where(center_valid, filtered, depth)


def apply_depth_filter(
    depth: torch.Tensor,
    method: str = "none",
    bilateral_kernel_size: int = 5,
    bilateral_sigma_spatial_pixels: float = 2.0,
    bilateral_sigma_depth_mm: float = 3.0,
) -> torch.Tensor:
    """Apply one fixed, explicitly named Phase-11 source-depth treatment."""

    if method == "none":
        return depth
    if method == "median3":
        return median_depth_3x3(depth)
    if method == "bilateral":
        return bilateral_depth(
            depth,
            kernel_size=bilateral_kernel_size,
            sigma_spatial_pixels=bilateral_sigma_spatial_pixels,
            sigma_depth_mm=bilateral_sigma_depth_mm,
        )
    raise ValueError(f"unsupported depth-filter method: {method}")
