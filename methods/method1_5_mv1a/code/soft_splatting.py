"""Bilinear and depth-aware forward splatting implemented with PyTorch scatters."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from reprojection import RenderResult, project_source_rgbd


def render_soft_splat(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    K_source: torch.Tensor,
    K_target: torch.Tensor,
    T_target_source: torch.Tensor,
    target_size: Tuple[int, int],
    source_valid_mask: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
    visibility_tolerance_mm: float = 1.0,
    visibility_relative: float = 0.01,
    depth_softness: float = 8.0,
) -> RenderResult:
    """Render with bilinear or z-aware bilinear splatting.

    ``mode='bilinear'`` uses only bilinear footprint weights. In
    ``mode='soft_depth'``, a per-pixel nearest-Z buffer establishes visibility;
    contributions farther than ``tolerance_mm + relative*nearest_Z`` are
    rejected and the remainder receive an exponential depth weight.
    """

    if mode not in {"bilinear", "soft_depth"}:
        raise ValueError(f"unsupported soft-splat mode: {mode}")
    if visibility_tolerance_mm < 0 or visibility_relative < 0 or depth_softness < 0:
        raise ValueError("visibility parameters must be non-negative")

    projected = project_source_rgbd(
        source_rgb,
        source_depth,
        K_source,
        K_target,
        T_target_source,
        target_size,
        source_valid_mask=source_valid_mask,
    )
    target_height, target_width = target_size
    pixel_count = target_height * target_width
    device = source_rgb.device
    rgb_sum = torch.zeros((pixel_count, 3), dtype=torch.float32, device=device)
    depth_sum = torch.zeros((pixel_count,), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((pixel_count,), dtype=torch.float32, device=device)
    hit_count = torch.zeros((pixel_count,), dtype=torch.int64, device=device)
    source_uv_sum = torch.zeros((pixel_count, 2), dtype=torch.float32, device=device)

    if projected.target_uv.shape[0] > 0:
        u = projected.target_uv[:, 0]
        v = projected.target_uv[:, 1]
        x0 = torch.floor(u).to(torch.int64)
        y0 = torch.floor(v).to(torch.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = u - x0.to(u.dtype)
        wy = v - y0.to(v.dtype)

        neighbor_x = torch.stack((x0, x1, x0, x1), dim=1).reshape(-1)
        neighbor_y = torch.stack((y0, y0, y1, y1), dim=1).reshape(-1)
        bilinear_weight = torch.stack(
            ((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy), dim=1
        ).reshape(-1)
        point_index = torch.arange(u.shape[0], device=device).unsqueeze(1).expand(-1, 4).reshape(-1)
        contribution_valid = (
            (bilinear_weight > 0)
            & (neighbor_x >= 0)
            & (neighbor_x < target_width)
            & (neighbor_y >= 0)
            & (neighbor_y < target_height)
        )
        neighbor_x = neighbor_x[contribution_valid]
        neighbor_y = neighbor_y[contribution_valid]
        bilinear_weight = bilinear_weight[contribution_valid]
        point_index = point_index[contribution_valid]
        target_linear = neighbor_y * target_width + neighbor_x
        target_z = projected.target_points[point_index, 2]

        if mode == "soft_depth":
            nearest_z = torch.full((pixel_count,), float("inf"), dtype=torch.float32, device=device)
            nearest_z.scatter_reduce_(0, target_linear, target_z, reduce="amin", include_self=True)
            delta_z = target_z - nearest_z[target_linear]
            tolerance = visibility_tolerance_mm + visibility_relative * nearest_z[target_linear]
            visible = delta_z <= tolerance
            target_linear = target_linear[visible]
            point_index = point_index[visible]
            target_z = target_z[visible]
            delta_z = delta_z[visible]
            bilinear_weight = bilinear_weight[visible]
            tolerance = tolerance[visible].clamp_min(1e-6)
            weights = bilinear_weight * torch.exp(-depth_softness * delta_z / tolerance)
        else:
            weights = bilinear_weight

        rgb_sum.index_add_(0, target_linear, projected.colors[point_index] * weights[:, None])
        depth_sum.index_add_(0, target_linear, target_z * weights)
        source_uv_sum.index_add_(0, target_linear, projected.source_uv[point_index] * weights[:, None])
        weight_sum.index_add_(0, target_linear, weights)
        hit_count.index_add_(0, target_linear, torch.ones_like(target_linear, dtype=torch.int64))

    valid = weight_sum > 1e-8
    safe_weight = weight_sum.clamp_min(1e-8)
    rgb = rgb_sum / safe_weight[:, None]
    depth = depth_sum / safe_weight
    correspondence = source_uv_sum / safe_weight[:, None]
    rgb[~valid] = 0
    depth[~valid] = 0
    correspondence[~valid] = float("nan")
    # A bounded occupancy confidence; the raw sum remains recoverable as
    # -log(1-confidence) and does not pretend large disocclusions are valid.
    confidence = 1.0 - torch.exp(-weight_sum)
    coverage = float(valid.float().mean().item() * 100.0)
    stats = {
        "source_pixels": float(source_depth.numel()),
        "source_valid_pixels": float(projected.source_valid_pixels),
        "projected_in_bounds_pixels": float(projected.target_uv.shape[0]),
        "target_covered_pixels": float(valid.sum().item()),
        "target_coverage_percent": coverage,
    }
    return RenderResult(
        rgb=rgb.reshape(target_height, target_width, 3),
        depth=depth.reshape(target_height, target_width),
        valid_mask=valid.reshape(target_height, target_width),
        confidence=confidence.reshape(target_height, target_width),
        hit_count=hit_count.reshape(target_height, target_width),
        correspondence=correspondence.reshape(target_height, target_width, 2),
        source_projection=projected.source_projection,
        stats=stats,
    )
