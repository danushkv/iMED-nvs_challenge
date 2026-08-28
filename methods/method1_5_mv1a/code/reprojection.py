"""Vectorized forward projection and nearest-pixel z-buffer rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from camera import backproject_depth, project_points, transform_points


@dataclass
class RenderResult:
    rgb: torch.Tensor  # [Ht,Wt,3], float32 [0,1]
    depth: torch.Tensor  # [Ht,Wt], metric camera-Z depth
    valid_mask: torch.Tensor  # [Ht,Wt], bool
    confidence: torch.Tensor  # [Ht,Wt], float32 [0,1]
    hit_count: torch.Tensor  # [Ht,Wt], int64
    correspondence: torch.Tensor  # [Ht,Wt,2], source [u,v], NaN if invalid
    source_projection: torch.Tensor  # [Hs,Ws,2], target [u,v], NaN if rejected
    stats: Dict[str, float]


@dataclass
class ProjectedSource:
    colors: torch.Tensor  # [N,3]
    target_points: torch.Tensor  # [N,3]
    target_uv: torch.Tensor  # [N,2]
    source_uv: torch.Tensor  # [N,2]
    source_linear_index: torch.Tensor  # [N]
    source_projection: torch.Tensor  # [Hs,Ws,2]
    source_valid_pixels: int


def _require_shapes(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    K_source: torch.Tensor,
    K_target: torch.Tensor,
    T_target_source: torch.Tensor,
    source_valid_mask: Optional[torch.Tensor],
) -> None:
    if source_rgb.ndim != 3 or source_rgb.shape[-1] != 3:
        raise ValueError(f"source_rgb must be [H,W,3], got {tuple(source_rgb.shape)}")
    if source_depth.shape != source_rgb.shape[:2]:
        raise ValueError("source_rgb and source_depth resolutions differ")
    if tuple(K_source.shape) != (3, 3) or tuple(K_target.shape) != (3, 3):
        raise ValueError("K_source and K_target must be 3x3")
    if tuple(T_target_source.shape) != (4, 4):
        raise ValueError("T_target_source must be 4x4")
    if source_valid_mask is not None and source_valid_mask.shape != source_depth.shape:
        raise ValueError("source_valid_mask must match source_depth")


def project_source_rgbd(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    K_source: torch.Tensor,
    K_target: torch.Tensor,
    T_target_source: torch.Tensor,
    target_size: Tuple[int, int],
    source_valid_mask: Optional[torch.Tensor] = None,
    min_target_z: float = 1e-6,
    projection_epsilon: float = 1e-4,
) -> ProjectedSource:
    """Backproject source RGB-D and return valid in-bounds target projections.

    The transform notation is explicit:

        X_target = T_target_source @ X_source
    """

    _require_shapes(source_rgb, source_depth, K_source, K_target, T_target_source, source_valid_mask)
    target_height, target_width = target_size
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"invalid target_size {target_size}")

    source_rgb = source_rgb.to(dtype=torch.float32)
    source_depth = source_depth.to(device=source_rgb.device, dtype=torch.float32)
    K_source = K_source.to(device=source_rgb.device, dtype=torch.float32)
    K_target = K_target.to(device=source_rgb.device, dtype=torch.float32)
    T_target_source = T_target_source.to(device=source_rgb.device, dtype=torch.float32)

    source_height, source_width = source_depth.shape
    valid = torch.isfinite(source_depth) & (source_depth > 0)
    if source_valid_mask is not None:
        valid &= source_valid_mask.to(device=source_rgb.device, dtype=torch.bool)
    source_valid_pixels = int(valid.sum().item())

    points_source = backproject_depth(source_depth, K_source).reshape(-1, 3)
    points_target = transform_points(points_source, T_target_source)
    target_uv_all = project_points(points_target, K_target)
    colors_all = source_rgb.reshape(-1, 3)

    v_grid, u_grid = torch.meshgrid(
        torch.arange(source_height, device=source_rgb.device, dtype=torch.float32),
        torch.arange(source_width, device=source_rgb.device, dtype=torch.float32),
        indexing="ij",
    )
    source_uv_all = torch.stack((u_grid, v_grid), dim=-1).reshape(-1, 2)
    valid_flat = valid.reshape(-1)
    valid_flat &= torch.isfinite(points_target).all(dim=-1)
    valid_flat &= torch.isfinite(target_uv_all).all(dim=-1)
    valid_flat &= points_target[:, 2] > min_target_z
    # The tiny margin prevents floating-point cancellation at exact image
    # borders from failing the identity gate. It is far below a pixel and does
    # not admit geometrically out-of-frame samples.
    valid_flat &= target_uv_all[:, 0] >= -projection_epsilon
    valid_flat &= target_uv_all[:, 0] <= target_width - 1 + projection_epsilon
    valid_flat &= target_uv_all[:, 1] >= -projection_epsilon
    valid_flat &= target_uv_all[:, 1] <= target_height - 1 + projection_epsilon

    source_linear_index = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
    source_projection = torch.full(
        (source_height * source_width, 2),
        float("nan"),
        dtype=torch.float32,
        device=source_rgb.device,
    )
    target_uv = target_uv_all[source_linear_index].clone()
    target_uv[:, 0].clamp_(0, target_width - 1)
    target_uv[:, 1].clamp_(0, target_height - 1)
    source_projection[source_linear_index] = target_uv
    return ProjectedSource(
        colors=colors_all[source_linear_index],
        target_points=points_target[source_linear_index],
        target_uv=target_uv,
        source_uv=source_uv_all[source_linear_index],
        source_linear_index=source_linear_index,
        source_projection=source_projection.reshape(source_height, source_width, 2),
        source_valid_pixels=source_valid_pixels,
    )


def render_nearest_zbuffer(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    K_source: torch.Tensor,
    K_target: torch.Tensor,
    T_target_source: torch.Tensor,
    target_size: Tuple[int, int],
    source_valid_mask: Optional[torch.Tensor] = None,
) -> RenderResult:
    """Rasterize rounded target pixels, keeping the smallest positive target Z."""

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
    rgb_flat = torch.zeros((pixel_count, 3), dtype=torch.float32, device=device)
    depth_flat = torch.full((pixel_count,), float("inf"), dtype=torch.float32, device=device)
    hits_flat = torch.zeros((pixel_count,), dtype=torch.int64, device=device)
    correspondence_flat = torch.full((pixel_count, 2), float("nan"), dtype=torch.float32, device=device)

    if projected.target_uv.shape[0] > 0:
        target_x = torch.round(projected.target_uv[:, 0]).to(torch.int64)
        target_y = torch.round(projected.target_uv[:, 1]).to(torch.int64)
        target_linear = target_y * target_width + target_x
        target_z = projected.target_points[:, 2]

        depth_flat.scatter_reduce_(0, target_linear, target_z, reduce="amin", include_self=True)
        hits_flat.scatter_add_(0, target_linear, torch.ones_like(target_linear, dtype=torch.int64))

        # Resolve exact equal-depth ties by the lowest source linear index.
        nearest = torch.isclose(target_z, depth_flat[target_linear], rtol=1e-6, atol=1e-6)
        sentinel = source_depth.numel()
        winner_source = torch.full((pixel_count,), sentinel, dtype=torch.int64, device=device)
        candidate_source = torch.where(
            nearest,
            projected.source_linear_index,
            torch.full_like(projected.source_linear_index, sentinel),
        )
        winner_source.scatter_reduce_(0, target_linear, candidate_source, reduce="amin", include_self=True)
        valid_flat = winner_source != sentinel
        rgb_flat[valid_flat] = source_rgb.reshape(-1, 3)[winner_source[valid_flat]]
        source_height, source_width = source_depth.shape
        winner_u = (winner_source[valid_flat] % source_width).to(torch.float32)
        winner_v = torch.div(winner_source[valid_flat], source_width, rounding_mode="floor").to(torch.float32)
        correspondence_flat[valid_flat] = torch.stack((winner_u, winner_v), dim=-1)
    else:
        valid_flat = torch.zeros((pixel_count,), dtype=torch.bool, device=device)

    depth_flat = torch.where(valid_flat, depth_flat, torch.zeros_like(depth_flat))
    confidence_flat = valid_flat.to(torch.float32)
    coverage = float(valid_flat.float().mean().item() * 100.0)
    stats = {
        "source_pixels": float(source_depth.numel()),
        "source_valid_pixels": float(projected.source_valid_pixels),
        "projected_in_bounds_pixels": float(projected.target_uv.shape[0]),
        "target_covered_pixels": float(valid_flat.sum().item()),
        "target_coverage_percent": coverage,
    }
    return RenderResult(
        rgb=rgb_flat.reshape(target_height, target_width, 3),
        depth=depth_flat.reshape(target_height, target_width),
        valid_mask=valid_flat.reshape(target_height, target_width),
        confidence=confidence_flat.reshape(target_height, target_width),
        hit_count=hits_flat.reshape(target_height, target_width),
        correspondence=correspondence_flat.reshape(target_height, target_width, 2),
        source_projection=projected.source_projection,
        stats=stats,
    )
