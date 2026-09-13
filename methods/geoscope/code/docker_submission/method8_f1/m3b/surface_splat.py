"""Bounded surface-aware target-space splatting for Method 3 M3-B/M3-C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from reprojection import RenderResult, project_source_rgbd
from surface_geometry import ProjectedSurfaceAttributes, estimate_projected_surface_attributes


@dataclass
class SurfaceRenderOutput:
    result: RenderResult
    normal_map_source: torch.Tensor
    radius_map_source: torch.Tensor
    surface_valid_map_source: torch.Tensor
    surface_used_map_source: torch.Tensor
    geometry_confidence_map_source: torch.Tensor
    depth_confidence_map_source: torch.Tensor
    tangent_confidence_map_source: torch.Tensor
    viewing_confidence_map_source: torch.Tensor
    surface_recovered_mask_target: torch.Tensor


def _empty_contributions(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.empty((0,), dtype=torch.float32, device=device),
    )


def _build_contributions(
    projected,
    attributes: ProjectedSurfaceAttributes,
    start: int,
    end: int,
    target_height: int,
    target_width: int,
    support_size: int,
    mahalanobis_cutoff: float,
    confidence_aware: bool,
    geometry_confidence_threshold: float,
    geometry_confidence_power: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create bounded contributions for a projected-point chunk.

    Valid surfels use an anisotropic Gaussian. Points whose local surface is
    unreliable retain MV1A's exact four-neighbour bilinear footprint.
    """

    device = projected.target_uv.device
    if start >= end:
        return _empty_contributions(device)

    uv = projected.target_uv[start:end]
    surface_valid = attributes.surface_valid[start:end]
    geometry_confidence = attributes.geometry_confidence[start:end]
    if confidence_aware:
        surface_valid = surface_valid & (geometry_confidence >= geometry_confidence_threshold)
    point_ids = torch.arange(start, end, dtype=torch.int64, device=device)
    target_linear_parts = []
    point_index_parts = []
    weight_parts = []

    if bool(surface_valid.any()):
        support_radius = support_size // 2
        offsets = torch.arange(-support_radius, support_radius + 1, device=device)
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_x = offset_x.reshape(1, -1)
        offset_y = offset_y.reshape(1, -1)

        center_x = torch.round(uv[:, 0]).to(torch.int64)[:, None]
        center_y = torch.round(uv[:, 1]).to(torch.int64)[:, None]
        target_x = center_x + offset_x
        target_y = center_y + offset_y
        dx = target_x.to(torch.float32) - uv[:, 0:1]
        dy = target_y.to(torch.float32) - uv[:, 1:2]
        inverse = attributes.inverse_covariance_target[start:end]
        mahalanobis = (
            inverse[:, 0, 0:1] * dx.square()
            + 2.0 * inverse[:, 0, 1:2] * dx * dy
            + inverse[:, 1, 1:2] * dy.square()
        )
        weights = torch.exp(-0.5 * mahalanobis)
        contribution_valid = (
            surface_valid[:, None]
            & (target_x >= 0)
            & (target_x < target_width)
            & (target_y >= 0)
            & (target_y < target_height)
            & torch.isfinite(weights)
            & (mahalanobis <= mahalanobis_cutoff)
            & (weights > 1e-8)
        )
        weights = torch.where(contribution_valid, weights, torch.zeros_like(weights))
        # Conserve each point's total weight so larger footprints do not gain
        # arbitrary influence merely by touching more target pixels.
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if confidence_aware:
            # Apply confidence after footprint normalization. Applying it
            # before normalization would cancel the per-point scalar and have
            # no effect on target-space fusion.
            weights = weights * geometry_confidence.clamp(0.0, 1.0).pow(
                geometry_confidence_power
            )[:, None]
        contribution_valid &= weights > 0
        if bool(contribution_valid.any()):
            expanded_point_ids = point_ids[:, None].expand_as(target_x)
            target_linear_parts.append((target_y * target_width + target_x)[contribution_valid])
            point_index_parts.append(expanded_point_ids[contribution_valid])
            weight_parts.append(weights[contribution_valid])

    fallback = ~surface_valid
    if bool(fallback.any()):
        u = uv[:, 0]
        v = uv[:, 1]
        x0 = torch.floor(u).to(torch.int64)
        y0 = torch.floor(v).to(torch.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = u - x0.to(torch.float32)
        wy = v - y0.to(torch.float32)
        target_x = torch.stack((x0, x1, x0, x1), dim=1)
        target_y = torch.stack((y0, y0, y1, y1), dim=1)
        weights = torch.stack(
            ((1.0 - wx) * (1.0 - wy), wx * (1.0 - wy), (1.0 - wx) * wy, wx * wy),
            dim=1,
        )
        contribution_valid = (
            fallback[:, None]
            & (weights > 0)
            & (target_x >= 0)
            & (target_x < target_width)
            & (target_y >= 0)
            & (target_y < target_height)
        )
        if bool(contribution_valid.any()):
            expanded_point_ids = point_ids[:, None].expand_as(target_x)
            target_linear_parts.append((target_y * target_width + target_x)[contribution_valid])
            point_index_parts.append(expanded_point_ids[contribution_valid])
            weight_parts.append(weights[contribution_valid])

    if not target_linear_parts:
        return _empty_contributions(device)
    return (
        torch.cat(target_linear_parts),
        torch.cat(point_index_parts),
        torch.cat(weight_parts).to(torch.float32),
    )


def _mv1a_bilinear_support(projected, target_height: int, target_width: int) -> torch.Tensor:
    """Return the raw target support of MV1A's four-pixel footprint."""

    device = projected.target_uv.device
    support = torch.zeros((target_height * target_width,), dtype=torch.bool, device=device)
    if projected.target_uv.shape[0] == 0:
        return support.reshape(target_height, target_width)
    u = projected.target_uv[:, 0]
    v = projected.target_uv[:, 1]
    x0 = torch.floor(u).to(torch.int64)
    y0 = torch.floor(v).to(torch.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = u - x0.to(torch.float32)
    wy = v - y0.to(torch.float32)
    target_x = torch.stack((x0, x1, x0, x1), dim=1).reshape(-1)
    target_y = torch.stack((y0, y0, y1, y1), dim=1).reshape(-1)
    weights = torch.stack(
        ((1.0 - wx) * (1.0 - wy), wx * (1.0 - wy), (1.0 - wx) * wy, wx * wy),
        dim=1,
    ).reshape(-1)
    valid = (
        (weights > 0)
        & (target_x >= 0)
        & (target_x < target_width)
        & (target_y >= 0)
        & (target_y < target_height)
    )
    support[target_y[valid] * target_width + target_x[valid]] = True
    return support.reshape(target_height, target_width)


def render_surface_splat(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    K_source: torch.Tensor,
    K_target: torch.Tensor,
    T_target_source: torch.Tensor,
    target_size: Tuple[int, int],
    source_valid_mask: Optional[torch.Tensor] = None,
    visibility_tolerance_mm: float = 1.0,
    visibility_relative: float = 0.01,
    depth_softness: float = 8.0,
    depth_discontinuity_mm: float = 2.0,
    depth_discontinuity_relative: float = 0.02,
    footprint_scale: float = 1.0,
    radius_min: float = 0.5,
    radius_max: float = 2.0,
    support_size: int = 5,
    mahalanobis_cutoff: float = 9.0,
    chunk_size: int = 65536,
    confidence_aware: bool = False,
    geometry_confidence_threshold: float = 0.25,
    geometry_confidence_power: float = 1.0,
) -> SurfaceRenderOutput:
    """Render organized source RGB-D as bounded projected surface elements."""

    if visibility_tolerance_mm < 0 or visibility_relative < 0 or depth_softness < 0:
        raise ValueError("visibility parameters must be non-negative")
    if support_size < 3 or support_size % 2 == 0:
        raise ValueError("support_size must be an odd integer >= 3")
    if support_size > 7:
        raise ValueError("support_size is intentionally bounded to at most 7")
    if mahalanobis_cutoff <= 0 or chunk_size <= 0:
        raise ValueError("mahalanobis_cutoff and chunk_size must be positive")
    if not 0.0 <= geometry_confidence_threshold <= 1.0:
        raise ValueError("geometry_confidence_threshold must be in [0,1]")
    if geometry_confidence_power < 0:
        raise ValueError("geometry_confidence_power must be non-negative")

    projected = project_source_rgbd(
        source_rgb,
        source_depth,
        K_source,
        K_target,
        T_target_source,
        target_size,
        source_valid_mask=source_valid_mask,
    )
    attributes = estimate_projected_surface_attributes(
        source_depth=source_depth,
        K_source=K_source,
        K_target=K_target,
        T_target_source=T_target_source,
        projected=projected,
        source_valid_mask=source_valid_mask,
        depth_discontinuity_mm=depth_discontinuity_mm,
        depth_discontinuity_relative=depth_discontinuity_relative,
        footprint_scale=footprint_scale,
        radius_min=radius_min,
        radius_max=radius_max,
    )

    target_height, target_width = target_size
    pixel_count = target_height * target_width
    device = source_rgb.device
    nearest_z = torch.full((pixel_count,), float("inf"), dtype=torch.float32, device=device)
    point_count = projected.target_uv.shape[0]

    # Pass 1: establish the closest target-camera depth for every touched
    # target pixel without storing all surfel contributions simultaneously.
    for start in range(0, point_count, chunk_size):
        end = min(start + chunk_size, point_count)
        target_linear, point_index, _ = _build_contributions(
            projected,
            attributes,
            start,
            end,
            target_height,
            target_width,
            support_size,
            mahalanobis_cutoff,
            confidence_aware,
            geometry_confidence_threshold,
            geometry_confidence_power,
        )
        if target_linear.numel() > 0:
            target_z = projected.target_points[point_index, 2]
            nearest_z.scatter_reduce_(0, target_linear, target_z, reduce="amin", include_self=True)

    rgb_sum = torch.zeros((pixel_count, 3), dtype=torch.float32, device=device)
    depth_sum = torch.zeros((pixel_count,), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((pixel_count,), dtype=torch.float32, device=device)
    source_uv_sum = torch.zeros((pixel_count, 2), dtype=torch.float32, device=device)
    hit_count = torch.zeros((pixel_count,), dtype=torch.int64, device=device)

    # Pass 2: reproduce contributions and apply MV1A's unchanged depth-aware
    # visibility weighting before accumulation.
    for start in range(0, point_count, chunk_size):
        end = min(start + chunk_size, point_count)
        target_linear, point_index, footprint_weight = _build_contributions(
            projected,
            attributes,
            start,
            end,
            target_height,
            target_width,
            support_size,
            mahalanobis_cutoff,
            confidence_aware,
            geometry_confidence_threshold,
            geometry_confidence_power,
        )
        if target_linear.numel() == 0:
            continue
        target_z = projected.target_points[point_index, 2]
        delta_z = target_z - nearest_z[target_linear]
        tolerance = visibility_tolerance_mm + visibility_relative * nearest_z[target_linear]
        visible = delta_z <= tolerance
        if not bool(visible.any()):
            continue
        target_linear = target_linear[visible]
        point_index = point_index[visible]
        target_z = target_z[visible]
        delta_z = delta_z[visible]
        tolerance = tolerance[visible].clamp_min(1e-6)
        weights = footprint_weight[visible] * torch.exp(-depth_softness * delta_z / tolerance)

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
    confidence = 1.0 - torch.exp(-weight_sum)

    mv1a_support = _mv1a_bilinear_support(projected, target_height, target_width)
    valid_map = valid.reshape(target_height, target_width)
    recovered_mask = valid_map & ~mv1a_support
    surface_projected = attributes.surface_valid
    if confidence_aware:
        surface_used = surface_projected & (
            attributes.geometry_confidence >= geometry_confidence_threshold
        )
    else:
        surface_used = surface_projected
    valid_radius = attributes.radius_target[surface_used]
    valid_geometry_confidence = attributes.geometry_confidence[surface_projected]
    surface_used_map_source = attributes.surface_valid_map_source
    if confidence_aware:
        surface_used_map_source = surface_used_map_source & (
            attributes.geometry_confidence_map_source >= geometry_confidence_threshold
        )
    coverage = float(valid.float().mean().item() * 100.0)
    stats = {
        "source_pixels": float(source_depth.numel()),
        "source_valid_pixels": float(projected.source_valid_pixels),
        "projected_in_bounds_pixels": float(point_count),
        "projected_surface_pixels": float(surface_projected.sum().item()),
        "projected_surface_used_pixels": float(surface_used.sum().item()),
        "projected_bilinear_fallback_pixels": float((~surface_used).sum().item()),
        "source_surface_valid_percent": float(
            attributes.surface_valid_map_source.float().mean().item() * 100.0
        ),
        "source_surface_used_percent": float(
            surface_used_map_source.float().mean().item() * 100.0
        ),
        "mean_source_geometry_confidence": float(
            attributes.geometry_confidence_map_source.mean().item()
        ),
        "mean_valid_surface_geometry_confidence": (
            float(valid_geometry_confidence.mean().item())
            if valid_geometry_confidence.numel()
            else 0.0
        ),
        "mean_projected_surface_radius_pixels": (
            float(valid_radius.mean().item()) if valid_radius.numel() else 0.0
        ),
        "target_covered_pixels": float(valid.sum().item()),
        "target_coverage_percent": coverage,
        "surface_recovered_target_pixels": float(recovered_mask.sum().item()),
        "surface_recovered_target_percent": float(recovered_mask.float().mean().item() * 100.0),
    }
    result = RenderResult(
        rgb=rgb.reshape(target_height, target_width, 3),
        depth=depth.reshape(target_height, target_width),
        valid_mask=valid_map,
        confidence=confidence.reshape(target_height, target_width),
        hit_count=hit_count.reshape(target_height, target_width),
        correspondence=correspondence.reshape(target_height, target_width, 2),
        source_projection=projected.source_projection,
        stats=stats,
    )
    return SurfaceRenderOutput(
        result=result,
        normal_map_source=attributes.normal_map_source,
        radius_map_source=attributes.radius_map_source,
        surface_valid_map_source=attributes.surface_valid_map_source,
        surface_used_map_source=surface_used_map_source,
        geometry_confidence_map_source=attributes.geometry_confidence_map_source,
        depth_confidence_map_source=attributes.depth_confidence_map_source,
        tangent_confidence_map_source=attributes.tangent_confidence_map_source,
        viewing_confidence_map_source=attributes.viewing_confidence_map_source,
        surface_recovered_mask_target=recovered_mask,
    )
