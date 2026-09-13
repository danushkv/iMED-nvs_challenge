"""Source-only organized RGB-D surface geometry for Method 3.

This module is imported only after ``render_sequence.py`` has made the exact
MV1A geometry modules available. It does not read files or target imagery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from camera import backproject_depth, project_points, transform_points
from confidence import estimate_surface_confidence
from reprojection import ProjectedSource


@dataclass
class ProjectedSurfaceAttributes:
    """Surface attributes aligned with ``ProjectedSource`` point order."""

    normals_source: torch.Tensor  # [N,3]
    covariance_target: torch.Tensor  # [N,2,2], target-pixel covariance
    inverse_covariance_target: torch.Tensor  # [N,2,2]
    radius_target: torch.Tensor  # [N], sqrt(max covariance eigenvalue)
    surface_valid: torch.Tensor  # [N], otherwise use exact MV1A bilinear fallback
    geometry_confidence: torch.Tensor  # [N], source-only combined confidence
    normal_map_source: torch.Tensor  # [Hs,Ws,3]
    radius_map_source: torch.Tensor  # [Hs,Ws]
    surface_valid_map_source: torch.Tensor  # [Hs,Ws]
    geometry_confidence_map_source: torch.Tensor  # [Hs,Ws]
    depth_confidence_map_source: torch.Tensor  # [Hs,Ws]
    tangent_confidence_map_source: torch.Tensor  # [Hs,Ws]
    viewing_confidence_map_source: torch.Tensor  # [Hs,Ws]


def _clamp_symmetric_covariance(
    covariance: torch.Tensor,
    radius_min: float,
    radius_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clamp eigenvalues of batched symmetric 2x2 covariance matrices."""

    a = covariance[..., 0, 0]
    b = 0.5 * (covariance[..., 0, 1] + covariance[..., 1, 0])
    c = covariance[..., 1, 1]
    discriminant = torch.sqrt(((a - c) * 0.5).square() + b.square()).clamp_min(0.0)
    midpoint = 0.5 * (a + c)
    eigenvalue_major = (midpoint + discriminant).clamp(
        min=radius_min * radius_min,
        max=radius_max * radius_max,
    )
    eigenvalue_minor = (midpoint - discriminant).clamp(
        min=radius_min * radius_min,
        max=radius_max * radius_max,
    )

    angle = 0.5 * torch.atan2(2.0 * b, a - c)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    cos_sq = cos_angle.square()
    sin_sq = sin_angle.square()
    cross = cos_angle * sin_angle

    clamped = torch.empty_like(covariance)
    clamped[..., 0, 0] = eigenvalue_major * cos_sq + eigenvalue_minor * sin_sq
    clamped[..., 1, 1] = eigenvalue_major * sin_sq + eigenvalue_minor * cos_sq
    clamped[..., 0, 1] = (eigenvalue_major - eigenvalue_minor) * cross
    clamped[..., 1, 0] = clamped[..., 0, 1]
    return clamped, torch.sqrt(eigenvalue_major)


def estimate_projected_surface_attributes(
    source_depth: torch.Tensor,
    K_source: torch.Tensor,
    K_target: torch.Tensor,
    T_target_source: torch.Tensor,
    projected: ProjectedSource,
    source_valid_mask: Optional[torch.Tensor] = None,
    depth_discontinuity_mm: float = 2.0,
    depth_discontinuity_relative: float = 0.02,
    footprint_scale: float = 1.0,
    radius_min: float = 0.5,
    radius_max: float = 2.0,
    normal_epsilon: float = 1e-8,
) -> ProjectedSurfaceAttributes:
    """Estimate organized tangents, normals, and projected surfel covariance.

    Invalid or discontinuous neighbourhoods are not discarded from rendering.
    They are marked ``surface_valid=False`` so the renderer can fall back to
    MV1A's exact four-neighbour bilinear point footprint.
    """

    if depth_discontinuity_mm < 0 or depth_discontinuity_relative < 0:
        raise ValueError("depth-discontinuity thresholds must be non-negative")
    if footprint_scale <= 0:
        raise ValueError("footprint_scale must be positive")
    if not 0 < radius_min <= radius_max:
        raise ValueError("require 0 < radius_min <= radius_max")

    device = source_depth.device
    depth = source_depth.to(device=device, dtype=torch.float32)
    K_source = K_source.to(device=device, dtype=torch.float32)
    K_target = K_target.to(device=device, dtype=torch.float32)
    T_target_source = T_target_source.to(device=device, dtype=torch.float32)
    height, width = depth.shape

    base_valid = torch.isfinite(depth) & (depth > 0)
    if source_valid_mask is not None:
        base_valid &= source_valid_mask.to(device=device, dtype=torch.bool)

    points_source = backproject_depth(depth, K_source)
    points_target = transform_points(points_source.reshape(-1, 3), T_target_source).reshape(
        height, width, 3
    )
    target_uv = project_points(points_target.reshape(-1, 3), K_target).reshape(height, width, 2)

    tangent_u = torch.zeros_like(points_source)
    tangent_v = torch.zeros_like(points_source)
    tangent_u[:, 1:-1] = 0.5 * (points_source[:, 2:] - points_source[:, :-2])
    tangent_v[1:-1] = 0.5 * (points_source[2:] - points_source[:-2])

    normal_unnormalized = torch.linalg.cross(tangent_u, tangent_v, dim=-1)
    normal_norm = torch.linalg.vector_norm(normal_unnormalized, dim=-1)
    normals = normal_unnormalized / normal_norm.clamp_min(normal_epsilon)[..., None]

    neighbours_valid = torch.zeros_like(base_valid)
    neighbours_valid[1:-1, 1:-1] = (
        base_valid[1:-1, 1:-1]
        & base_valid[1:-1, :-2]
        & base_valid[1:-1, 2:]
        & base_valid[:-2, 1:-1]
        & base_valid[2:, 1:-1]
    )

    maximum_depth_delta = torch.full_like(depth, float("inf"))
    center = depth[1:-1, 1:-1]
    depth_deltas = torch.stack(
        (
            (depth[1:-1, :-2] - center).abs(),
            (depth[1:-1, 2:] - center).abs(),
            (depth[:-2, 1:-1] - center).abs(),
            (depth[2:, 1:-1] - center).abs(),
        ),
        dim=0,
    )
    maximum_depth_delta[1:-1, 1:-1] = depth_deltas.amax(dim=0)
    allowed_depth_delta = torch.maximum(
        torch.full_like(depth, depth_discontinuity_mm),
        depth_discontinuity_relative * depth,
    )

    target_neighbours_valid = torch.zeros_like(base_valid)
    finite_target_uv = torch.isfinite(target_uv).all(dim=-1)
    positive_target_z = torch.isfinite(points_target).all(dim=-1) & (points_target[..., 2] > 1e-6)
    target_ok = finite_target_uv & positive_target_z
    target_neighbours_valid[1:-1, 1:-1] = (
        target_ok[1:-1, 1:-1]
        & target_ok[1:-1, :-2]
        & target_ok[1:-1, 2:]
        & target_ok[:-2, 1:-1]
        & target_ok[2:, 1:-1]
    )

    surface_valid_map = (
        neighbours_valid
        & target_neighbours_valid
        & torch.isfinite(normals).all(dim=-1)
        & (normal_norm > normal_epsilon)
        & (maximum_depth_delta <= allowed_depth_delta)
    )

    projected_tangent_u = torch.zeros((height, width, 2), dtype=torch.float32, device=device)
    projected_tangent_v = torch.zeros_like(projected_tangent_u)
    projected_tangent_u[:, 1:-1] = 0.5 * (target_uv[:, 2:] - target_uv[:, :-2])
    projected_tangent_v[1:-1] = 0.5 * (target_uv[2:] - target_uv[:-2])

    # A projected tangent spans approximately one target pixel per source
    # pixel under identity. Half of each tangent is used as a conservative
    # Gaussian standard-deviation direction, limiting blur in M3-B.
    tangent_factor = 0.5 * footprint_scale
    e_u = tangent_factor * projected_tangent_u
    e_v = tangent_factor * projected_tangent_v
    covariance_map = e_u[..., :, None] * e_u[..., None, :] + e_v[..., :, None] * e_v[..., None, :]

    identity = torch.eye(2, dtype=torch.float32, device=device).expand(height, width, 2, 2)
    covariance_safe = torch.where(
        surface_valid_map[..., None, None],
        covariance_map,
        identity * (radius_min * radius_min),
    )
    covariance_safe, radius_map = _clamp_symmetric_covariance(
        covariance_safe,
        radius_min=radius_min,
        radius_max=radius_max,
    )

    determinant = (
        covariance_safe[..., 0, 0] * covariance_safe[..., 1, 1]
        - covariance_safe[..., 0, 1] * covariance_safe[..., 1, 0]
    )
    # PyTorch 2.1.2 (the confirmed MV1A runtime) accepts only one integer
    # dimension in Tensor.all(). Reduce the two covariance axes sequentially;
    # this is exactly equivalent to all(dim=(-1, -2)) on newer releases.
    covariance_finite = torch.isfinite(covariance_safe).all(dim=-1).all(dim=-1)
    surface_valid_map &= covariance_finite & (determinant > 1e-10)
    determinant = determinant.clamp_min(1e-10)
    inverse_map = torch.empty_like(covariance_safe)
    inverse_map[..., 0, 0] = covariance_safe[..., 1, 1] / determinant
    inverse_map[..., 1, 1] = covariance_safe[..., 0, 0] / determinant
    inverse_map[..., 0, 1] = -covariance_safe[..., 0, 1] / determinant
    inverse_map[..., 1, 0] = inverse_map[..., 0, 1]

    normals = torch.where(surface_valid_map[..., None], normals, torch.zeros_like(normals))
    radius_map = torch.where(surface_valid_map, radius_map, torch.zeros_like(radius_map))

    confidence_maps = estimate_surface_confidence(
        points_source=points_source,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        normals_source=normals,
        maximum_depth_delta=maximum_depth_delta,
        allowed_depth_delta=allowed_depth_delta,
        surface_valid=surface_valid_map,
        epsilon=normal_epsilon,
    )

    source_index = projected.source_linear_index
    covariance_flat = covariance_safe.reshape(-1, 2, 2)
    inverse_flat = inverse_map.reshape(-1, 2, 2)
    return ProjectedSurfaceAttributes(
        normals_source=normals.reshape(-1, 3)[source_index],
        covariance_target=covariance_flat[source_index],
        inverse_covariance_target=inverse_flat[source_index],
        radius_target=radius_map.reshape(-1)[source_index],
        surface_valid=surface_valid_map.reshape(-1)[source_index],
        geometry_confidence=confidence_maps.combined.reshape(-1)[source_index],
        normal_map_source=normals,
        radius_map_source=radius_map,
        surface_valid_map_source=surface_valid_map,
        geometry_confidence_map_source=confidence_maps.combined,
        depth_confidence_map_source=confidence_maps.depth,
        tangent_confidence_map_source=confidence_maps.tangent,
        viewing_confidence_map_source=confidence_maps.viewing,
    )
