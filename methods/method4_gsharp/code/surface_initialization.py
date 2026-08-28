"""Source-only M3 geometry bootstrap for Method 4B.

This module reproduces the frozen M3 surface-geometry definitions needed for
3D initialization without importing or modifying the read-only Method 3
repository. It consumes only the legal Endoscope2 items already exposed by
``IMEDNVSDataset``. No target camera or Endoscope1 observation is accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from gsplat.cuda._math import _rotmat_to_quat


@dataclass
class SurfaceObservationBatch:
    means: torch.Tensor
    colors: torch.Tensor
    normals: torch.Tensor
    tangent_1: torch.Tensor
    tangent_scale_x: torch.Tensor
    tangent_scale_y: torch.Tensor
    confidence: torch.Tensor
    viewing_quality: torch.Tensor
    times: torch.Tensor
    pixels_uv: torch.Tensor
    counts: dict[str, int]


@dataclass
class SurfaceInitialization:
    means: torch.Tensor
    colors: torch.Tensor
    quats: torch.Tensor
    log_scales: torch.Tensor
    normals: torch.Tensor
    confidence: torch.Tensor
    times: torch.Tensor
    statistics: dict[str, Any]


def _normalize(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(
        epsilon
    )


def _empty_observations(
    device: torch.device,
    counts: dict[str, int],
) -> SurfaceObservationBatch:
    empty3 = torch.empty((0, 3), dtype=torch.float32, device=device)
    empty1 = torch.empty((0,), dtype=torch.float32, device=device)
    return SurfaceObservationBatch(
        means=empty3,
        colors=empty3.clone(),
        normals=empty3.clone(),
        tangent_1=empty3.clone(),
        tangent_scale_x=empty1,
        tangent_scale_y=empty1.clone(),
        confidence=empty1.clone(),
        viewing_quality=empty1.clone(),
        times=empty1.clone(),
        pixels_uv=torch.empty((0, 2), dtype=torch.int64, device=device),
        counts=counts,
    )


def extract_surface_observations(
    item: dict[str, Any],
    device: torch.device,
    candidate_cap: int | None,
    generator: torch.Generator,
    depth_discontinuity_mm: float = 2.0,
    depth_discontinuity_relative: float = 0.02,
    confidence_floor: float = 1e-6,
) -> SurfaceObservationBatch:
    """Extract M3-compatible source surfels from one Endoscope2 RGB-D frame."""

    if depth_discontinuity_mm < 0.0 or depth_discontinuity_relative < 0.0:
        raise ValueError("Depth-discontinuity thresholds must be non-negative")
    if confidence_floor < 0.0:
        raise ValueError("confidence_floor must be non-negative")
    if candidate_cap is not None and candidate_cap <= 0:
        raise ValueError("candidate_cap must be positive when provided")

    depth = item["depth"].to(device=device, dtype=torch.float32)
    image = item["image"].to(device=device, dtype=torch.float32)
    tissue = item["mask"].to(device=device, dtype=torch.float32) > 0.5
    K = item["K"].to(device=device, dtype=torch.float32)
    camtoworld = item["camtoworld"].to(device=device, dtype=torch.float32)
    time_value = item["time"].to(device=device, dtype=torch.float32).reshape(())
    height, width = depth.shape

    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    z = depth
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    points_camera = torch.stack((x, y, z), dim=-1)

    depth_valid = torch.isfinite(depth) & (depth > 0.0)
    masked_valid = depth_valid & tissue
    neighbours_valid = torch.zeros_like(masked_valid)
    neighbours_valid[1:-1, 1:-1] = (
        masked_valid[1:-1, 1:-1]
        & masked_valid[1:-1, :-2]
        & masked_valid[1:-1, 2:]
        & masked_valid[:-2, 1:-1]
        & masked_valid[2:, 1:-1]
    )

    tangent_u = torch.zeros_like(points_camera)
    tangent_v = torch.zeros_like(points_camera)
    tangent_u[:, 1:-1] = 0.5 * (points_camera[:, 2:] - points_camera[:, :-2])
    tangent_v[1:-1] = 0.5 * (points_camera[2:] - points_camera[:-2])
    normal_raw = torch.linalg.cross(tangent_u, tangent_v, dim=-1)
    normal_norm = torch.linalg.vector_norm(normal_raw, dim=-1)
    normals_camera = normal_raw / normal_norm.clamp_min(1e-8)[..., None]

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
    surface_valid = (
        neighbours_valid
        & torch.isfinite(points_camera).all(dim=-1)
        & torch.isfinite(tangent_u).all(dim=-1)
        & torch.isfinite(tangent_v).all(dim=-1)
        & torch.isfinite(normals_camera).all(dim=-1)
        & (normal_norm > 1e-8)
        & (maximum_depth_delta <= allowed_depth_delta)
    )

    # These are the frozen M3-C source-only confidence definitions. M4-B1
    # uses confidence only for fusion/ranking; opacity remains exactly 0.1.
    depth_confidence = (
        1.0 - maximum_depth_delta / allowed_depth_delta.clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    tangent_area = torch.linalg.vector_norm(normal_raw, dim=-1)
    tangent_denominator = (
        torch.linalg.vector_norm(tangent_u, dim=-1)
        * torch.linalg.vector_norm(tangent_v, dim=-1)
    )
    tangent_confidence = (
        tangent_area / tangent_denominator.clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    camera_view_direction = -points_camera / torch.linalg.vector_norm(
        points_camera, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    viewing_quality = (
        (normals_camera * camera_view_direction).sum(dim=-1).abs().clamp(0.0, 1.0)
    )
    confidence = depth_confidence * tangent_confidence * viewing_quality
    confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    confidence_valid = surface_valid & (confidence >= confidence_floor)

    counts = {
        "raw_observations": int(height * width),
        "after_masking": int(masked_valid.sum().item()),
        "after_confidence_filtering": int(confidence_valid.sum().item()),
    }
    flat_indices = torch.nonzero(confidence_valid.reshape(-1), as_tuple=False).squeeze(-1)
    if flat_indices.numel() == 0:
        counts["after_candidate_budget"] = 0
        return _empty_observations(device, counts)

    if candidate_cap is not None and flat_indices.numel() > candidate_cap:
        selection = torch.randperm(
            flat_indices.numel(), device=device, generator=generator
        )[:candidate_cap]
        flat_indices = flat_indices[selection]
    counts["after_candidate_budget"] = int(flat_indices.numel())

    rows = torch.div(flat_indices, width, rounding_mode="floor")
    cols = flat_indices.remainder(width)
    pixels_uv = torch.stack((cols, rows), dim=-1)
    points_selected = points_camera.reshape(-1, 3)[flat_indices]
    tangent_u_selected = tangent_u.reshape(-1, 3)[flat_indices]
    tangent_v_selected = tangent_v.reshape(-1, 3)[flat_indices]
    normal_selected = normals_camera.reshape(-1, 3)[flat_indices]

    rotation = camtoworld[:3, :3]
    translation = camtoworld[:3, 3]
    means_world = points_selected @ rotation.T + translation
    tangent_u_world = tangent_u_selected @ rotation.T
    tangent_v_world = tangent_v_selected @ rotation.T
    normals_world = _normalize(normal_selected @ rotation.T)

    # Resolve the sign ambiguity consistently: normals face the source camera.
    direction_to_camera = _normalize(translation[None] - means_world)
    flip = (normals_world * direction_to_camera).sum(dim=-1) < 0.0
    normals_world = torch.where(flip[:, None], -normals_world, normals_world)
    tangent_1_world = _normalize(tangent_u_world)

    # M3 uses half of each central-difference tangent as the conservative
    # Gaussian standard-deviation direction in image space. Preserve the same
    # half-footprint convention in 3D world units.
    scale_x = 0.5 * torch.linalg.vector_norm(tangent_u_world, dim=-1)
    scale_y = 0.5 * torch.linalg.vector_norm(tangent_v_world, dim=-1)
    return SurfaceObservationBatch(
        means=means_world,
        colors=image.reshape(-1, 3)[flat_indices].clamp(0.0, 1.0),
        normals=normals_world,
        tangent_1=tangent_1_world,
        tangent_scale_x=scale_x,
        tangent_scale_y=scale_y,
        confidence=confidence.reshape(-1)[flat_indices],
        viewing_quality=viewing_quality.reshape(-1)[flat_indices],
        times=time_value.expand(flat_indices.numel()),
        pixels_uv=pixels_uv,
        counts=counts,
    )


def _group_sum(
    values: torch.Tensor,
    inverse: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    output = torch.zeros(
        (group_count,) + values.shape[1:], dtype=values.dtype, device=values.device
    )
    output.index_add_(0, inverse, values)
    return output


def rotation_matrices_to_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    """Convert local-to-world rotation matrices to canonical gsplat wxyz."""

    quats = _rotmat_to_quat(rotation)
    quats = _normalize(quats)
    return torch.where(quats[:, :1] < 0.0, -quats, quats)


def _fuse_voxels(
    means: torch.Tensor,
    colors: torch.Tensor,
    normals: torch.Tensor,
    tangent_1: torch.Tensor,
    scale_x: torch.Tensor,
    scale_y: torch.Tensor,
    confidence: torch.Tensor,
    viewing_quality: torch.Tensor,
    times: torch.Tensor,
    voxel_size: float,
) -> tuple[dict[str, torch.Tensor], int]:
    voxel_coordinates = torch.floor(means / voxel_size).to(torch.int64)
    _, inverse = torch.unique(voxel_coordinates, dim=0, return_inverse=True)
    group_count = int(inverse.max().item()) + 1

    base_weight = confidence.clamp_min(1e-6) * viewing_quality.clamp_min(1e-3)
    preliminary_normal = _normalize(
        _group_sum(normals * base_weight[:, None], inverse, group_count)
    )
    normal_agreement = (
        normals * preliminary_normal[inverse]
    ).sum(dim=-1).clamp(0.0, 1.0)
    weight = base_weight * (0.5 + 0.5 * normal_agreement)
    weight_sum = _group_sum(weight[:, None], inverse, group_count).squeeze(-1)
    denominator = weight_sum.clamp_min(1e-8)

    def weighted(values: torch.Tensor) -> torch.Tensor:
        expanded_weight = weight.reshape((-1,) + (1,) * (values.ndim - 1))
        summed = _group_sum(values * expanded_weight, inverse, group_count)
        expanded_denominator = denominator.reshape(
            (-1,) + (1,) * (values.ndim - 1)
        )
        return summed / expanded_denominator

    fused_normal = _normalize(weighted(normals))
    fused_tangent = weighted(tangent_1)
    fused_tangent = fused_tangent - (
        fused_tangent * fused_normal
    ).sum(dim=-1, keepdim=True) * fused_normal
    tangent_norm = torch.linalg.vector_norm(fused_tangent, dim=-1)
    valid_basis = (
        torch.isfinite(fused_normal).all(dim=-1)
        & torch.isfinite(fused_tangent).all(dim=-1)
        & (tangent_norm > 1e-8)
        & (weight_sum > 0.0)
    )
    fused_tangent = _normalize(fused_tangent)

    fused = {
        "means": weighted(means)[valid_basis],
        "colors": weighted(colors).clamp(0.0, 1.0)[valid_basis],
        "normals": fused_normal[valid_basis],
        "tangent_1": fused_tangent[valid_basis],
        "scale_x": weighted(scale_x)[valid_basis],
        "scale_y": weighted(scale_y)[valid_basis],
        "confidence": weighted(confidence).clamp(0.0, 1.0)[valid_basis],
        "times": weighted(times).clamp(0.0, 1.0)[valid_basis],
    }
    return fused, group_count


def build_surface_initialization(
    dataset: Any,
    device: torch.device,
    max_points: int = 50_000,
    voxel_size: float = 0.5,
    thickness_ratio: float = 0.2,
    candidate_multiplier: float = 4.0,
    seed: int = 42,
    frame_indices: Iterable[int] | None = None,
) -> SurfaceInitialization:
    """Build a deterministic, source-only geometry-derived Gaussian set."""

    if max_points < 4:
        raise ValueError("max_points must be at least 4")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if not 0.0 < thickness_ratio < 1.0:
        raise ValueError("thickness_ratio must lie strictly between 0 and 1")
    if candidate_multiplier < 1.0:
        raise ValueError("candidate_multiplier must be at least 1")

    indices = list(range(len(dataset))) if frame_indices is None else list(frame_indices)
    if not indices:
        raise ValueError("No source frames selected for M4-B initialization")
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError("M4-B initialization frame index is outside the dataset")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate M4-B initialization frame indices")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    candidate_cap = int(
        math.ceil(max_points * candidate_multiplier / max(len(indices), 1))
    )
    chunks: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "means",
            "colors",
            "normals",
            "tangent_1",
            "tangent_scale_x",
            "tangent_scale_y",
            "confidence",
            "viewing_quality",
            "times",
        )
    }
    totals = {
        "raw_observations": 0,
        "after_masking": 0,
        "after_confidence_filtering": 0,
        "after_candidate_budget": 0,
    }
    frames_with_surface = 0
    for index in indices:
        batch = extract_surface_observations(
            dataset[index],
            device=device,
            candidate_cap=candidate_cap,
            generator=generator,
        )
        for key in totals:
            totals[key] += batch.counts[key]
        if batch.means.numel() == 0:
            continue
        frames_with_surface += 1
        for key in chunks:
            chunks[key].append(getattr(batch, key))

    if not chunks["means"]:
        raise RuntimeError("No valid M3-derived tissue surfels were found")
    concatenated = {key: torch.cat(value, dim=0) for key, value in chunks.items()}
    fused, voxel_count = _fuse_voxels(
        means=concatenated["means"],
        colors=concatenated["colors"],
        normals=concatenated["normals"],
        tangent_1=concatenated["tangent_1"],
        scale_x=concatenated["tangent_scale_x"],
        scale_y=concatenated["tangent_scale_y"],
        confidence=concatenated["confidence"],
        viewing_quality=concatenated["viewing_quality"],
        times=concatenated["times"],
        voxel_size=voxel_size,
    )
    after_basis_validation = int(fused["means"].shape[0])
    if after_basis_validation < 4:
        raise RuntimeError(
            f"Only {after_basis_validation} valid fused M4-B surface elements"
        )

    if after_basis_validation > max_points:
        selected = torch.randperm(
            after_basis_validation, device=device, generator=generator
        )[:max_points]
        selected = selected.sort().values
        fused = {key: value[selected] for key, value in fused.items()}

    # A fused voxel is itself a surface footprint. Prevent the original
    # per-pixel half-footprint from becoming smaller than half a voxel after
    # multi-frame fusion and final point budgeting.
    tangent_floor = 0.5 * voxel_size
    scale_x = fused["scale_x"].clamp_min(tangent_floor)
    scale_y = fused["scale_y"].clamp_min(tangent_floor)
    scale_z = torch.minimum(scale_x, scale_y) * thickness_ratio
    scales = torch.stack((scale_x, scale_y, scale_z), dim=-1).clamp_min(1e-7)

    normal = _normalize(fused["normals"])
    tangent_1 = fused["tangent_1"] - (
        fused["tangent_1"] * normal
    ).sum(dim=-1, keepdim=True) * normal
    tangent_1 = _normalize(tangent_1)
    tangent_2 = _normalize(torch.linalg.cross(normal, tangent_1, dim=-1))
    tangent_1 = _normalize(torch.linalg.cross(tangent_2, normal, dim=-1))
    rotation = torch.stack((tangent_1, tangent_2, normal), dim=-1)
    quats = rotation_matrices_to_wxyz(rotation)

    means = fused["means"].contiguous()
    colors = fused["colors"].contiguous()
    final_count = int(means.shape[0])
    mins = means.amin(dim=0).detach().cpu().tolist()
    maxs = means.amax(dim=0).detach().cpu().tolist()
    extent = (means.amax(dim=0) - means.amin(dim=0)).detach()
    statistics: dict[str, Any] = {
        **totals,
        "after_deduplication": voxel_count,
        "after_basis_validation": after_basis_validation,
        "final_gaussian_count": final_count,
        "count": final_count,
        "frames_considered": len(indices),
        "frames_with_surface": frames_with_surface,
        "candidate_cap_per_frame": candidate_cap,
        "candidate_multiplier": candidate_multiplier,
        "voxel_size_world_units": voxel_size,
        "thickness_ratio": thickness_ratio,
        "tangent_scale_floor": tangent_floor,
        "xyz_min": mins,
        "xyz_max": maxs,
        "scene_extent_xyz": extent.cpu().tolist(),
        "scene_extent_max": float(extent.max().item()),
        "median_tangent_scale_x": float(scale_x.median().item()),
        "median_tangent_scale_y": float(scale_y.median().item()),
        "median_normal_scale": float(scale_z.median().item()),
        "mean_geometry_confidence": float(fused["confidence"].mean().item()),
        "mean_source_time": float(fused["times"].mean().item()),
        "orientation": "local axes [tangent_1, tangent_2, normal]",
        "quaternion_convention": "gsplat wxyz",
        "target_rgb_access": False,
        "target_camera_access": False,
    }
    return SurfaceInitialization(
        means=means,
        colors=colors,
        quats=quats.contiguous(),
        log_scales=scales.log().contiguous(),
        normals=normal.contiguous(),
        confidence=fused["confidence"].contiguous(),
        times=fused["times"].contiguous(),
        statistics=statistics,
    )
