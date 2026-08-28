#!/usr/bin/env python3
"""Source-only numerical gate for Method 4B initialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


M4_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = M4_ROOT.parent
GSPLAT_ROOT = M4_ROOT / "third_party" / "gsplat"
for path in (REPO_ROOT, GSPLAT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gsplat.utils import normalized_quat_to_rotmat  # noqa: E402
from method4_gsharp.datasets.imed_nvs import (  # noqa: E402
    IMEDNVSDataset,
    IMEDNVSParser,
    backproject_pixels_to_world,
)
from method4_gsharp.surface_initialization import (  # noqa: E402
    build_surface_initialization,
    extract_surface_observations,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate M4-B source initialization")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--world-scale", type=float, default=1.0)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--thickness-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M4-B validation requires CUDA")
    parser = IMEDNVSParser(
        args.data_root / args.sequence,
        world_scale=args.world_scale,
    )
    dataset = IMEDNVSDataset(parser)
    if not 0 <= args.frame_index < len(dataset):
        raise IndexError(args.frame_index)

    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    item = dataset[args.frame_index]
    observations = extract_surface_observations(
        item=item,
        device=device,
        candidate_cap=max(4096, args.samples),
        generator=generator,
    )
    if len(observations.means) < args.samples:
        raise RuntimeError(
            f"Only {len(observations.means)} validated source surfels in test frame"
        )

    selected_uv = observations.pixels_uv[: args.samples].detach().cpu().numpy()
    reference_world = backproject_pixels_to_world(
        depth=item["depth"].numpy(),
        K=item["K"].numpy(),
        camtoworld=item["camtoworld"].numpy(),
        pixels_uv=selected_uv,
    )
    observed_world = observations.means[: args.samples].detach().cpu().numpy()
    world_error = observed_world.astype(np.float64) - reference_world
    world_max_abs_error = float(np.abs(world_error).max())
    world_max_l2_error = float(np.linalg.norm(world_error, axis=-1).max())

    initialization = build_surface_initialization(
        dataset=dataset,
        device=device,
        max_points=2048,
        voxel_size=args.voxel_size,
        thickness_ratio=args.thickness_ratio,
        candidate_multiplier=2.0,
        seed=42,
        frame_indices=[args.frame_index],
    )
    rotations = normalized_quat_to_rotmat(initialization.quats)
    identity = torch.eye(3, dtype=rotations.dtype, device=device)
    orthogonality_error = float(
        (rotations.transpose(-2, -1) @ rotations - identity).abs().max().item()
    )
    determinant_error = float((torch.linalg.det(rotations) - 1.0).abs().max().item())
    normal_axis_error = float(
        (rotations[..., :, 2] - initialization.normals).abs().max().item()
    )
    quaternion_norm_error = float(
        (torch.linalg.vector_norm(initialization.quats, dim=-1) - 1.0)
        .abs()
        .max()
        .item()
    )
    scales = initialization.log_scales.exp()
    measured_ratio = scales[:, 2] / torch.minimum(scales[:, 0], scales[:, 1])
    thickness_ratio_error = float(
        (measured_ratio - args.thickness_ratio).abs().max().item()
    )
    finite = bool(
        torch.isfinite(initialization.means).all()
        and torch.isfinite(initialization.colors).all()
        and torch.isfinite(initialization.quats).all()
        and torch.isfinite(initialization.log_scales).all()
    )

    tolerance = 1e-4
    status = "PASS" if (
        finite
        and world_max_abs_error <= tolerance
        and orthogonality_error <= tolerance
        and determinant_error <= tolerance
        and normal_axis_error <= tolerance
        and quaternion_norm_error <= tolerance
        and thickness_ratio_error <= tolerance
    ) else "FAIL"
    report = {
        "status": status,
        "sequence": args.sequence,
        "frame_index": args.frame_index,
        "frame_id": int(item["frame_id"].item()),
        "samples": args.samples,
        "world_scale": args.world_scale,
        "world_max_abs_error": world_max_abs_error,
        "world_max_l2_error": world_max_l2_error,
        "quaternion_convention": "gsplat wxyz",
        "quaternion_norm_error": quaternion_norm_error,
        "rotation_orthogonality_error": orthogonality_error,
        "rotation_determinant_error": determinant_error,
        "surface_normal_axis_error": normal_axis_error,
        "configured_thickness_ratio": args.thickness_ratio,
        "thickness_ratio_error": thickness_ratio_error,
        "mini_initialization_statistics": initialization.statistics,
        "reference": str(REPO_ROOT / "method1_rgbd_reprojection"),
        "m3_geometry_reference": "/mnt/cluster/workspaces/venkateda/method3_surface_fusion/surface_geometry.py (read-only)",
        "target_rgb_access": False,
        "target_camera_access": False,
        "tolerance": tolerance,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"M4-B INITIALIZATION: {status}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
