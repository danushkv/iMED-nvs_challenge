#!/usr/bin/env python3
"""Compare the M4 adapter against the frozen validated RGB-D geometry.

This is a read-only source-camera test. It never discovers or opens
Endoscope1 RGB/depth/masks and performs no training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method4_gsharp.datasets.imed_nvs import (  # noqa: E402
    IMEDNVSDataset,
    IMEDNVSParser,
    backproject_pixels_to_world,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify M4 iMED pose/intrinsics/backprojection conventions"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--world-scale", type=float, default=1.0)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=REPO_ROOT / "method1_rgbd_reprojection",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    sequence_dir = args.data_root / args.sequence
    parser = IMEDNVSParser(sequence_dir, world_scale=args.world_scale)
    dataset = IMEDNVSDataset(parser)
    if not 0 <= args.frame_index < len(dataset):
        raise IndexError(f"frame index {args.frame_index} outside 0..{len(dataset)-1}")

    camera_ref = _load_module(
        "m4_validated_camera_reference", args.reference_root / "camera.py"
    )
    imed_ref = _load_module(
        "m4_validated_imed_reference",
        args.reference_root / "docker_submission" / "phase11_candidate" / "imed_io.py",
    )

    item = dataset[args.frame_index]
    depth = item["depth"].numpy()
    mask = item["mask"].numpy() != 0
    K = item["K"].numpy()
    camtoworld = item["camtoworld"].numpy()

    valid_vu = np.argwhere(mask & np.isfinite(depth) & (depth > 0))
    if valid_vu.shape[0] < args.samples:
        raise RuntimeError(
            f"Only {valid_vu.shape[0]} valid tissue-depth pixels; "
            f"cannot select {args.samples}"
        )
    selected = np.linspace(0, valid_vu.shape[0] - 1, args.samples, dtype=np.int64)
    pixels_vu = valid_vu[selected]
    pixels_uv = pixels_vu[:, ::-1].copy()

    world_adapter = backproject_pixels_to_world(depth, K, camtoworld, pixels_uv)
    camera_grid_ref = camera_ref.backproject_depth(depth.astype(np.float64), K.astype(np.float64))
    camera_points_ref = camera_grid_ref[pixels_vu[:, 0], pixels_vu[:, 1]]
    world_reference = camera_ref.transform_points(
        camera_points_ref, camtoworld.astype(np.float64)
    )

    world_delta = world_adapter - world_reference
    world_abs = np.abs(world_delta)
    world_l2 = np.linalg.norm(world_delta, axis=-1)

    world_to_camera = camera_ref.invert_transform(camtoworld.astype(np.float64))
    roundtrip_camera = camera_ref.transform_points(world_adapter, world_to_camera)
    roundtrip_uv = camera_ref.project_points(roundtrip_camera, K.astype(np.float64))
    pixel_error = np.linalg.norm(roundtrip_uv - pixels_uv, axis=-1)

    frozen_calibration = imed_ref.load_calibration(sequence_dir)
    expected_source_c2w = frozen_calibration.T_world_source.copy()
    expected_target_c2w = frozen_calibration.T_world_target.copy()
    expected_source_c2w[:3, 3] *= args.world_scale
    expected_target_c2w[:3, 3] *= args.world_scale
    expected_K = imed_ref.scale_intrinsics(
        frozen_calibration.K_source_full,
        parser.native_size_wh,
        parser.source_size_wh,
    )

    calibration_error = {
        "K_source_max_abs": float(np.max(np.abs(K - expected_K))),
        "source_camtoworld_max_abs": float(
            np.max(np.abs(camtoworld - expected_source_c2w))
        ),
        "target_camtoworld_max_abs": float(
            np.max(np.abs(parser.target_camtoworlds[0] - expected_target_c2w))
        ),
    }
    report = {
        "status": "PASS",
        "sequence": args.sequence,
        "frame_index": args.frame_index,
        "frame_id": int(item["frame_id"].item()),
        "samples": args.samples,
        "world_scale": args.world_scale,
        "depth_units": parser.summary()["depth_units"],
        "pose_convention": "camera-to-world",
        "reference": str(args.reference_root.resolve()),
        "world_max_abs_error": float(world_abs.max()),
        "world_mean_abs_error": float(world_abs.mean()),
        "world_max_l2_error": float(world_l2.max()),
        "roundtrip_max_pixel_error": float(pixel_error.max()),
        "calibration_error": calibration_error,
        "target_rgb_access": False,
        "tolerance": args.tolerance,
    }
    observed_max = max(
        report["world_max_abs_error"],
        report["roundtrip_max_pixel_error"],
        *calibration_error.values(),
    )
    if observed_max > args.tolerance:
        report["status"] = "FAIL"

    output = json.dumps(report, indent=2)
    print(output)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n")
    if report["status"] != "PASS":
        raise SystemExit(
            "GEOMETRY CONVENTION: FAIL — do not start M4-A training"
        )
    print("GEOMETRY CONVENTION: PASS")


if __name__ == "__main__":
    main()
