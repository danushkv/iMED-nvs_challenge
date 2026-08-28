#!/usr/bin/env python3
"""Read-only inspection and loading helpers for the iMED NVS layout.

This module deliberately has no dependency on the Endo-4DGS package. Inference
loaders below only open Endoscope 2 RGB/depth and, when requested, its tool mask.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


FRAME_RE = re.compile(r"frame_(\d{6})$")


@dataclass(frozen=True)
class SequenceCalibration:
    K1_L_full: np.ndarray
    K2_L_full: np.ndarray
    T_world_cam1: np.ndarray
    T_world_cam2: np.ndarray

    @property
    def T_cam1_cam2(self) -> np.ndarray:
        """Transform satisfying X_cam1 = T_cam1_cam2 @ X_cam2."""

        return np.linalg.inv(self.T_world_cam1) @ self.T_world_cam2


@dataclass(frozen=True)
class SourceFrame:
    frame_id: int
    rgb_path: Path
    depth_path: Path
    tool_mask_path: Optional[Path]


def _quat_xyzw_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Match scipy Rotation.from_quat: quaternion order is x,y,z,w."""

    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"invalid quaternion: {q}")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_intrinsics(k_path: Path) -> Dict[str, np.ndarray]:
    lines = [line.strip() for line in Path(k_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    matrices: Dict[str, np.ndarray] = {}
    index = 0
    while index < len(lines):
        if lines[index].startswith("# K"):
            key = lines[index][1:].strip().split()[0]
            if index + 3 >= len(lines):
                raise ValueError(f"incomplete {key} block in {k_path}")
            rows = [[float(v) for v in lines[index + offset].split()] for offset in (1, 2, 3)]
            if any(len(row) != 3 for row in rows):
                raise ValueError(f"{key} is not 3x3 in {k_path}")
            matrices[key] = np.asarray(rows, dtype=np.float64)
            index += 4
        else:
            index += 1
    for required in ("K1_L", "K2_L"):
        if required not in matrices:
            raise ValueError(f"missing {required} in {k_path}")
    return matrices


def parse_camera_to_world_poses(pose_path: Path) -> Dict[int, np.ndarray]:
    """Parse ``id tx ty tz qx qy qz qw`` rows as camera-to-world transforms."""

    lines = [line.strip() for line in Path(pose_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"expected two pose rows in {pose_path}, found {len(lines)}")
    transforms: Dict[int, np.ndarray] = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"pose row must contain 8 values: {line}")
        camera_id = int(fields[0])
        T_world_camera = np.eye(4, dtype=np.float64)
        T_world_camera[:3, :3] = _quat_xyzw_to_matrix(np.asarray(fields[4:8], dtype=np.float64))
        T_world_camera[:3, 3] = np.asarray(fields[1:4], dtype=np.float64)
        transforms[camera_id] = T_world_camera
    if set(transforms) != {0, 1}:
        raise ValueError(f"expected camera ids 0 and 1 in {pose_path}, got {sorted(transforms)}")
    return transforms


def load_calibration(sequence_dir: Path) -> SequenceCalibration:
    sequence_dir = Path(sequence_dir)
    intrinsics = parse_intrinsics(sequence_dir / "K.txt")
    c2w = parse_camera_to_world_poses(sequence_dir / "pose.txt")
    return SequenceCalibration(
        K1_L_full=intrinsics["K1_L"],
        K2_L_full=intrinsics["K2_L"],
        T_world_cam1=c2w[1],
        T_world_cam2=c2w[0],
    )


def scale_intrinsics(K_full: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Scale a full-resolution K for resized pixels without changing K[2,2]."""

    K = np.asarray(K_full, dtype=np.float64).copy()
    K[0, :] /= float(scale_x)
    K[1, :] /= float(scale_y)
    return K


def _frame_id(path: Path) -> int:
    match = FRAME_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"unexpected frame filename: {path.name}")
    return int(match.group(1))


def collect_source_frames(sequence_dir: Path) -> List[SourceFrame]:
    """Collect synchronized Endoscope 2 inputs without touching target RGB."""

    source = Path(sequence_dir) / "endoscope2"
    rgb_by_id = {_frame_id(p): p for p in sorted((source / "L").glob("frame_*.png"))}
    depth_by_id = {_frame_id(p): p for p in sorted((source / "depthL").glob("frame_*.npy"))}
    mask_by_id = {_frame_id(p): p for p in sorted((source / "toolL").glob("frame_*.png"))}
    if not rgb_by_id:
        raise FileNotFoundError(f"no source RGB frames under {source / 'L'}")
    if set(rgb_by_id) != set(depth_by_id):
        raise ValueError("Endoscope 2 RGB/depth frame ids do not match")
    if mask_by_id and set(rgb_by_id) != set(mask_by_id):
        raise ValueError("Endoscope 2 RGB/tool-mask frame ids do not match")
    return [
        SourceFrame(frame_id, rgb_by_id[frame_id], depth_by_id[frame_id], mask_by_id.get(frame_id))
        for frame_id in sorted(rgb_by_id)
    ]


def load_source_frame(
    frame: SourceFrame,
    load_tool_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Tuple[float, float]]:
    """Load one source frame at depth resolution.

    Returns RGB float32 [H,W,3] in [0,1], depth float32 [H,W], an optional
    boolean tissue-valid mask (True means not tool), and (scale_x, scale_y).
    """

    depth = np.load(frame.depth_path).astype(np.float32, copy=False)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D: {frame.depth_path}")
    height, width = depth.shape
    rgb_image = Image.open(frame.rgb_path).convert("RGB")
    scale_x = rgb_image.width / width
    scale_y = rgb_image.height / height
    if rgb_image.size != (width, height):
        rgb_image = rgb_image.resize((width, height), Image.Resampling.BILINEAR)
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0

    tissue_mask = None
    if load_tool_mask:
        if frame.tool_mask_path is None:
            raise FileNotFoundError(
                f"--mask_source_tools was requested, but no source tool mask exists for frame {frame.frame_id:06d}"
            )
        raw_image = Image.open(frame.tool_mask_path).convert("L")
        if raw_image.size != (int(round(width * scale_x)), int(round(height * scale_y))):
            raise ValueError(f"tool mask and source RGB sizes differ: {frame.tool_mask_path}")
        if raw_image.size != (width, height):
            raw_image = raw_image.resize((width, height), Image.Resampling.NEAREST)
        raw = np.asarray(raw_image)
        if not np.all(np.isin(np.unique(raw), [0, 255])):
            raise ValueError(f"tool mask is not binary 0/255: {frame.tool_mask_path}")
        # Verified visually and matches Endo-4DGS: raw 255 denotes tool/excluded.
        tissue_mask = raw == 0
    return rgb, depth, tissue_mask, (scale_x, scale_y)


def _inspect_depths(paths: Iterable[Path]) -> dict:
    stats = {
        "files": 0,
        "pixels": 0,
        "dtypes": set(),
        "shapes": set(),
        "min": float("inf"),
        "max": float("-inf"),
        "zero": 0,
        "negative": 0,
        "nonfinite": 0,
    }
    for path in paths:
        depth = np.load(path, mmap_mode="r")
        stats["files"] += 1
        stats["pixels"] += depth.size
        stats["dtypes"].add(str(depth.dtype))
        stats["shapes"].add(tuple(depth.shape))
        stats["min"] = min(stats["min"], float(np.nanmin(depth)))
        stats["max"] = max(stats["max"], float(np.nanmax(depth)))
        stats["zero"] += int((depth == 0).sum())
        stats["negative"] += int((depth < 0).sum())
        stats["nonfinite"] += int((~np.isfinite(depth)).sum())
    stats["dtypes"] = sorted(stats["dtypes"])
    stats["shapes"] = sorted(stats["shapes"])
    return stats


def inspect_sequence(sequence_dir: Path, scan_all_depths: bool = False) -> dict:
    sequence_dir = Path(sequence_dir)
    calibration = load_calibration(sequence_dir)
    frames = collect_source_frames(sequence_dir)
    rgb, depth, _, scales = load_source_frame(frames[0], load_tool_mask=False)
    target_rgb_paths = sorted((sequence_dir / "endoscope1" / "L").glob("frame_*.png"))
    target_ids = [_frame_id(p) for p in target_rgb_paths]
    source_ids = [f.frame_id for f in frames]
    paths = [f.depth_path for f in frames] if scan_all_depths else [frames[0].depth_path, frames[len(frames) // 2].depth_path, frames[-1].depth_path]
    return {
        "sequence": sequence_dir.name,
        "source_frames": len(frames),
        "source_target_frame_ids_equal": source_ids == target_ids,
        "first_frame_id": source_ids[0],
        "last_frame_id": source_ids[-1],
        "source_rgb_at_depth_shape": list(rgb.shape),
        "source_depth_shape": list(depth.shape),
        "rgb_to_depth_scale_xy": list(scales),
        "source_tool_masks_present": all(frame.tool_mask_path is not None for frame in frames),
        "depth_scan": _inspect_depths(paths),
        "K2_L_full": calibration.K2_L_full.tolist(),
        "K1_L_full": calibration.K1_L_full.tolist(),
        "T_world_cam2": calibration.T_world_cam2.tolist(),
        "T_world_cam1": calibration.T_world_cam1.tolist(),
        "T_cam1_cam2": calibration.T_cam1_cam2.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only iMED NVS dataset inspection")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--sequence", help="Inspect one sequence; default: all")
    parser.add_argument("--scan_all_depths", action="store_true", help="Read every source depth pixel (slow on network storage)")
    args = parser.parse_args()
    sequences = [args.data_root / args.sequence] if args.sequence else sorted(p for p in args.data_root.glob("session_*") if p.is_dir())
    reports = [inspect_sequence(path, scan_all_depths=args.scan_all_depths) for path in sequences]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
