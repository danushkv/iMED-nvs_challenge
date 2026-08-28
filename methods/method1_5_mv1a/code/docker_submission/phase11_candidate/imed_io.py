"""Inference-only loading for the permitted iMED-NVS source inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


FRAME_RE = re.compile(r"frame_(\d{6})$")


@dataclass(frozen=True)
class SourceFrame:
    frame_id: int
    rgb_path: Path
    depth_path: Path


@dataclass(frozen=True)
class Calibration:
    K_target_full: np.ndarray
    K_source_full: np.ndarray
    T_world_target: np.ndarray
    T_world_source: np.ndarray

    @property
    def T_target_source(self) -> np.ndarray:
        """Transform satisfying ``X_target = T_target_source @ X_source``."""

        return np.linalg.inv(self.T_world_target) @ self.T_world_source


@dataclass(frozen=True)
class LoadedSource:
    rgb_at_depth_resolution: np.ndarray
    depth: np.ndarray
    native_rgb_size_wh: tuple[int, int]


def _frame_id(path: Path) -> int:
    match = FRAME_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected source filename: {path.name}")
    return int(match.group(1))


def collect_source_frames(sequence_dir: Path) -> list[SourceFrame]:
    """Match only Endoscope-2 RGB and metric depth by original frame id."""

    source_dir = Path(sequence_dir) / "endoscope2"
    rgb_by_id = {_frame_id(path): path for path in sorted((source_dir / "L").glob("frame_*.png"))}
    depth_by_id = {
        _frame_id(path): path for path in sorted((source_dir / "depthL").glob("frame_*.npy"))
    }

    if not rgb_by_id:
        raise FileNotFoundError(f"No source RGB frames under {source_dir / 'L'}")
    if not depth_by_id:
        raise FileNotFoundError(f"No source metric depth under {source_dir / 'depthL'}")
    if set(rgb_by_id) != set(depth_by_id):
        raise ValueError("Endoscope 2 RGB/depth frame ids do not match")

    return [
        SourceFrame(
            frame_id=frame_id,
            rgb_path=rgb_by_id[frame_id],
            depth_path=depth_by_id[frame_id],
        )
        for frame_id in sorted(rgb_by_id)
    ]


def load_source_frame(frame: SourceFrame) -> LoadedSource:
    depth = np.load(frame.depth_path).astype(np.float32, copy=False)
    if depth.ndim != 2:
        raise ValueError(f"Source depth must be two-dimensional: {frame.depth_path}")

    depth_height, depth_width = depth.shape
    with Image.open(frame.rgb_path) as source_image:
        source_rgb = source_image.convert("RGB")
        native_size = source_rgb.size
        if source_rgb.size != (depth_width, depth_height):
            source_rgb = source_rgb.resize(
                (depth_width, depth_height), Image.Resampling.BILINEAR
            )
        rgb = np.array(source_rgb, dtype=np.float32, copy=True) / 255.0

    return LoadedSource(
        rgb_at_depth_resolution=rgb,
        depth=depth,
        native_rgb_size_wh=native_size,
    )


def _parse_intrinsics(path: Path) -> dict[str, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrices: dict[str, np.ndarray] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("# K"):
            key = line[1:].strip().split()[0]
            if index + 3 >= len(lines):
                raise ValueError(f"Incomplete {key} block in {path}")
            rows = [[float(value) for value in lines[index + offset].split()] for offset in (1, 2, 3)]
            if any(len(row) != 3 for row in rows):
                raise ValueError(f"{key} is not 3x3 in {path}")
            matrices[key] = np.asarray(rows, dtype=np.float64)
            index += 4
        else:
            index += 1
    for key in ("K1_L", "K2_L"):
        if key not in matrices:
            raise ValueError(f"Missing {key} in {path}")
    return matrices


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Invalid quaternion: {quaternion}")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _parse_camera_to_world(path: Path) -> dict[int, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"Expected two pose rows in {path}, found {len(lines)}")

    transforms: dict[int, np.ndarray] = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"Pose row must contain id tx ty tz qx qy qz qw: {line}")
        camera_id = int(fields[0])
        T_world_camera = np.eye(4, dtype=np.float64)
        T_world_camera[:3, :3] = _quaternion_xyzw_to_matrix(
            np.asarray(fields[4:8], dtype=np.float64)
        )
        T_world_camera[:3, 3] = np.asarray(fields[1:4], dtype=np.float64)
        transforms[camera_id] = T_world_camera
    if set(transforms) != {0, 1}:
        raise ValueError(f"Expected camera ids 0 and 1 in {path}, got {sorted(transforms)}")
    return transforms


def load_calibration(sequence_dir: Path) -> Calibration:
    """Load id 0 as source camera and id 1 as target, both camera-to-world."""

    sequence_dir = Path(sequence_dir)
    intrinsics = _parse_intrinsics(sequence_dir / "K.txt")
    camera_to_world = _parse_camera_to_world(sequence_dir / "pose.txt")
    return Calibration(
        K_target_full=intrinsics["K1_L"],
        K_source_full=intrinsics["K2_L"],
        T_world_target=camera_to_world[1],
        T_world_source=camera_to_world[0],
    )


def scale_intrinsics(
    K_native: np.ndarray,
    native_size_wh: tuple[int, int],
    resized_size_wh: tuple[int, int],
) -> np.ndarray:
    """Scale pinhole intrinsics from native pixels to a resized image grid."""

    native_width, native_height = native_size_wh
    resized_width, resized_height = resized_size_wh
    if min(native_width, native_height, resized_width, resized_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = np.asarray(
        [
            [resized_width / native_width, 0.0, 0.0],
            [0.0, resized_height / native_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return scale @ np.asarray(K_native, dtype=np.float64)
