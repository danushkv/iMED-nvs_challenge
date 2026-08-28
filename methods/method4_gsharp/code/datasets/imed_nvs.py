"""Read-only iMED-NVS adapter for Method 4A.

Only Endoscope2/L source RGB-D-mask observations are exposed as training
items. Target camera calibration is exposed separately for legal rendering;
this module never discovers or opens Endoscope1 imagery, depth, or masks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


_FRAME_RE = re.compile(r"frame_(\d+)$")


@dataclass(frozen=True)
class SourceFrame:
    frame_id: int
    rgb_path: Path
    depth_path: Path
    mask_path: Path


@dataclass(frozen=True)
class CameraCalibration:
    K_source_full: np.ndarray
    K_target_full: np.ndarray
    camtoworld_source: np.ndarray
    camtoworld_target: np.ndarray


def _frame_id(path: Path) -> int:
    match = _FRAME_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected iMED frame filename: {path.name}")
    return int(match.group(1))


def _paths_by_id(directory: Path, pattern: str) -> dict[int, Path]:
    return {_frame_id(path): path for path in sorted(directory.glob(pattern))}


def _parse_intrinsics(path: Path) -> dict[str, np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    matrices: dict[str, np.ndarray] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("# K"):
            index += 1
            continue
        key = line[1:].strip().split()[0]
        if index + 3 >= len(lines):
            raise ValueError(f"Incomplete {key} block in {path}")
        rows = [
            [float(value) for value in lines[index + offset].split()]
            for offset in (1, 2, 3)
        ]
        if any(len(row) != 3 for row in rows):
            raise ValueError(f"{key} is not a 3x3 matrix in {path}")
        matrices[key] = np.asarray(rows, dtype=np.float64)
        index += 4
    for required in ("K1_L", "K2_L"):
        if required not in matrices:
            raise ValueError(f"Missing {required} in {path}")
    return matrices


def _xyzw_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid pose quaternion: {q}")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _parse_camtoworlds(path: Path) -> dict[int, np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"Expected two pose rows in {path}, found {len(lines)}")
    transforms: dict[int, np.ndarray] = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(
                "Pose row must be: camera_id tx ty tz qx qy qz qw; "
                f"got {line!r}"
            )
        camera_id = int(fields[0])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _xyzw_to_rotation([float(v) for v in fields[4:8]])
        transform[:3, 3] = np.asarray([float(v) for v in fields[1:4]])
        transforms[camera_id] = transform
    if set(transforms) != {0, 1}:
        raise ValueError(f"Expected camera ids 0 and 1, got {sorted(transforms)}")
    return transforms


def scale_intrinsics(
    K: np.ndarray,
    native_size_wh: tuple[int, int],
    resized_size_wh: tuple[int, int],
) -> np.ndarray:
    native_w, native_h = native_size_wh
    resized_w, resized_h = resized_size_wh
    if min(native_w, native_h, resized_w, resized_h) <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = np.asarray(
        [
            [resized_w / native_w, 0.0, 0.0],
            [0.0, resized_h / native_h, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return scale @ np.asarray(K, dtype=np.float64)


def backproject_pixels_to_world(
    depth: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    pixels_uv: np.ndarray,
) -> np.ndarray:
    """Backproject selected `(u,v)` pixels using optical-axis Z-depth."""

    pixels = np.asarray(pixels_uv, dtype=np.int64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels_uv must have shape (N,2), got {pixels.shape}")
    u = pixels[:, 0]
    v = pixels[:, 1]
    if (
        (u < 0).any()
        or (u >= depth.shape[1]).any()
        or (v < 0).any()
        or (v >= depth.shape[0]).any()
    ):
        raise ValueError("Selected pixel lies outside the depth image")
    z = np.asarray(depth, dtype=np.float64)[v, u]
    K64 = np.asarray(K, dtype=np.float64)
    x = (u.astype(np.float64) - K64[0, 2]) * z / K64[0, 0]
    y = (v.astype(np.float64) - K64[1, 2]) * z / K64[1, 1]
    camera = np.stack((x, y, z), axis=-1)
    c2w = np.asarray(camtoworld, dtype=np.float64)
    return camera @ c2w[:3, :3].T + c2w[:3, 3]


class IMEDNVSParser:
    """Parse one iMED sequence without accessing Endoscope1 observations.

    Args:
        sequence_dir: Sequence containing `K.txt`, `pose.txt`, and
            `endoscope2`.
        world_scale: One global multiplier applied to source depth and both
            camera translations. The vanilla M4-A setting is 1.0, preserving
            iMED millimetres. It is never applied independently per frame.
        target_size_wh: Legal target render size. If omitted, use the source
            RGB native size; both supplied camera streams use that grid.
    """

    def __init__(
        self,
        sequence_dir: Path | str,
        world_scale: float = 1.0,
        target_size_wh: tuple[int, int] | None = None,
    ) -> None:
        self.sequence_dir = Path(sequence_dir)
        if not self.sequence_dir.is_dir():
            raise FileNotFoundError(f"Sequence directory not found: {self.sequence_dir}")
        if not np.isfinite(world_scale) or world_scale <= 0.0:
            raise ValueError(f"world_scale must be positive, got {world_scale}")
        self.world_scale = float(world_scale)

        source_root = self.sequence_dir / "endoscope2"
        rgb = _paths_by_id(source_root / "L", "frame_*.png")
        depth = _paths_by_id(source_root / "depthL", "frame_*.npy")
        mask = _paths_by_id(source_root / "toolL", "frame_*.png")
        if not rgb or not depth:
            raise FileNotFoundError(f"Missing Endoscope2/L RGB-D in {source_root}")
        if set(rgb) != set(depth):
            raise ValueError("Endoscope2 RGB and depth frame IDs do not match")
        if set(rgb) != set(mask):
            raise ValueError(
                "Endoscope2 source tool masks are required for vanilla M4-A and "
                "must match RGB/depth frame IDs"
            )
        self.frames = [
            SourceFrame(frame_id=i, rgb_path=rgb[i], depth_path=depth[i], mask_path=mask[i])
            for i in sorted(rgb)
        ]

        with Image.open(self.frames[0].rgb_path) as image:
            self.native_size_wh = tuple(int(v) for v in image.size)
        first_depth = np.load(self.frames[0].depth_path, mmap_mode="r")
        if first_depth.ndim != 2:
            raise ValueError(f"Depth must be 2D: {self.frames[0].depth_path}")
        self.height, self.width = (int(first_depth.shape[0]), int(first_depth.shape[1]))
        self.source_size_wh = (self.width, self.height)
        self.target_size_wh = target_size_wh or self.native_size_wh
        self.target_width, self.target_height = self.target_size_wh

        intrinsics = _parse_intrinsics(self.sequence_dir / "K.txt")
        poses = _parse_camtoworlds(self.sequence_dir / "pose.txt")
        source_c2w = poses[0].copy()
        target_c2w = poses[1].copy()
        source_c2w[:3, 3] *= self.world_scale
        target_c2w[:3, 3] *= self.world_scale
        self.calibration = CameraCalibration(
            K_source_full=intrinsics["K2_L"].copy(),
            K_target_full=intrinsics["K1_L"].copy(),
            camtoworld_source=source_c2w,
            camtoworld_target=target_c2w,
        )
        self.K = scale_intrinsics(
            self.calibration.K_source_full,
            self.native_size_wh,
            self.source_size_wh,
        ).astype(np.float32)
        self.K_target = scale_intrinsics(
            self.calibration.K_target_full,
            self.native_size_wh,
            self.target_size_wh,
        ).astype(np.float32)
        self.camtoworlds = np.repeat(
            self.calibration.camtoworld_source.astype(np.float32)[None],
            len(self.frames),
            axis=0,
        )
        self.target_camtoworlds = np.repeat(
            self.calibration.camtoworld_target.astype(np.float32)[None],
            len(self.frames),
            axis=0,
        )
        denominator = max(len(self.frames) - 1, 1)
        self.times = np.arange(len(self.frames), dtype=np.float32) / denominator
        self.train_idxs = list(range(len(self.frames)))
        self.video_idxs = list(range(len(self.frames)))

    def target_camera(self, index: int) -> dict[str, Any]:
        """Return target calibration/time only; never target observations."""

        if not 0 <= index < len(self.frames):
            raise IndexError(index)
        return {
            "camtoworld": torch.from_numpy(self.target_camtoworlds[index].copy()),
            "K": torch.from_numpy(self.K_target.copy()),
            "time": torch.tensor(float(self.times[index]), dtype=torch.float32),
            "frame_id": self.frames[index].frame_id,
            "height": self.target_height,
            "width": self.target_width,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence_dir.name,
            "frames": len(self.frames),
            "source_size_wh": list(self.source_size_wh),
            "target_size_wh": list(self.target_size_wh),
            "native_rgb_size_wh": list(self.native_size_wh),
            "depth_units": "iMED millimetres" if self.world_scale == 1.0 else "scaled",
            "world_scale": self.world_scale,
            "source_camera": "Endoscope2/L, pose camera id 0",
            "target_camera": "Endoscope1/L calibration only, pose camera id 1",
            "target_rgb_access": False,
            "mask_convention": "1=tissue/include, 0=tool/exclude",
            "pose_convention": "camera-to-world",
            "time_convention": "ordered synchronized index/(N-1)",
        }


class IMEDNVSDataset(torch.utils.data.Dataset):
    """Endoscope2 training observations in upstream G-SHARP item format."""

    def __init__(self, parser: IMEDNVSParser) -> None:
        self.parser = parser

    def __len__(self) -> int:
        return len(self.parser.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame = self.parser.frames[index]
        depth = np.load(frame.depth_path).astype(np.float32, copy=False)
        if depth.shape != (self.parser.height, self.parser.width):
            raise ValueError(f"Depth resolution changed at {frame.depth_path}")
        depth = np.asarray(depth * self.parser.world_scale, dtype=np.float32)

        with Image.open(frame.rgb_path) as image:
            image = image.convert("RGB")
            if image.size != self.parser.native_size_wh:
                raise ValueError(f"RGB resolution changed at {frame.rgb_path}")
            if image.size != self.parser.source_size_wh:
                image = image.resize(self.parser.source_size_wh, Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.float32) / 255.0

        with Image.open(frame.mask_path) as image:
            raw = np.asarray(image.convert("L"), dtype=np.uint8)
        if not np.isin(np.unique(raw), (0, 255)).all():
            raise ValueError(f"Source tool mask is not binary: {frame.mask_path}")
        if (raw.shape[1], raw.shape[0]) != self.parser.source_size_wh:
            raw = np.asarray(
                Image.fromarray(raw).resize(
                    self.parser.source_size_wh, Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            )
        tissue_mask = 1.0 - raw.astype(np.float32) / 255.0

        return {
            "image": torch.from_numpy(np.ascontiguousarray(rgb)),
            "depth": torch.from_numpy(np.ascontiguousarray(depth)),
            "mask": torch.from_numpy(np.ascontiguousarray(tissue_mask)),
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[index].copy()),
            "K": torch.from_numpy(self.parser.K.copy()),
            "time": torch.tensor(float(self.parser.times[index]), dtype=torch.float32),
            "frame_id": torch.tensor(frame.frame_id, dtype=torch.int64),
        }
