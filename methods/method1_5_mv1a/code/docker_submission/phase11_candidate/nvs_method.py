"""Phase-11 direct Endoscope-2 RGB-D reprojection submission inference."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hole_filling import nearest_valid_fill
from imed_io import collect_source_frames, load_calibration, load_source_frame, scale_intrinsics
from soft_splatting import render_soft_splat


RENDERER = "soft_depth"
VISIBILITY_TOLERANCE_MM = 1.0
VISIBILITY_RELATIVE = 0.01
DEPTH_SOFTNESS = 8.0
SOURCE_TOOL_MASKING = "disabled"
HOLE_HANDLING = "nearest_valid"
FILL_RADIUS = 3
DEPTH_FILTERING = "none"


def _device() -> torch.device:
    requested = os.environ.get("IMED_NVS_DEVICE", "cuda").strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available inside the container")
    return torch.device(requested)


def _to_uint8_full_resolution(
    rgb_at_depth_resolution: torch.Tensor,
    native_size_wh: tuple[int, int],
) -> np.ndarray:
    """Match the official baseline export: bilinear RGB resize to native size."""

    native_width, native_height = native_size_wh
    rgb_nchw = rgb_at_depth_resolution.permute(2, 0, 1).unsqueeze(0)
    if tuple(rgb_at_depth_resolution.shape[:2]) != (native_height, native_width):
        rgb_nchw = F.interpolate(
            rgb_nchw,
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )
    rgb_hwc = rgb_nchw[0].permute(1, 2, 0)
    return (
        rgb_hwc.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


@torch.inference_mode()
def render_target_views(sequence_dir: str | Path, output_dir: str | Path) -> None:
    """Render one target-view RGB PNG for each synchronized source RGB-D frame."""

    sequence_dir = Path(sequence_dir)
    output_dir = Path(output_dir)
    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    frames = collect_source_frames(sequence_dir)
    device = _device()
    calibration = load_calibration(sequence_dir)

    first = load_source_frame(frames[0])
    depth_height, depth_width = first.depth.shape
    internal_size_wh = (depth_width, depth_height)
    native_size_wh = first.native_rgb_size_wh
    K_source = torch.from_numpy(
        scale_intrinsics(calibration.K_source_full, native_size_wh, internal_size_wh).astype(np.float32)
    ).to(device)
    K_target = torch.from_numpy(
        scale_intrinsics(calibration.K_target_full, native_size_wh, internal_size_wh).astype(np.float32)
    ).to(device)
    T_target_source = torch.from_numpy(
        calibration.T_target_source.astype(np.float32)
    ).to(device)

    print(
        f"[METHOD1] sequence={sequence_dir.name} frames={len(frames)} device={device} "
        f"renderer={RENDERER} visibility_tolerance_mm={VISIBILITY_TOLERANCE_MM} "
        f"visibility_relative={VISIBILITY_RELATIVE} depth_softness={DEPTH_SOFTNESS} "
        f"source_tool_masking={SOURCE_TOOL_MASKING} hole_handling={HOLE_HANDLING} "
        f"fill_radius={FILL_RADIUS} depth_filtering={DEPTH_FILTERING} "
        f"internal_size={depth_width}x{depth_height} "
        f"output_size={native_size_wh[0]}x{native_size_wh[1]}",
        flush=True,
    )

    sequence_started = time.perf_counter()
    raw_coverage_sum = 0.0
    filled_coverage_sum = 0.0
    for index, frame in enumerate(frames):
        loaded = first if index == 0 else load_source_frame(frame)
        if loaded.depth.shape != (depth_height, depth_width):
            raise ValueError(f"Depth resolution changed at {frame.depth_path}")
        if loaded.native_rgb_size_wh != native_size_wh:
            raise ValueError(f"Source RGB resolution changed at {frame.rgb_path}")

        source_rgb = torch.from_numpy(
            np.ascontiguousarray(loaded.rgb_at_depth_resolution)
        ).to(device=device, dtype=torch.float32)
        source_depth = torch.from_numpy(np.ascontiguousarray(loaded.depth)).to(
            device=device, dtype=torch.float32
        )

        result = render_soft_splat(
            source_rgb=source_rgb,
            source_depth=source_depth,
            K_source=K_source,
            K_target=K_target,
            T_target_source=T_target_source,
            target_size=(depth_height, depth_width),
            source_valid_mask=None,
            mode=RENDERER,
            visibility_tolerance_mm=VISIBILITY_TOLERANCE_MM,
            visibility_relative=VISIBILITY_RELATIVE,
            depth_softness=DEPTH_SOFTNESS,
        )
        filled = nearest_valid_fill(
            result.rgb,
            result.depth,
            result.valid_mask,
            result.confidence,
            radius=FILL_RADIUS,
        )

        prediction = _to_uint8_full_resolution(filled.rgb, native_size_wh)
        output_path = render_dir / f"{index:05d}.png"
        Image.fromarray(prediction, mode="RGB").save(output_path)
        raw_coverage = result.stats["target_coverage_percent"]
        filled_coverage = float(filled.filled_valid_mask.float().mean().item() * 100.0)
        raw_coverage_sum += raw_coverage
        filled_coverage_sum += filled_coverage

        if index == 0 or index + 1 == len(frames) or (index + 1) % 25 == 0:
            print(
                f"[METHOD1] {sequence_dir.name} {index + 1}/{len(frames)} "
                f"source={frame.rgb_path.name} output={output_path.name} "
                f"raw_coverage={raw_coverage:.2f}% filled_coverage={filled_coverage:.2f}%",
                flush=True,
            )

    elapsed = time.perf_counter() - sequence_started
    print(
        f"[METHOD1] completed sequence={sequence_dir.name} frames={len(frames)} "
        f"seconds={elapsed:.3f} seconds_per_frame={elapsed / len(frames):.4f} "
        f"mean_raw_coverage={raw_coverage_sum / len(frames):.2f}% "
        f"mean_filled_coverage={filled_coverage_sum / len(frames):.2f}%",
        flush=True,
    )
