#!/usr/bin/env python3
"""Production iMED-NVS adapter for fixed Method-3 candidates.

Inference reads only Endoscope-2 left RGB, Endoscope-2 left metric depth,
K.txt, and pose.txt. It writes only required RGB renders beneath /output.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hole_filling import nearest_valid_fill
from imed_io import collect_source_frames, load_calibration, load_source_frame, scale_intrinsics
from surface_splat import render_surface_splat


VISIBILITY_TOLERANCE_MM = 1.0
VISIBILITY_RELATIVE = 0.01
DEPTH_SOFTNESS = 8.0
FILL_RADIUS = 3
FOOTPRINT_SCALE = 1.0
RADIUS_MIN = 0.5
RADIUS_MAX = 2.0
SURFACE_SUPPORT = 5
MAHALANOBIS_CUTOFF = 9.0
DEPTH_DISCONTINUITY_MM = 2.0
DEPTH_DISCONTINUITY_RELATIVE = 0.02
CHUNK_SIZE = 65536
GEOMETRY_CONFIDENCE_THRESHOLD = 0.25
GEOMETRY_CONFIDENCE_POWER = 1.0


def _renderer() -> str:
    renderer = os.environ.get("METHOD3_RENDERER", "").strip()
    if renderer not in {"surface", "surface_confidence"}:
        raise ValueError(
            "METHOD3_RENDERER baked into the image must be surface or surface_confidence"
        )
    return renderer


def _device() -> torch.device:
    requested = os.environ.get("IMED_NVS_DEVICE", "cuda").strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available inside the container")
    return torch.device(requested)


def _is_sequence_dir(path: Path) -> bool:
    """Recognize sequences without probing Endoscope-1 or right-camera data."""

    return (
        path.is_dir()
        and (path / "K.txt").is_file()
        and (path / "pose.txt").is_file()
        and (path / "endoscope2" / "L").is_dir()
        and (path / "endoscope2" / "depthL").is_dir()
    )


def _discover_sequences(input_root: Path) -> list[Path]:
    if _is_sequence_dir(input_root):
        return [input_root]
    return sorted(path for path in input_root.rglob("*") if _is_sequence_dir(path))


def _to_native_rgb(rgb_internal: torch.Tensor, native_size_wh: tuple[int, int]) -> np.ndarray:
    native_width, native_height = native_size_wh
    rgb_nchw = rgb_internal.permute(2, 0, 1).unsqueeze(0)
    if tuple(rgb_internal.shape[:2]) != (native_height, native_width):
        rgb_nchw = F.interpolate(
            rgb_nchw,
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )
    return (
        rgb_nchw[0]
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


@torch.inference_mode()
def _render_sequence(sequence: Path, output_dir: Path, renderer: str, device: torch.device) -> None:
    frames = collect_source_frames(sequence)
    calibration = load_calibration(sequence)
    first = load_source_frame(frames[0])
    depth_height, depth_width = first.depth.shape
    native_size_wh = first.native_rgb_size_wh
    internal_size_wh = (depth_width, depth_height)

    render_dir = output_dir / "renders"
    if render_dir.exists() and any(render_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty render directory: {render_dir}")
    render_dir.mkdir(parents=True, exist_ok=True)

    K_source = torch.from_numpy(
        scale_intrinsics(
            calibration.K_source_full,
            native_size_wh,
            internal_size_wh,
        ).astype(np.float32)
    ).to(device)
    K_target = torch.from_numpy(
        scale_intrinsics(
            calibration.K_target_full,
            native_size_wh,
            internal_size_wh,
        ).astype(np.float32)
    ).to(device)
    T_target_source = torch.from_numpy(
        calibration.T_target_source.astype(np.float32)
    ).to(device)

    confidence_aware = renderer == "surface_confidence"
    print(
        f"[METHOD3] sequence={sequence.name} renderer={renderer} frames={len(frames)} "
        f"device={device} internal_size={depth_width}x{depth_height} "
        f"output_size={native_size_wh[0]}x{native_size_wh[1]} "
        "source_tool_masking=disabled depth_filtering=none "
        f"hole_fill=nearest_valid fill_radius={FILL_RADIUS} "
        "stereo=disabled temporal_fusion=disabled diagnostics=disabled "
        f"geometry_confidence={'enabled' if confidence_aware else 'disabled'} "
        f"output={render_dir}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
            device=device,
            dtype=torch.float32,
        )

        frame_started = time.perf_counter()
        surface_output = render_surface_splat(
            source_rgb=source_rgb,
            source_depth=source_depth,
            K_source=K_source,
            K_target=K_target,
            T_target_source=T_target_source,
            target_size=(depth_height, depth_width),
            source_valid_mask=None,
            visibility_tolerance_mm=VISIBILITY_TOLERANCE_MM,
            visibility_relative=VISIBILITY_RELATIVE,
            depth_softness=DEPTH_SOFTNESS,
            depth_discontinuity_mm=DEPTH_DISCONTINUITY_MM,
            depth_discontinuity_relative=DEPTH_DISCONTINUITY_RELATIVE,
            footprint_scale=FOOTPRINT_SCALE,
            radius_min=RADIUS_MIN,
            radius_max=RADIUS_MAX,
            support_size=SURFACE_SUPPORT,
            mahalanobis_cutoff=MAHALANOBIS_CUTOFF,
            chunk_size=CHUNK_SIZE,
            confidence_aware=confidence_aware,
            geometry_confidence_threshold=GEOMETRY_CONFIDENCE_THRESHOLD,
            geometry_confidence_power=GEOMETRY_CONFIDENCE_POWER,
        )
        raw_result = surface_output.result
        filled = nearest_valid_fill(
            raw_result.rgb,
            raw_result.depth,
            raw_result.valid_mask,
            raw_result.confidence,
            radius=FILL_RADIUS,
        )
        prediction = _to_native_rgb(filled.rgb, native_size_wh)
        output_path = render_dir / f"{index:05d}.png"
        Image.fromarray(prediction, mode="RGB").save(output_path)

        raw_coverage = float(raw_result.stats["target_coverage_percent"])
        filled_coverage = float(filled.filled_valid_mask.float().mean().item() * 100.0)
        raw_coverage_sum += raw_coverage
        filled_coverage_sum += filled_coverage
        frame_seconds = time.perf_counter() - frame_started

        if index == 0 or index + 1 == len(frames) or (index + 1) % 25 == 0:
            print(
                f"[METHOD3] sequence={sequence.name} frame={index + 1}/{len(frames)} "
                f"source={frame.rgb_path.name} output={output_path.name} "
                f"raw_coverage={raw_coverage:.2f}% "
                f"filled_coverage={filled_coverage:.2f}% seconds={frame_seconds:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - sequence_started
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        if device.type == "cuda"
        else 0.0
    )
    print(
        f"[METHOD3] completed sequence={sequence.name} renderer={renderer} "
        f"frames={len(frames)} runtime_seconds={elapsed:.3f} "
        f"seconds_per_frame={elapsed / len(frames):.4f} "
        f"mean_raw_coverage={raw_coverage_sum / len(frames):.2f}% "
        f"mean_filled_coverage={filled_coverage_sum / len(frames):.2f}% "
        f"peak_gpu_memory_mb={peak_gpu_memory_mb:.1f} output={render_dir}",
        flush=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Method-3 iMED-NVS submission")
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    renderer = _renderer()
    device = _device()
    sequences = _discover_sequences(args.input)
    if not sequences:
        raise SystemExit(f"No iMED-NVS sequence directories found under {args.input}")

    print(f"[METHOD3] discovered_sequences={len(sequences)} renderer={renderer}", flush=True)
    for sequence in sequences:
        _render_sequence(sequence, args.output / sequence.name, renderer, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
