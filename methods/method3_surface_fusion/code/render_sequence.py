#!/usr/bin/env python3
"""Render exact MV1A (M3-A), surface splats (M3-B), or confidence-gated M3-C.

The script reuses the confirmed MV1A implementation from a read-only source
directory. It reads only Endoscope2/L RGB-D and challenge calibration files.
It never discovers or opens Endoscope1 imagery, depth, or masks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_MV1A_ROOT = Path(
    "/mnt/cluster/workspaces/venkateda/Endo-4DGS/method1_rgbd_reprojection"
)
MV1A_IMAGE_ID = "aaa952c110d1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Method 3 surface-fusion development renderer")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mv1a_root", type=Path, default=DEFAULT_MV1A_ROOT)
    parser.add_argument(
        "--renderer", choices=("mv1a", "surface", "surface_confidence"), required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame_id", type=int)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--overwrite", action="store_true")

    # Exact confirmed MV1A settings. Defaults must remain unchanged for M3-A.
    parser.add_argument("--visibility_tolerance_mm", type=float, default=1.0)
    parser.add_argument("--visibility_relative", type=float, default=0.01)
    parser.add_argument("--depth_softness", type=float, default=8.0)
    parser.add_argument("--fill_radius", type=int, choices=(1, 2, 3), default=3)

    # M3-B surface settings. They are ignored by exact MV1A mode.
    parser.add_argument("--footprint_scale", type=float, default=1.0)
    parser.add_argument("--radius_min", type=float, default=0.5)
    parser.add_argument("--radius_max", type=float, default=2.0)
    parser.add_argument("--surface_support", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument("--mahalanobis_cutoff", type=float, default=9.0)
    parser.add_argument("--depth_discontinuity_mm", type=float, default=2.0)
    parser.add_argument("--depth_discontinuity_relative", type=float, default=0.02)
    parser.add_argument("--chunk_size", type=int, default=65536)

    # M3-C source-only geometry confidence. These settings are ignored by
    # M3-A/M3-B so their established configurations remain reproducible.
    parser.add_argument("--geometry_confidence_threshold", type=float, default=0.25)
    parser.add_argument("--geometry_confidence_power", type=float, default=1.0)
    return parser.parse_args()


def _bootstrap_mv1a(mv1a_root: Path) -> None:
    """Expose the exact read-only MV1A modules without copying or changing them."""

    root = mv1a_root.resolve()
    candidate = root / "docker_submission" / "phase11_candidate"
    required = (
        root / "camera.py",
        root / "reprojection.py",
        root / "soft_splatting.py",
        root / "hole_filling.py",
        candidate / "imed_io.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MV1A source is incomplete; missing: {missing}")
    # Candidate first for its inference-only imed_io; root supplies geometry.
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(candidate))


def _select_indices(frames: Sequence, frame_id: int | None, max_frames: int | None) -> list[int]:
    if frame_id is not None:
        matches = [index for index, frame in enumerate(frames) if frame.frame_id == frame_id]
        if not matches:
            raise ValueError(f"frame_{frame_id:06d} is not in the synchronized source stream")
        if max_frames is not None:
            raise ValueError("--frame_id and --max_frames cannot be combined")
        return matches
    indices = list(range(len(frames)))
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max_frames must be positive")
        indices = indices[:max_frames]
    if not indices:
        raise ValueError("selected frame set is empty")
    return indices


def _prepare_output(output_dir: Path, selected_indices: Sequence[int], overwrite: bool) -> None:
    conflicts = [
        output_dir / "renders" / f"{index:05d}.png"
        for index in selected_indices
        if (output_dir / "renders" / f"{index:05d}.png").exists()
    ]
    if conflicts and not overwrite:
        raise FileExistsError(
            f"{len(conflicts)} selected render(s) already exist; use a fresh output or --overwrite"
        )
    for subdirectory in (
        "renders",
        "rgb",
        "depth",
        "valid_mask",
        "filled_valid_mask",
        "fill_mask",
        "confidence",
        "normals",
        "surface_radius",
        "surface_validity",
        "surface_used",
        "surface_recovered",
        "geometry_confidence",
        "depth_confidence",
        "tangent_confidence",
        "viewing_confidence",
    ):
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _rgb_uint8(rgb: torch.Tensor) -> np.ndarray:
    return (
        rgb.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


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
    return _rgb_uint8(rgb_nchw[0].permute(1, 2, 0))


def _normal_visualization(normals: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    visual = ((normals.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round().to(torch.uint8)
    visual = torch.where(valid[..., None], visual, torch.zeros_like(visual))
    return visual.cpu().numpy()


def _mean(rows: Sequence[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


@torch.inference_mode()
def _run(args: argparse.Namespace) -> None:
    from hole_filling import nearest_valid_fill
    from imed_io import collect_source_frames, load_calibration, load_source_frame, scale_intrinsics
    from soft_splatting import render_soft_splat
    from surface_splat import render_surface_splat

    sequence_dir = args.data_root / args.sequence
    frames = collect_source_frames(sequence_dir)
    selected_indices = _select_indices(frames, args.frame_id, args.max_frames)
    _prepare_output(args.output_dir, selected_indices, args.overwrite)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    calibration = load_calibration(sequence_dir)
    first_index = selected_indices[0]
    first = load_source_frame(frames[first_index])
    depth_height, depth_width = first.depth.shape
    native_size_wh = first.native_rgb_size_wh
    internal_size_wh = (depth_width, depth_height)
    K_source = torch.from_numpy(
        scale_intrinsics(calibration.K_source_full, native_size_wh, internal_size_wh).astype(
            np.float32
        )
    ).to(device)
    K_target = torch.from_numpy(
        scale_intrinsics(calibration.K_target_full, native_size_wh, internal_size_wh).astype(
            np.float32
        )
    ).to(device)
    T_target_source = torch.from_numpy(calibration.T_target_source.astype(np.float32)).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"[METHOD3] sequence={args.sequence} renderer={args.renderer} "
        f"frames={len(selected_indices)}/{len(frames)} device={device} "
        f"internal={depth_width}x{depth_height} "
        f"output={native_size_wh[0]}x{native_size_wh[1]} "
        "source_tool_masking=disabled stereo=disabled temporal=disabled",
        flush=True,
    )

    sequence_started = time.perf_counter()
    frame_rows = []
    for position, stream_index in enumerate(selected_indices):
        frame = frames[stream_index]
        loaded = first if stream_index == first_index else load_source_frame(frame)
        if loaded.depth.shape != (depth_height, depth_width):
            raise ValueError(f"depth resolution changed at {frame.depth_path}")
        if loaded.native_rgb_size_wh != native_size_wh:
            raise ValueError(f"source RGB resolution changed at {frame.rgb_path}")

        source_rgb = torch.from_numpy(np.ascontiguousarray(loaded.rgb_at_depth_resolution)).to(
            device=device, dtype=torch.float32
        )
        source_depth = torch.from_numpy(np.ascontiguousarray(loaded.depth)).to(
            device=device, dtype=torch.float32
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        frame_started = time.perf_counter()
        surface_output = None
        if args.renderer == "mv1a":
            raw_result = render_soft_splat(
                source_rgb=source_rgb,
                source_depth=source_depth,
                K_source=K_source,
                K_target=K_target,
                T_target_source=T_target_source,
                target_size=(depth_height, depth_width),
                source_valid_mask=None,
                mode="soft_depth",
                visibility_tolerance_mm=args.visibility_tolerance_mm,
                visibility_relative=args.visibility_relative,
                depth_softness=args.depth_softness,
            )
        else:
            surface_output = render_surface_splat(
                source_rgb=source_rgb,
                source_depth=source_depth,
                K_source=K_source,
                K_target=K_target,
                T_target_source=T_target_source,
                target_size=(depth_height, depth_width),
                source_valid_mask=None,
                visibility_tolerance_mm=args.visibility_tolerance_mm,
                visibility_relative=args.visibility_relative,
                depth_softness=args.depth_softness,
                depth_discontinuity_mm=args.depth_discontinuity_mm,
                depth_discontinuity_relative=args.depth_discontinuity_relative,
                footprint_scale=args.footprint_scale,
                radius_min=args.radius_min,
                radius_max=args.radius_max,
                support_size=args.surface_support,
                mahalanobis_cutoff=args.mahalanobis_cutoff,
                chunk_size=args.chunk_size,
                confidence_aware=args.renderer == "surface_confidence",
                geometry_confidence_threshold=args.geometry_confidence_threshold,
                geometry_confidence_power=args.geometry_confidence_power,
            )
            raw_result = surface_output.result

        filled = nearest_valid_fill(
            raw_result.rgb,
            raw_result.depth,
            raw_result.valid_mask,
            raw_result.confidence,
            radius=args.fill_radius,
        )
        native_prediction = _to_native_rgb(filled.rgb, native_size_wh)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        frame_seconds = time.perf_counter() - frame_started

        stem = f"{stream_index:05d}"
        _save_png(args.output_dir / "renders" / f"{stem}.png", native_prediction)
        _save_png(args.output_dir / "rgb" / f"{stem}.png", _rgb_uint8(filled.rgb))
        np.save(
            args.output_dir / "depth" / f"{stem}.npy",
            filled.depth.detach().cpu().numpy().astype(np.float32),
        )
        _save_png(
            args.output_dir / "valid_mask" / f"{stem}.png",
            raw_result.valid_mask.detach().cpu().numpy().astype(np.uint8) * 255,
        )
        _save_png(
            args.output_dir / "filled_valid_mask" / f"{stem}.png",
            filled.filled_valid_mask.detach().cpu().numpy().astype(np.uint8) * 255,
        )
        _save_png(
            args.output_dir / "fill_mask" / f"{stem}.png",
            filled.fill_mask.detach().cpu().numpy().astype(np.uint8) * 255,
        )
        confidence_np = filled.confidence.detach().cpu().numpy().astype(np.float32)
        np.save(args.output_dir / "confidence" / f"{stem}.npy", confidence_np)
        _save_png(
            args.output_dir / "confidence" / f"{stem}.png",
            (np.clip(confidence_np, 0.0, 1.0) * 255.0).round().astype(np.uint8),
        )

        if surface_output is not None:
            surface_valid = surface_output.surface_valid_map_source
            radius = surface_output.radius_map_source
            _save_png(
                args.output_dir / "normals" / f"{stem}.png",
                _normal_visualization(surface_output.normal_map_source, surface_valid),
            )
            radius_np = radius.detach().cpu().numpy().astype(np.float32)
            np.save(args.output_dir / "surface_radius" / f"{stem}.npy", radius_np)
            _save_png(
                args.output_dir / "surface_radius" / f"{stem}.png",
                (np.clip(radius_np / args.radius_max, 0.0, 1.0) * 255.0)
                .round()
                .astype(np.uint8),
            )
            _save_png(
                args.output_dir / "surface_validity" / f"{stem}.png",
                surface_valid.detach().cpu().numpy().astype(np.uint8) * 255,
            )
            _save_png(
                args.output_dir / "surface_used" / f"{stem}.png",
                surface_output.surface_used_map_source.detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
                * 255,
            )
            geometry_confidence_np = (
                surface_output.geometry_confidence_map_source.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            np.save(
                args.output_dir / "geometry_confidence" / f"{stem}.npy",
                geometry_confidence_np,
            )
            _save_png(
                args.output_dir / "geometry_confidence" / f"{stem}.png",
                (np.clip(geometry_confidence_np, 0.0, 1.0) * 255.0)
                .round()
                .astype(np.uint8),
            )
            for confidence_name, confidence_map in (
                ("depth_confidence", surface_output.depth_confidence_map_source),
                ("tangent_confidence", surface_output.tangent_confidence_map_source),
                ("viewing_confidence", surface_output.viewing_confidence_map_source),
            ):
                confidence_component = (
                    confidence_map.detach().cpu().numpy().astype(np.float32)
                )
                _save_png(
                    args.output_dir / confidence_name / f"{stem}.png",
                    (np.clip(confidence_component, 0.0, 1.0) * 255.0)
                    .round()
                    .astype(np.uint8),
                )
            _save_png(
                args.output_dir / "surface_recovered" / f"{stem}.png",
                surface_output.surface_recovered_mask_target.detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
                * 255,
            )

        row = {
            "stream_index": stream_index,
            # Compatibility aliases used by the fixed Method-1 evaluator.
            "sequence_index": stream_index,
            "frame_id": frame.frame_id,
            "source_frame_id": frame.frame_id,
            "source_rgb": frame.rgb_path.name,
            "output": f"{stem}.png",
            "time_seconds": frame_seconds,
            "raw_coverage_percent": float(raw_result.stats["target_coverage_percent"]),
            "filled_coverage_percent": float(
                filled.filled_valid_mask.float().mean().item() * 100.0
            ),
            "fill_fraction_percent": float(filled.fill_mask.float().mean().item() * 100.0),
            "mean_confidence": float(filled.confidence.mean().item()),
            "source_surface_valid_percent": float(
                raw_result.stats.get("source_surface_valid_percent", 0.0)
            ),
            "source_surface_used_percent": float(
                raw_result.stats.get("source_surface_used_percent", 0.0)
            ),
            "mean_source_geometry_confidence": float(
                raw_result.stats.get("mean_source_geometry_confidence", 0.0)
            ),
            "mean_valid_surface_geometry_confidence": float(
                raw_result.stats.get("mean_valid_surface_geometry_confidence", 0.0)
            ),
            "mean_projected_surface_radius_pixels": float(
                raw_result.stats.get("mean_projected_surface_radius_pixels", 0.0)
            ),
            "surface_recovered_target_pixels": float(
                raw_result.stats.get("surface_recovered_target_pixels", 0.0)
            ),
            "surface_recovered_target_percent": float(
                raw_result.stats.get("surface_recovered_target_percent", 0.0)
            ),
        }
        frame_rows.append(row)
        if position == 0 or position + 1 == len(selected_indices) or (position + 1) % 25 == 0:
            print(
                f"[METHOD3] {position + 1}/{len(selected_indices)} "
                f"source={frame.rgb_path.name} output={stem}.png "
                f"raw_coverage={row['raw_coverage_percent']:.2f}% "
                f"filled_coverage={row['filled_coverage_percent']:.2f}% "
                f"seconds={frame_seconds:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - sequence_started
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        "method": {
            "mv1a": "M3-A_exact_MV1A",
            "surface": "M3-B_surface",
            "surface_confidence": "M3-C_confidence_surface",
        }[args.renderer],
        "sequence": args.sequence,
        "mv1a_source_root": str(args.mv1a_root.resolve()),
        "mv1a_image_id": MV1A_IMAGE_ID,
        "input_policy": {
            "source_rgb": "endoscope2/L only",
            "source_depth": "endoscope2/depthL only",
            "source_tool_masking": "disabled",
            "stereo": "disabled: missing right depth and L/R extrinsic",
            "target_rgb_access": False,
            "target_depth_access": False,
            "target_mask_access": False,
        },
        "configuration": {
            "renderer": args.renderer,
            "visibility_tolerance_mm": args.visibility_tolerance_mm,
            "visibility_relative": args.visibility_relative,
            "depth_softness": args.depth_softness,
            "hole_fill": "nearest_valid",
            "fill_radius": args.fill_radius,
            "depth_filtering": "none",
            "footprint_scale": args.footprint_scale if args.renderer != "mv1a" else None,
            "radius_min": args.radius_min if args.renderer != "mv1a" else None,
            "radius_max": args.radius_max if args.renderer != "mv1a" else None,
            "surface_support": args.surface_support if args.renderer != "mv1a" else None,
            "mahalanobis_cutoff": (
                args.mahalanobis_cutoff if args.renderer != "mv1a" else None
            ),
            "depth_discontinuity_mm": (
                args.depth_discontinuity_mm if args.renderer != "mv1a" else None
            ),
            "depth_discontinuity_relative": (
                args.depth_discontinuity_relative if args.renderer != "mv1a" else None
            ),
            "geometry_confidence": args.renderer == "surface_confidence",
            "geometry_confidence_formula": (
                "depth_margin * tangent_stability * abs(normal_dot_view)"
                if args.renderer == "surface_confidence"
                else None
            ),
            "geometry_confidence_threshold": (
                args.geometry_confidence_threshold
                if args.renderer == "surface_confidence"
                else None
            ),
            "geometry_confidence_power": (
                args.geometry_confidence_power
                if args.renderer == "surface_confidence"
                else None
            ),
        },
        "internal_size_hw": [depth_height, depth_width],
        "target_size_hw": [depth_height, depth_width],
        "output_size_wh": list(native_size_wh),
        "selected_frames": len(selected_indices),
        "total_source_frames": len(frames),
        "total_time_seconds": elapsed,
        "mean_time_seconds_per_frame": _mean(frame_rows, "time_seconds"),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "mean_raw_coverage_percent": _mean(frame_rows, "raw_coverage_percent"),
        "mean_filled_coverage_percent": _mean(frame_rows, "filled_coverage_percent"),
        "mean_fill_fraction_percent": _mean(frame_rows, "fill_fraction_percent"),
        "mean_confidence": _mean(frame_rows, "mean_confidence"),
        "mean_source_surface_valid_percent": _mean(
            frame_rows, "source_surface_valid_percent"
        ),
        "mean_source_surface_used_percent": _mean(
            frame_rows, "source_surface_used_percent"
        ),
        "mean_source_geometry_confidence": _mean(
            frame_rows, "mean_source_geometry_confidence"
        ),
        "mean_valid_surface_geometry_confidence": _mean(
            frame_rows, "mean_valid_surface_geometry_confidence"
        ),
        "mean_projected_surface_radius_pixels": _mean(
            frame_rows, "mean_projected_surface_radius_pixels"
        ),
        "mean_surface_recovered_target_percent": _mean(
            frame_rows, "surface_recovered_target_percent"
        ),
        "frames": frame_rows,
    }
    (args.output_dir / "render_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[METHOD3] completed renderer={args.renderer} frames={len(selected_indices)} "
        f"seconds={elapsed:.3f} seconds_per_frame={summary['mean_time_seconds_per_frame']:.4f} "
        f"raw_coverage={summary['mean_raw_coverage_percent']:.2f}% "
        f"filled_coverage={summary['mean_filled_coverage_percent']:.2f}% "
        f"peak_gpu_memory_mb={peak_gpu_memory_mb:.1f}",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    _bootstrap_mv1a(args.mv1a_root)
    _run(args)


if __name__ == "__main__":
    main()
