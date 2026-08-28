#!/usr/bin/env python3
"""Phase-7 source-only one-frame, subset, and full-sequence renderer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image

from depth_filtering import apply_depth_filter
from hole_filling import apply_hole_fill
from inspect_dataset import SourceFrame, collect_source_frames, load_calibration, load_source_frame, scale_intrinsics
from reprojection import render_nearest_zbuffer
from soft_splatting import render_soft_splat
from visualize_debug import depth_to_rgb, scalar_to_rgb, save_hole_fill_debug, save_inference_debug


OUTPUT_SUBDIRECTORIES = (
    "rgb",
    "depth",
    "valid_mask",
    "filled_valid_mask",
    "fill_mask",
    "confidence",
    "correspondence",
    "debug",
)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Endoscope-2 RGB-D frames into Endoscope 1")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--frame_id",
        type=int,
        help="Render exactly one dataset id, e.g. 2 for frame_000002; omit for a sequence",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Inclusive sorted-stream index")
    parser.add_argument("--end_index", type=int, help="Exclusive sorted-stream index")
    parser.add_argument("--max_frames", type=int, help="Optional cap after start/end selection")
    parser.add_argument("--renderer", choices=("nearest", "bilinear", "soft_depth"), default="nearest")
    parser.add_argument("--mask_source_tools", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_height", type=int)
    parser.add_argument("--target_width", type=int)
    parser.add_argument("--visibility_tolerance_mm", type=float, default=1.0)
    parser.add_argument("--visibility_relative", type=float, default=0.01)
    parser.add_argument("--depth_softness", type=float, default=8.0)
    parser.add_argument(
        "--depth_filter",
        choices=("none", "median3", "bilateral"),
        default="none",
        help="Source-only depth preprocessing; default preserves the supplied metric depth",
    )
    parser.add_argument("--bilateral_kernel_size", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument("--bilateral_sigma_spatial_pixels", type=float, default=2.0)
    parser.add_argument("--bilateral_sigma_depth_mm", type=float, default=3.0)
    parser.add_argument(
        "--hole_fill",
        choices=("none", "morphological", "nearest", "inpaint"),
        default="none",
        help="Raw H0 validity is always preserved",
    )
    parser.add_argument("--fill_radius", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--morph_min_neighbors", type=int, choices=range(1, 9), default=7)
    parser.add_argument("--morph_min_confidence", type=float, default=0.1)
    parser.add_argument("--inpaint_max_area", type=int, default=25)
    parser.add_argument(
        "--debug_every",
        type=int,
        default=0,
        help="Save a montage every N selected frames; 0 saves first/middle/last",
    )
    parser.add_argument("--no_debug", action="store_true", help="Disable montage images; JSON statistics are still saved")
    parser.add_argument("--no_correspondence", action="store_true", help="Skip large correspondence arrays")
    parser.add_argument(
        "--minimal_outputs",
        action="store_true",
        help="Save only RGB and raw/filled validity needed by evaluation, plus JSON provenance",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of matching frame outputs; never deletes files")
    return parser.parse_args()


def _select_frame_indices(frames: Sequence[SourceFrame], args: argparse.Namespace) -> List[int]:
    if args.frame_id is not None:
        if args.start_index != 0 or args.end_index is not None or args.max_frames is not None:
            raise ValueError("--frame_id cannot be combined with start/end/max frame selection")
        matches = [index for index, frame in enumerate(frames) if frame.frame_id == args.frame_id]
        if not matches:
            raise ValueError(f"frame_{args.frame_id:06d} is not present")
        return matches

    end_index = len(frames) if args.end_index is None else args.end_index
    if args.start_index < 0 or end_index < 0 or args.start_index >= end_index or end_index > len(frames):
        raise ValueError(
            f"invalid index interval [{args.start_index},{end_index}) for {len(frames)} source frames"
        )
    indices = list(range(args.start_index, end_index))
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max_frames must be positive")
        indices = indices[: args.max_frames]
    if not indices:
        raise ValueError("frame selection is empty")
    return indices


def _debug_frame_indices(selected_indices: Sequence[int], args: argparse.Namespace) -> Set[int]:
    if args.no_debug:
        return set()
    if args.debug_every < 0:
        raise ValueError("--debug_every must be non-negative")
    if args.debug_every > 0:
        debug = {index for position, index in enumerate(selected_indices) if position % args.debug_every == 0}
        debug.add(selected_indices[-1])
        return debug
    return {
        selected_indices[0],
        selected_indices[len(selected_indices) // 2],
        selected_indices[-1],
    }


def _prepare_output(output_dir: Path, selected_indices: Sequence[int], overwrite: bool, save_raw: bool) -> None:
    existing = [output_dir / "rgb" / f"{index:05d}.png" for index in selected_indices]
    conflicts = [path for path in existing if path.exists()]
    if conflicts and not overwrite:
        preview = ", ".join(str(path) for path in conflicts[:3])
        raise FileExistsError(
            f"{len(conflicts)} selected RGB output(s) already exist ({preview}); use a new directory or --overwrite"
        )
    for subdirectory in OUTPUT_SUBDIRECTORIES:
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    if save_raw:
        for subdirectory in ("raw/rgb", "raw/depth", "raw/confidence", "raw/valid_mask"):
            (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)


def _render_raw(
    args: argparse.Namespace,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    tissue: Optional[torch.Tensor],
    K_source: torch.Tensor,
    K_target: torch.Tensor,
    T_target_source: torch.Tensor,
    target_size: Tuple[int, int],
):
    if args.renderer == "nearest":
        return render_nearest_zbuffer(
            rgb, depth, K_source, K_target, T_target_source, target_size, tissue
        )
    return render_soft_splat(
        rgb,
        depth,
        K_source,
        K_target,
        T_target_source,
        target_size,
        source_valid_mask=tissue,
        mode=args.renderer,
        visibility_tolerance_mm=args.visibility_tolerance_mm,
        visibility_relative=args.visibility_relative,
        depth_softness=args.depth_softness,
    )


def _to_numpy_outputs(raw_result, filled_result) -> Dict[str, np.ndarray]:
    return {
        "raw_rgb": raw_result.rgb.detach().cpu().numpy(),
        "raw_depth": raw_result.depth.detach().cpu().numpy(),
        "raw_valid": filled_result.raw_valid_mask.detach().cpu().numpy(),
        "raw_confidence": raw_result.confidence.detach().cpu().numpy(),
        "rgb": filled_result.rgb.detach().cpu().numpy(),
        "depth": filled_result.depth.detach().cpu().numpy(),
        "confidence": filled_result.confidence.detach().cpu().numpy(),
        "fill_mask": filled_result.fill_mask.detach().cpu().numpy(),
        "filled_valid": filled_result.filled_valid_mask.detach().cpu().numpy(),
        "correspondence": raw_result.correspondence.detach().cpu().numpy(),
        "source_projection": raw_result.source_projection.detach().cpu().numpy(),
        "hit_count": raw_result.hit_count.detach().cpu().numpy(),
    }


def _save_frame_outputs(
    output_dir: Path,
    stem: str,
    arrays: Dict[str, np.ndarray],
    source_rgb: np.ndarray,
    source_depth: np.ndarray,
    save_raw: bool,
    save_correspondence: bool,
    save_debug: bool,
    minimal_outputs: bool,
) -> None:
    _save_png(output_dir / "rgb" / f"{stem}.png", (np.clip(arrays["rgb"], 0, 1) * 255).round().astype(np.uint8))
    # Raw geometric validity is never silently replaced by filled validity.
    _save_png(output_dir / "valid_mask" / f"{stem}.png", arrays["raw_valid"].astype(np.uint8) * 255)
    _save_png(
        output_dir / "filled_valid_mask" / f"{stem}.png",
        arrays["filled_valid"].astype(np.uint8) * 255,
    )
    if not minimal_outputs:
        np.save(output_dir / "depth" / f"{stem}.npy", arrays["depth"].astype(np.float32))
        _save_png(output_dir / "depth" / f"{stem}.png", depth_to_rgb(arrays["depth"], arrays["filled_valid"]))
        _save_png(output_dir / "fill_mask" / f"{stem}.png", arrays["fill_mask"].astype(np.uint8) * 255)
        np.save(output_dir / "confidence" / f"{stem}.npy", arrays["confidence"].astype(np.float32))
        _save_png(output_dir / "confidence" / f"{stem}.png", scalar_to_rgb(arrays["confidence"]))

        if save_raw:
            _save_png(
                output_dir / "raw" / "rgb" / f"{stem}.png",
                (np.clip(arrays["raw_rgb"], 0, 1) * 255).round().astype(np.uint8),
            )
            np.save(output_dir / "raw" / "depth" / f"{stem}.npy", arrays["raw_depth"].astype(np.float32))
            _save_png(
                output_dir / "raw" / "depth" / f"{stem}.png",
                depth_to_rgb(arrays["raw_depth"], arrays["raw_valid"]),
            )
            np.save(
                output_dir / "raw" / "confidence" / f"{stem}.npy",
                arrays["raw_confidence"].astype(np.float32),
            )
            _save_png(
                output_dir / "raw" / "valid_mask" / f"{stem}.png",
                arrays["raw_valid"].astype(np.uint8) * 255,
            )

        if save_correspondence:
            np.save(
                output_dir / "correspondence" / f"{stem}.npy",
                arrays["correspondence"].astype(np.float32),
            )
            np.save(
                output_dir / "correspondence" / f"{stem}_source_to_target.npy",
                arrays["source_projection"].astype(np.float32),
            )

    if save_debug:
        np.save(output_dir / "debug" / f"{stem}_hit_count.npy", arrays["hit_count"])
        save_inference_debug(
            output_dir / "debug" / f"{stem}.png",
            source_rgb,
            source_depth,
            arrays["rgb"],
            arrays["depth"],
            arrays["filled_valid"],
            arrays["confidence"],
        )
        save_hole_fill_debug(
            output_dir / "debug" / f"{stem}_hole_fill.png",
            arrays["raw_rgb"],
            arrays["raw_depth"],
            arrays["raw_valid"],
            arrays["rgb"],
            arrays["depth"],
            arrays["confidence"],
            arrays["fill_mask"],
            arrays["filled_valid"],
        )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def main() -> None:
    args = _parse_args()
    sequence_dir = args.data_root / args.sequence
    frames = collect_source_frames(sequence_dir)
    selected_indices = _select_frame_indices(frames, args)
    debug_indices = _debug_frame_indices(selected_indices, args)
    if args.mask_source_tools:
        missing = [frames[index].frame_id for index in selected_indices if frames[index].tool_mask_path is None]
        if missing:
            raise FileNotFoundError(
                f"--mask_source_tools requested, but {len(missing)} selected source masks are missing; "
                f"first missing frame is {missing[0]:06d}"
            )
    if (args.target_height is None) != (args.target_width is None):
        raise ValueError("target_height and target_width must be supplied together")

    # Load only the first selected Endoscope-2 frame to establish the invariant
    # stream resolution and full-RGB-to-depth scale. Reuse it in the main loop.
    first_index = selected_indices[0]
    first_loaded = load_source_frame(frames[first_index], load_tool_mask=args.mask_source_tools)
    first_rgb, first_depth, _, (scale_x, scale_y) = first_loaded
    target_height = args.target_height or first_depth.shape[0]
    target_width = args.target_width or first_depth.shape[1]
    target_size = (target_height, target_width)

    calibration = load_calibration(sequence_dir)
    K2 = scale_intrinsics(calibration.K2_L_full, scale_x, scale_y)
    target_scale_x = (first_depth.shape[1] * scale_x) / target_width
    target_scale_y = (first_depth.shape[0] * scale_y) / target_height
    K1 = scale_intrinsics(calibration.K1_L_full, target_scale_x, target_scale_y)
    T_cam1_cam2 = calibration.T_cam1_cam2

    save_raw = args.hole_fill != "none"
    _prepare_output(args.output_dir, selected_indices, args.overwrite, save_raw=save_raw)
    device = torch.device(args.device)
    K_source = torch.from_numpy(K2.astype(np.float32)).to(device)
    K_target = torch.from_numpy(K1.astype(np.float32)).to(device)
    T_target_source = torch.from_numpy(T_cam1_cam2.astype(np.float32)).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    sequence_started = time.perf_counter()
    reports: List[dict] = []
    for position, sequence_index in enumerate(selected_indices):
        frame_wall_started = time.perf_counter()
        if sequence_index == first_index:
            rgb_np, depth_np, tissue_np, frame_scales = first_loaded
        else:
            rgb_np, depth_np, tissue_np, frame_scales = load_source_frame(
                frames[sequence_index], load_tool_mask=args.mask_source_tools
            )
        if depth_np.shape != first_depth.shape or not np.allclose(frame_scales, (scale_x, scale_y)):
            raise ValueError(f"source resolution changed at {frames[sequence_index].rgb_path}")

        rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).to(device=device, dtype=torch.float32)
        depth = torch.from_numpy(np.ascontiguousarray(depth_np)).to(device=device, dtype=torch.float32)
        tissue = (
            None
            if tissue_np is None
            else torch.from_numpy(np.ascontiguousarray(tissue_np)).to(device=device, dtype=torch.bool)
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        depth_filter_started = time.perf_counter()
        depth = apply_depth_filter(
            depth,
            method=args.depth_filter,
            bilateral_kernel_size=args.bilateral_kernel_size,
            bilateral_sigma_spatial_pixels=args.bilateral_sigma_spatial_pixels,
            bilateral_sigma_depth_mm=args.bilateral_sigma_depth_mm,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        depth_filter_elapsed = time.perf_counter() - depth_filter_started

        render_started = time.perf_counter()
        raw_result = _render_raw(
            args, rgb, depth, tissue, K_source, K_target, T_target_source, target_size
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        render_elapsed = time.perf_counter() - render_started

        fill_started = time.perf_counter()
        filled_result = apply_hole_fill(
            raw_result.rgb,
            raw_result.depth,
            raw_result.valid_mask,
            raw_result.confidence,
            method=args.hole_fill,
            radius=args.fill_radius,
            morph_min_neighbors=args.morph_min_neighbors,
            morph_min_confidence=args.morph_min_confidence,
            inpaint_max_area=args.inpaint_max_area,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fill_elapsed = time.perf_counter() - fill_started

        arrays = _to_numpy_outputs(raw_result, filled_result)
        stem = f"{sequence_index:05d}"
        _save_frame_outputs(
            args.output_dir,
            stem,
            arrays,
            rgb_np,
            depth_np,
            save_raw=save_raw,
            save_correspondence=not args.no_correspondence,
            save_debug=sequence_index in debug_indices,
            minimal_outputs=args.minimal_outputs,
        )
        raw_covered = int(arrays["raw_valid"].sum())
        filled_pixels = int(arrays["fill_mask"].sum())
        filled_covered = int(arrays["filled_valid"].sum())
        target_pixels = target_height * target_width
        frame_wall_elapsed = time.perf_counter() - frame_wall_started
        report = {
            "sequence_index": sequence_index,
            "source_frame_id": frames[sequence_index].frame_id,
            "source_frame": frames[sequence_index].rgb_path.name,
            "output_stem": stem,
            "render_time_seconds": render_elapsed,
            "depth_filter_time_seconds": depth_filter_elapsed,
            "hole_fill_time_seconds": fill_elapsed,
            "compute_time_seconds": depth_filter_elapsed + render_elapsed + fill_elapsed,
            "wall_time_seconds": frame_wall_elapsed,
            "source_pixels": int(depth_np.size),
            "source_valid_pixels": int(raw_result.stats["source_valid_pixels"]),
            "projected_in_bounds_pixels": int(raw_result.stats["projected_in_bounds_pixels"]),
            "raw_target_covered_pixels": raw_covered,
            "raw_target_coverage_percent": 100.0 * raw_covered / target_pixels,
            "filled_pixels": filled_pixels,
            "filled_target_covered_pixels": filled_covered,
            "filled_target_coverage_percent": 100.0 * filled_covered / target_pixels,
            "debug_saved": sequence_index in debug_indices,
        }
        reports.append(report)
        (args.output_dir / "debug" / f"{stem}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{position + 1:04d}/{len(selected_indices):04d}] {frames[sequence_index].rgb_path.name} "
            f"depth_filter={depth_filter_elapsed:.4f}s render={render_elapsed:.4f}s fill={fill_elapsed:.4f}s "
            f"raw={report['raw_target_coverage_percent']:.2f}% "
            f"filled={report['filled_target_coverage_percent']:.2f}%"
        )

    sequence_wall_elapsed = time.perf_counter() - sequence_started
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    summary = {
        "sequence": args.sequence,
        "data_root": str(args.data_root),
        "device": str(device),
        "renderer": args.renderer,
        "depth_filter": args.depth_filter,
        "bilateral_kernel_size": args.bilateral_kernel_size,
        "bilateral_sigma_spatial_pixels": args.bilateral_sigma_spatial_pixels,
        "bilateral_sigma_depth_mm": args.bilateral_sigma_depth_mm,
        "mask_source_tools": args.mask_source_tools,
        "hole_fill": args.hole_fill,
        "fill_radius": args.fill_radius,
        "selected_frame_count": len(reports),
        "selected_sequence_indices": [selected_indices[0], selected_indices[-1]],
        "first_source_frame": reports[0]["source_frame"],
        "last_source_frame": reports[-1]["source_frame"],
        "target_size_hw": [target_height, target_width],
        "filename_convention": "zero-based sorted-stream index, five digits",
        "total_sequence_runtime_seconds": sequence_wall_elapsed,
        "mean_wall_time_per_frame_seconds": _mean([r["wall_time_seconds"] for r in reports]),
        "median_wall_time_per_frame_seconds": _median([r["wall_time_seconds"] for r in reports]),
        "mean_compute_time_per_frame_seconds": _mean([r["compute_time_seconds"] for r in reports]),
        "median_compute_time_per_frame_seconds": _median([r["compute_time_seconds"] for r in reports]),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "total_source_pixels": int(sum(r["source_pixels"] for r in reports)),
        "total_source_valid_pixels": int(sum(r["source_valid_pixels"] for r in reports)),
        "total_projected_in_bounds_pixels": int(sum(r["projected_in_bounds_pixels"] for r in reports)),
        "mean_raw_target_coverage_percent": _mean([r["raw_target_coverage_percent"] for r in reports]),
        "median_raw_target_coverage_percent": _median([r["raw_target_coverage_percent"] for r in reports]),
        "mean_filled_target_coverage_percent": _mean([r["filled_target_coverage_percent"] for r in reports]),
        "median_filled_target_coverage_percent": _median([r["filled_target_coverage_percent"] for r in reports]),
        "debug_sequence_indices": sorted(debug_indices),
        "correspondence_saved": not args.no_correspondence and not args.minimal_outputs,
        "minimal_outputs": args.minimal_outputs,
        "K2_L_at_render_resolution": K2.tolist(),
        "K1_L_at_render_resolution": K1.tolist(),
        "T_cam1_cam2": T_cam1_cam2.tolist(),
        "frames": reports,
    }
    (args.output_dir / "render_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
