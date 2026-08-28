#!/usr/bin/env python3
"""Evaluation-only PSNR/SSIM for iMED Endoscope-1 targets.

This is the only Method-1 program that opens Endoscope 1 RGB or target tool
masks. It never changes calibration, rendering, parameters, or predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from camera import backproject_depth, project_points, transform_points
from inspect_dataset import collect_source_frames, load_calibration, scale_intrinsics
from metrics_compat import masked_psnr, masked_ssim
from visualize_debug import (
    absolute_error_to_rgb,
    red_cyan_alignment_overlay,
    save_evaluation_debug,
)


PROTOCOL_CORRECTED = "corrected_geometry"
PROTOCOL_ADAPTED = "adapted_baseline"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rendered Endoscope-1 predictions")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--prediction_dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation_dir",
        type=Path,
        help="Default: <prediction_dir>/evaluation",
    )
    parser.add_argument(
        "--mask_protocol",
        choices=("both", PROTOCOL_CORRECTED, PROTOCOL_ADAPTED),
        default="both",
        help="Report both by default; never select a protocol from GT performance",
    )
    parser.add_argument(
        "--include_first_frame_support",
        action="store_true",
        help=(
            "Adapted-baseline only: multiply by first prediction nonzero support, matching render.py/metrics.py; "
            "method-dependent and therefore disabled by default"
        ),
    )
    parser.add_argument(
        "--allow_subset",
        action="store_true",
        help="Allow evaluation of an explicitly rendered frame subset; disabled for official sequence reports",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save Phase-10 evaluation-only montages for representative and metric-extreme frames",
    )
    parser.add_argument(
        "--visualization_protocol",
        choices=(PROTOCOL_CORRECTED, PROTOCOL_ADAPTED),
        default=PROTOCOL_ADAPTED,
    )
    parser.add_argument(
        "--visualization_dir",
        type=Path,
        help="Default: <evaluation_dir>/visualizations; use /tmp on restricted systems",
    )
    parser.add_argument(
        "--visualization_extra_indices",
        type=int,
        nargs="*",
        default=(),
        help="Additional zero-based sorted-stream indices",
    )
    return parser.parse_args()


def _extract_frame_id(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("frame_"):
        raise ValueError(f"unexpected target filename: {path.name}")
    return int(stem.split("_")[-1])


def _resize_rgb(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_prediction(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        raise ValueError(f"prediction resolution must be {width}x{height}: {path}")
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_binary_mask(path: Path, width: int, height: int, invert_tool: bool = False) -> np.ndarray:
    image = Image.open(path).convert("L")
    raw_original = np.asarray(image)
    if not np.all(np.isin(np.unique(raw_original), [0, 255])):
        raise ValueError(f"mask must contain only 0/255: {path}")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
    raw = np.asarray(image)
    return (raw == 0) if invert_tool else (raw > 127)


def _morphological_overlap(projected_uv: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    rounded = np.round(projected_uv).astype(np.int64)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < target_width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < target_height)
    )
    rounded = rounded[in_bounds]
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    mask[rounded[:, 1], rounded[:, 0]] = 1
    # Exactly match the challenge-adapted metrics.py morphology.
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=np.uint8), iterations=2)
    mask = ndimage.binary_closing(mask, structure=np.ones((11, 11), dtype=np.uint8), iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype(bool)


def _project_first_source_depth(
    sequence_dir: Path,
    K2: np.ndarray,
    K1: np.ndarray,
    target_height: int,
    target_width: int,
    numeric_dtype: np.dtype = np.float64,
    baseline_world_chain: bool = False,
) -> np.ndarray:
    depth_paths = sorted((sequence_dir / "endoscope2" / "depthL").glob("frame_*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"no source depth maps in {sequence_dir}")
    depth = np.load(depth_paths[0]).astype(numeric_dtype)
    calibration = load_calibration(sequence_dir)
    points_cam2 = backproject_depth(depth, K2).reshape(-1, 3)
    valid = np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)
    # Build the relative transform at the selected precision. The adapted
    # baseline parses and composes these matrices in float32.
    T_world_cam1 = calibration.T_world_cam1.astype(numeric_dtype)
    T_world_cam2 = calibration.T_world_cam2.astype(numeric_dtype)
    if baseline_world_chain:
        # Preserve metrics.py's float32 operation order for its compatibility
        # protocol: cam2 -> world -> cam1 using homogeneous column vectors.
        valid_points = points_cam2[valid]
        points_cam2_h = np.concatenate(
            [valid_points, np.ones((valid_points.shape[0], 1), dtype=numeric_dtype)], axis=1
        )
        points_world_h = (T_world_cam2 @ points_cam2_h.T).T
        points_cam1 = (np.linalg.inv(T_world_cam1) @ points_world_h.T).T[:, :3]
    else:
        T_cam1_cam2 = np.linalg.inv(T_world_cam1) @ T_world_cam2
        points_cam1 = transform_points(points_cam2[valid], T_cam1_cam2)
    front = np.isfinite(points_cam1).all(axis=1) & (points_cam1[:, 2] > 1e-6)
    uv_cam1 = project_points(points_cam1[front], K1)
    uv_cam1 = uv_cam1[np.isfinite(uv_cam1).all(axis=1)]
    return _morphological_overlap(uv_cam1, target_height, target_width)


def _corrected_overlap_mask(
    sequence_dir: Path,
    output_height: int,
    output_width: int,
    target_full_size: Tuple[int, int],
) -> np.ndarray:
    """Method-independent overlap with K scaled to actual depth/output grids."""

    calibration = load_calibration(sequence_dir)
    source_depth_path = sorted((sequence_dir / "endoscope2" / "depthL").glob("frame_*.npy"))[0]
    source_depth = np.load(source_depth_path, mmap_mode="r")
    source_rgb_path = sorted((sequence_dir / "endoscope2" / "L").glob("frame_*.png"))[0]
    source_full_width, source_full_height = Image.open(source_rgb_path).size
    source_scale_x = source_full_width / source_depth.shape[1]
    source_scale_y = source_full_height / source_depth.shape[0]
    target_full_width, target_full_height = target_full_size
    target_scale_x = target_full_width / output_width
    target_scale_y = target_full_height / output_height
    K2 = scale_intrinsics(calibration.K2_L_full, source_scale_x, source_scale_y)
    K1 = scale_intrinsics(calibration.K1_L_full, target_scale_x, target_scale_y)
    return _project_first_source_depth(sequence_dir, K2, K1, output_height, output_width)


def _adapted_baseline_overlap_mask(
    sequence_dir: Path,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    """Faithfully reproduce metrics.py's unscaled-K/provisional-size behavior."""

    calibration = load_calibration(sequence_dir)
    provisional_height = int(round(float(calibration.K1_L_full[1, 2]) * 2))
    provisional_width = int(round(float(calibration.K1_L_full[0, 2]) * 2))
    if provisional_height <= 0 or provisional_width <= 0:
        source_depth_path = sorted((sequence_dir / "endoscope2" / "depthL").glob("frame_*.npy"))[0]
        provisional_height, provisional_width = np.load(source_depth_path, mmap_mode="r").shape
    mask = _project_first_source_depth(
        sequence_dir,
        calibration.K2_L_full.astype(np.float32),
        calibration.K1_L_full.astype(np.float32),
        provisional_height,
        provisional_width,
        numeric_dtype=np.float32,
        baseline_world_chain=True,
    )
    if mask.shape != (output_height, output_width):
        mask_t = torch.from_numpy(mask.astype(np.float32))[None, None]
        mask_t = F.interpolate(mask_t, size=(output_height, output_width), mode="nearest")
        mask = mask_t[0, 0].numpy() > 0.5
    return mask


def _prediction_paths(prediction_dir: Path) -> List[Path]:
    paths = sorted((prediction_dir / "rgb").glob("[0-9][0-9][0-9][0-9][0-9].png"))
    if not paths:
        raise FileNotFoundError(f"no five-digit RGB predictions in {prediction_dir / 'rgb'}")
    indices = [int(path.stem) for path in paths]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate prediction indices")
    return paths


def _load_render_summary(prediction_dir: Path) -> dict:
    summary_path = prediction_dir / "render_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"missing {summary_path}; it is required to verify sequence and timestamp synchronization"
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("frame", "psnr", "ssim", "coverage"))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def _summary(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(np.median(array))


def _select_visualization_frames(rows: Sequence[dict], extra_indices: Sequence[int]) -> Dict[int, List[str]]:
    """Select temporal anchors and metric extremes without changing predictions."""

    if not rows:
        raise ValueError("cannot select visualizations from an empty evaluation")
    ordered = sorted(rows, key=lambda row: int(row["output_index"]))
    selected: Dict[int, List[str]] = {}

    def add(row: dict, reason: str) -> None:
        selected.setdefault(int(row["output_index"]), []).append(reason)

    add(ordered[0], "first")
    add(ordered[len(ordered) // 2], "middle")
    add(ordered[-1], "last")
    add(max(ordered, key=lambda row: float(row["psnr"])), "best_psnr")
    add(min(ordered, key=lambda row: float(row["psnr"])), "worst_psnr")
    add(max(ordered, key=lambda row: float(row["ssim"])), "best_ssim")
    add(min(ordered, key=lambda row: float(row["ssim"])), "worst_ssim")

    by_index = {int(row["output_index"]): row for row in ordered}
    for index in extra_indices:
        if index not in by_index:
            raise ValueError(f"visualization index {index} is not among evaluated predictions")
        add(by_index[index], "requested")
    return selected


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> Optional[float]:
    mask = np.asarray(mask, dtype=bool)
    return float(np.asarray(values, dtype=np.float64)[mask].mean()) if mask.any() else None


def _frame_failure_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    evaluation_mask: np.ndarray,
    raw_valid: np.ndarray,
    source_tool_fraction: Optional[float],
) -> dict:
    """Quantify failure correlates; these are diagnostics, not causal labels."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
    raw_valid = np.asarray(raw_valid, dtype=bool)
    error = np.mean(np.abs(prediction - target), axis=2)
    prediction_luma = 0.299 * prediction[..., 0] + 0.587 * prediction[..., 1] + 0.114 * prediction[..., 2]
    target_luma = 0.299 * target[..., 0] + 0.587 * target[..., 1] + 0.114 * target[..., 2]

    grad_y, grad_x = np.gradient(target_luma)
    gradient = np.hypot(grad_x, grad_y)
    gradient_threshold = float(np.percentile(gradient[evaluation_mask], 90))
    high_gradient = evaluation_mask & (gradient >= gradient_threshold)
    low_gradient = evaluation_mask & ~high_gradient
    specular = evaluation_mask & (np.max(target, axis=2) >= 0.90)
    boundary = evaluation_mask & (
        ndimage.binary_dilation(raw_valid, iterations=2)
        ^ ndimage.binary_erosion(raw_valid, iterations=2, border_value=0)
    )
    covered = evaluation_mask & raw_valid
    uncovered = evaluation_mask & ~raw_valid

    color_bias = [
        _masked_mean(prediction[..., channel] - target[..., channel], evaluation_mask)
        for channel in range(3)
    ]
    diagnostics = {
        "mean_absolute_rgb_error": _masked_mean(error, evaluation_mask),
        "mean_luminance_bias_prediction_minus_gt": _masked_mean(
            prediction_luma - target_luma, evaluation_mask
        ),
        "mean_rgb_bias_prediction_minus_gt": color_bias,
        "raw_coverage_in_evaluation_mask_percent": 100.0 * float(covered.sum()) / float(evaluation_mask.sum()),
        "covered_region_error": _masked_mean(error, covered),
        "uncovered_region_error": _masked_mean(error, uncovered),
        "projection_boundary_error": _masked_mean(error, boundary),
        "high_target_gradient_error": _masked_mean(error, high_gradient),
        "low_target_gradient_error": _masked_mean(error, low_gradient),
        "specular_region_error": _masked_mean(error, specular),
        "specular_evaluation_fraction_percent": 100.0 * float(specular.sum()) / float(evaluation_mask.sum()),
        "source_tool_fraction_percent": (
            None if source_tool_fraction is None else 100.0 * source_tool_fraction
        ),
    }

    hints = []
    overall = diagnostics["mean_absolute_rgb_error"] or 0.0
    luminance_bias = diagnostics["mean_luminance_bias_prediction_minus_gt"] or 0.0
    if abs(luminance_bias) >= 0.03:
        hints.append("global illumination/exposure mismatch is plausible")
    if diagnostics["raw_coverage_in_evaluation_mask_percent"] < 95.0:
        hints.append("disocclusion or projection holes materially affect the evaluation region")
    if diagnostics["specular_region_error"] is not None and diagnostics["specular_region_error"] > 1.25 * overall:
        hints.append("specular regions have disproportionately high error")
    edge_error = diagnostics["high_target_gradient_error"]
    smooth_error = diagnostics["low_target_gradient_error"]
    if edge_error is not None and smooth_error is not None and edge_error > 1.25 * smooth_error:
        hints.append("edges/thin structures have disproportionately high error")
    boundary_error = diagnostics["projection_boundary_error"]
    covered_error = diagnostics["covered_region_error"]
    if boundary_error is not None and covered_error is not None and boundary_error > 1.25 * covered_error:
        hints.append("projection boundaries are error-concentrated; inspect depth and disocclusion")
    hints.append("use the red/cyan overlay to assess global pose or calibration misalignment visually")
    diagnostics["heuristic_hints_not_causal_conclusions"] = hints
    return diagnostics


def _save_phase10_visualizations(
    visualization_dir: Path,
    protocol: str,
    prediction_dir: Path,
    sequence_dir: Path,
    predictions: Sequence[Path],
    target_rgb_paths: Sequence[Path],
    target_mask_paths: Sequence[Path],
    global_mask: np.ndarray,
    detailed_rows: Sequence[dict],
    extra_indices: Sequence[int],
    overwrite: bool,
) -> None:
    if visualization_dir.exists() and any(visualization_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"visualization directory is not empty: {visualization_dir}; use a new /tmp directory"
        )
    visualization_dir.mkdir(parents=True, exist_ok=True)
    for name in ("montages", "absolute_error", "red_cyan_overlay"):
        (visualization_dir / name).mkdir(parents=True, exist_ok=True)

    selected = _select_visualization_frames(detailed_rows, extra_indices)
    rows_by_index = {int(row["output_index"]): row for row in detailed_rows}
    predictions_by_index = {int(path.stem): path for path in predictions}
    source_frames = collect_source_frames(sequence_dir)
    selected_records = []

    for output_index in sorted(selected):
        row = rows_by_index[output_index]
        prediction_path = predictions_by_index[output_index]
        prediction_image = Image.open(prediction_path).convert("RGB")
        output_width, output_height = prediction_image.size
        prediction = np.asarray(prediction_image, dtype=np.float32) / 255.0
        target = _resize_rgb(target_rgb_paths[output_index], output_width, output_height)
        source_rgb = _resize_rgb(source_frames[output_index].rgb_path, output_width, output_height)
        source_depth = np.load(source_frames[output_index].depth_path).astype(np.float32)
        projected_depth_path = prediction_dir / "depth" / f"{output_index:05d}.npy"
        confidence_path = prediction_dir / "confidence" / f"{output_index:05d}.npy"
        raw_valid_path = prediction_dir / "valid_mask" / f"{output_index:05d}.png"
        for required in (projected_depth_path, confidence_path, raw_valid_path):
            if not required.exists():
                raise FileNotFoundError(f"Phase-10 visualization requires {required}")
        projected_depth = np.load(projected_depth_path).astype(np.float32)
        confidence = np.load(confidence_path).astype(np.float32)
        raw_valid = _load_binary_mask(raw_valid_path, output_width, output_height)
        if projected_depth.shape != (output_height, output_width):
            raise ValueError(f"projected depth shape mismatch: {projected_depth_path}")
        if confidence.shape != (output_height, output_width):
            raise ValueError(f"confidence shape mismatch: {confidence_path}")

        target_tissue = _load_binary_mask(
            target_mask_paths[output_index], output_width, output_height, invert_tool=True
        )
        evaluation_mask = target_tissue & global_mask
        source_tool_fraction = None
        source_mask_path = source_frames[output_index].tool_mask_path
        if source_mask_path is not None:
            source_tissue = _load_binary_mask(
                source_mask_path, source_depth.shape[1], source_depth.shape[0], invert_tool=True
            )
            source_tool_fraction = float((~source_tissue).mean())

        diagnostics = _frame_failure_diagnostics(
            prediction,
            target,
            evaluation_mask,
            raw_valid,
            source_tool_fraction,
        )
        reasons = selected[output_index]
        title = (
            f"{prediction_path.name} | {','.join(reasons)} | "
            f"PSNR={float(row['psnr']):.3f} SSIM={float(row['ssim']):.4f} "
            f"coverage={float(row['coverage']):.2f}%"
        )
        save_evaluation_debug(
            visualization_dir / "montages" / prediction_path.name,
            source_rgb,
            source_depth,
            prediction,
            projected_depth,
            confidence,
            raw_valid,
            evaluation_mask,
            target,
            title,
        )
        Image.fromarray(
            absolute_error_to_rgb(prediction, target, evaluation_mask), mode="RGB"
        ).save(visualization_dir / "absolute_error" / prediction_path.name)
        Image.fromarray(
            red_cyan_alignment_overlay(prediction, target, evaluation_mask), mode="RGB"
        ).save(visualization_dir / "red_cyan_overlay" / prediction_path.name)
        selected_records.append(
            {
                "frame": prediction_path.name,
                "output_index": output_index,
                "source_frame": source_frames[output_index].rgb_path.name,
                "target_frame": target_rgb_paths[output_index].name,
                "selection_reasons": reasons,
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "coverage": float(row["coverage"]),
                "diagnostics": diagnostics,
            }
        )

    report = {
        "phase": 10,
        "protocol": protocol,
        "evaluation_only_target_access": True,
        "predictions_or_parameters_modified": False,
        "diagnostic_warning": "Heuristic correlations do not establish failure causality.",
        "selected_frame_count": len(selected_records),
        "frames": selected_records,
    }
    (visualization_dir / "phase10_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
    sequence_dir = args.data_root / args.sequence
    evaluation_dir = args.evaluation_dir or (args.prediction_dir / "evaluation")
    results_path = evaluation_dir / "results.json"
    if evaluation_dir.exists() and any(evaluation_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"evaluation directory is not empty: {evaluation_dir}; use a new directory or --overwrite"
        )

    predictions = _prediction_paths(args.prediction_dir)
    first_prediction = Image.open(predictions[0]).convert("RGB")
    output_width, output_height = first_prediction.size
    target_rgb_paths = sorted((sequence_dir / "endoscope1" / "L").glob("frame_*.png"))
    target_mask_paths = sorted((sequence_dir / "endoscope1" / "toolL").glob("frame_*.png"))
    if not target_rgb_paths:
        raise FileNotFoundError(f"no evaluation GT in {sequence_dir / 'endoscope1' / 'L'}")
    if len(target_mask_paths) != len(target_rgb_paths):
        raise FileNotFoundError(
            f"official target tool masks are incomplete: {len(target_mask_paths)} masks for {len(target_rgb_paths)} GT frames"
        )
    target_full_size = Image.open(target_rgb_paths[0]).size
    target_ids = [_extract_frame_id(path) for path in target_rgb_paths]
    target_mask_ids = [_extract_frame_id(path) for path in target_mask_paths]
    if target_mask_ids != target_ids:
        raise ValueError("Endoscope 1 RGB/tool-mask frame ids do not match")
    render_summary = _load_render_summary(args.prediction_dir)
    if render_summary.get("sequence") != args.sequence:
        raise ValueError(
            f"prediction summary is for {render_summary.get('sequence')!r}, not {args.sequence!r}"
        )
    if render_summary.get("target_size_hw") != [output_height, output_width]:
        raise ValueError(
            f"render summary target size {render_summary.get('target_size_hw')} does not match "
            f"RGB prediction size {[output_height, output_width]}"
        )
    render_frame_map = {
        int(frame["sequence_index"]): int(frame["source_frame_id"])
        for frame in render_summary.get("frames", [])
    }
    prediction_indices = [int(path.stem) for path in predictions]
    if set(prediction_indices) != set(render_frame_map):
        raise ValueError("RGB prediction indices do not match render_summary.json frame indices")
    if not args.allow_subset and prediction_indices != list(range(len(target_rgb_paths))):
        raise ValueError(
            f"official sequence evaluation requires all {len(target_rgb_paths)} frames with indices 00000.."
            f"{len(target_rgb_paths) - 1:05d}; found {len(predictions)}. Use --allow_subset only for debugging."
        )

    protocols = (
        [PROTOCOL_CORRECTED, PROTOCOL_ADAPTED]
        if args.mask_protocol == "both"
        else [args.mask_protocol]
    )
    if args.save_visualizations and args.visualization_protocol not in protocols:
        raise ValueError(
            f"visualization protocol {args.visualization_protocol!r} was not evaluated; "
            f"select --mask_protocol both or {args.visualization_protocol}"
        )
    if args.include_first_frame_support and PROTOCOL_ADAPTED not in protocols:
        raise ValueError("--include_first_frame_support is valid only with the adapted_baseline protocol")
    global_masks: Dict[str, np.ndarray] = {}
    if PROTOCOL_CORRECTED in protocols:
        global_masks[PROTOCOL_CORRECTED] = _corrected_overlap_mask(
            sequence_dir, output_height, output_width, target_full_size
        )
    if PROTOCOL_ADAPTED in protocols:
        adapted = _adapted_baseline_overlap_mask(sequence_dir, output_height, output_width)
        if args.include_first_frame_support:
            first_array = np.asarray(first_prediction)
            first_support = first_array[..., :3].sum(axis=2) > 0
            adapted &= first_support
        global_masks[PROTOCOL_ADAPTED] = adapted

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for protocol, mask in global_masks.items():
        protocol_dir = evaluation_dir / protocol
        (protocol_dir / "masks").mkdir(parents=True, exist_ok=True)
        _save_mask = (mask.astype(np.uint8) * 255)
        Image.fromarray(_save_mask, mode="L").save(protocol_dir / "global_overlap_mask.png")
    if args.include_first_frame_support:
        support = (np.asarray(first_prediction)[..., :3].sum(axis=2) > 0).astype(np.uint8) * 255
        Image.fromarray(support, mode="L").save(evaluation_dir / "first_prediction_support_mask.png")

    device = torch.device(args.device)
    rows_by_protocol: Dict[str, List[dict]] = {protocol: [] for protocol in protocols}
    detailed_by_protocol: Dict[str, List[dict]] = {protocol: [] for protocol in protocols}
    for prediction_path in predictions:
        output_index = int(prediction_path.stem)
        if output_index >= len(target_rgb_paths):
            raise IndexError(f"prediction {prediction_path.name} has no target frame at sorted index {output_index}")
        expected_id = render_frame_map.get(output_index)
        if expected_id is not None and expected_id != target_ids[output_index]:
            raise ValueError(
                f"frame synchronization mismatch at {prediction_path.name}: rendered source id {expected_id}, "
                f"target id {target_ids[output_index]}"
            )

        prediction_np = _load_prediction(prediction_path, output_width, output_height)
        gt_np = _resize_rgb(target_rgb_paths[output_index], output_width, output_height)
        target_tissue = _load_binary_mask(
            target_mask_paths[output_index], output_width, output_height, invert_tool=True
        )
        raw_valid_path = args.prediction_dir / "valid_mask" / prediction_path.name
        filled_valid_path = args.prediction_dir / "filled_valid_mask" / prediction_path.name
        if not raw_valid_path.exists():
            raise FileNotFoundError(f"missing raw prediction validity: {raw_valid_path}")
        raw_valid = _load_binary_mask(raw_valid_path, output_width, output_height)
        filled_valid = (
            _load_binary_mask(filled_valid_path, output_width, output_height)
            if filled_valid_path.exists()
            else raw_valid
        )

        prediction_t = torch.from_numpy(np.ascontiguousarray(prediction_np)).permute(2, 0, 1)[None].to(device)
        gt_t = torch.from_numpy(np.ascontiguousarray(gt_np)).permute(2, 0, 1)[None].to(device)
        for protocol in protocols:
            evaluation_mask = target_tissue & global_masks[protocol]
            evaluation_pixels = int(evaluation_mask.sum())
            if evaluation_pixels == 0:
                raise ValueError(f"empty evaluation mask for {protocol}, frame {prediction_path.name}")
            mask_t = torch.from_numpy(np.ascontiguousarray(evaluation_mask))[None, None].to(
                device=device, dtype=torch.float32
            )
            with torch.no_grad():
                psnr = float(masked_psnr(prediction_t, gt_t, mask_t).item())
                ssim = float(masked_ssim(prediction_t, gt_t, mask_t).item())
            raw_covered = int((raw_valid & evaluation_mask).sum())
            filled_covered = int((filled_valid & evaluation_mask).sum())
            raw_coverage = 100.0 * raw_covered / evaluation_pixels
            coverage = 100.0 * filled_covered / evaluation_pixels
            row = {
                "frame": prediction_path.name,
                "psnr": psnr,
                "ssim": ssim,
                "coverage": coverage,
            }
            rows_by_protocol[protocol].append(row)
            detailed_by_protocol[protocol].append(
                {
                    **row,
                    "output_index": output_index,
                    "target_frame": target_rgb_paths[output_index].name,
                    "target_frame_id": target_ids[output_index],
                    "evaluation_pixels": evaluation_pixels,
                    "raw_covered_pixels": raw_covered,
                    "raw_coverage": raw_coverage,
                    "filled_covered_pixels": filled_covered,
                }
            )
            Image.fromarray(evaluation_mask.astype(np.uint8) * 255, mode="L").save(
                evaluation_dir / protocol / "masks" / prediction_path.name
            )

    protocol_results = {}
    for protocol in protocols:
        detailed = detailed_by_protocol[protocol]
        mean_psnr, median_psnr = _summary([row["psnr"] for row in detailed])
        mean_ssim, median_ssim = _summary([row["ssim"] for row in detailed])
        mean_coverage, median_coverage = _summary([row["coverage"] for row in detailed])
        mean_raw_coverage, median_raw_coverage = _summary([row["raw_coverage"] for row in detailed])
        protocol_results[protocol] = {
            "mean_psnr": mean_psnr,
            "median_psnr": median_psnr,
            "mean_ssim": mean_ssim,
            "median_ssim": median_ssim,
            "mean_coverage_percent": mean_coverage,
            "median_coverage_percent": median_coverage,
            "mean_raw_coverage_percent": mean_raw_coverage,
            "median_raw_coverage_percent": median_raw_coverage,
            "global_overlap_pixels": int(global_masks[protocol].sum()),
            "global_overlap_percent": float(global_masks[protocol].mean() * 100.0),
            "frames": detailed,
        }
        _write_csv(evaluation_dir / f"per_frame_{protocol}.csv", rows_by_protocol[protocol])
    if len(protocols) == 1:
        _write_csv(evaluation_dir / "per_frame.csv", rows_by_protocol[protocols[0]])

    results = {
        "sequence": args.sequence,
        "prediction_dir": str(args.prediction_dir),
        "evaluation_only_target_access": True,
        "target_rgb_used_for_rendering_or_tuning": False,
        "metric_implementation": "challenge-adapted Endo-4DGS masked PSNR/SSIM",
        "prediction_invalid_pixels_are_penalized_inside_evaluation_mask": True,
        "include_first_frame_support": args.include_first_frame_support,
        "subset_evaluation": args.allow_subset,
        "protocol_warning": (
            "corrected_geometry scales K to the actual grids; adapted_baseline faithfully reproduces the inspected "
            "metrics.py unscaled-K/provisional-size behavior. Organizer-official status remains unverified."
        ),
        "frame_count": len(predictions),
        "output_size_hw": [output_height, output_width],
        "protocols": protocol_results,
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.save_visualizations:
        visualization_dir = args.visualization_dir or (evaluation_dir / "visualizations")
        _save_phase10_visualizations(
            visualization_dir=visualization_dir,
            protocol=args.visualization_protocol,
            prediction_dir=args.prediction_dir,
            sequence_dir=sequence_dir,
            predictions=predictions,
            target_rgb_paths=target_rgb_paths,
            target_mask_paths=target_mask_paths,
            global_mask=global_masks[args.visualization_protocol],
            detailed_rows=detailed_by_protocol[args.visualization_protocol],
            extra_indices=args.visualization_extra_indices,
            overwrite=args.overwrite,
        )
    compact = {
        protocol: {key: value for key, value in protocol_results[protocol].items() if key != "frames"}
        for protocol in protocols
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
