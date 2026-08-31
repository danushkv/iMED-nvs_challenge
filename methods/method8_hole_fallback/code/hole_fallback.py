#!/usr/bin/env python3
"""Evaluation-gated, strict hole-only Method-2 fallback for frozen M3B.

The diagnostic accesses Endoscope1 RGB/tool masks only for local evaluation.
F1 itself is deterministic and uses only the saved M3B support mask plus an
already-trained Method-2 prediction.  No M3B-valid pixel is ever overwritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


DEFAULT_EVALUATOR = Path(
    "/mnt/cluster/workspaces/venkateda/Endo-4DGS/"
    "method1_rgbd_reprojection/evaluate.py"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M3B + strict hole-only Method-2 fallback")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--m3b_dir", type=Path, required=True)
    parser.add_argument("--m2_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--boundary_exclusion_pixels",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "Keep this many immediate hole layers as frozen M3B. Zero is strict F1; "
            "one/two are the only allowed F2 variants."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_evaluator(path: Path):
    root = path.resolve().parent
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("method8_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _five_digit_images(directory: Path) -> list[Path]:
    return sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9].png"))


def _rgb(path: Path, expected_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        if expected_size is not None and image.size != expected_size:
            raise ValueError(f"Expected {expected_size}, found {image.size}: {path}")
        return np.asarray(image, dtype=np.uint8).copy()


def _mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("L")
        if image.size != expected_size:
            raise ValueError(f"Expected {expected_size}, found {image.size}: {path}")
        values = np.asarray(image)
    if not np.all(np.isin(np.unique(values), (0, 255))):
        raise ValueError(f"Mask is not binary 0/255: {path}")
    return values > 127


def _target_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def _target_tissue(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        original = np.asarray(opened.convert("L"))
        if not np.all(np.isin(np.unique(original), (0, 255))):
            raise ValueError(f"Target tool mask is not binary: {path}")
        image = opened.convert("L")
        if image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        values = np.asarray(image)
    return values == 0


def _frame_id(path: Path) -> int:
    if not path.stem.startswith("frame_"):
        raise ValueError(f"Unexpected target filename: {path.name}")
    return int(path.stem.rsplit("_", 1)[-1])


def _psnr(sse: float, values: int) -> float:
    return float(-10.0 * math.log10(max(sse / max(values, 1), 1.0e-12)))


def _metric_pair(evaluator, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, device):
    prediction_t = torch.from_numpy(np.ascontiguousarray(prediction.astype(np.float32) / 255.0))
    prediction_t = prediction_t.permute(2, 0, 1)[None].to(device)
    target_t = torch.from_numpy(np.ascontiguousarray(target.astype(np.float32) / 255.0))
    target_t = target_t.permute(2, 0, 1)[None].to(device)
    mask_t = torch.from_numpy(np.ascontiguousarray(mask))[None, None].to(
        device=device, dtype=torch.float32
    )
    with torch.no_grad():
        return (
            float(evaluator.masked_psnr(prediction_t, target_t, mask_t).item()),
            float(evaluator.masked_ssim(prediction_t, target_t, mask_t).item()),
        )


def _find_internal_m2_renders(m2_dir: Path) -> Path:
    candidates = sorted((m2_dir / "test").glob("ours_*/renders"))
    if not candidates:
        raise FileNotFoundError(f"No Method-2 internal renders under {m2_dir / 'test'}")
    return candidates[-1]


def _validate_alignment(
    args: argparse.Namespace,
    m3b_paths: list[Path],
    m2_paths: list[Path],
    target_paths: list[Path],
    target_mask_paths: list[Path],
) -> dict:
    if not m3b_paths:
        raise FileNotFoundError(f"No M3B predictions under {args.m3b_dir / 'rgb'}")
    expected_names = [f"{index:05d}.png" for index in range(len(m3b_paths))]
    if [path.name for path in m3b_paths] != expected_names:
        raise ValueError("M3B predictions are not a contiguous five-digit stream")
    if [path.name for path in m2_paths] != expected_names:
        raise ValueError("Method-2 predictions do not exactly match the M3B filenames")
    if len(target_paths) != len(m3b_paths) or len(target_mask_paths) != len(m3b_paths):
        raise ValueError("Target evaluation streams do not match the prediction count")
    target_ids = [_frame_id(path) for path in target_paths]
    if [_frame_id(path) for path in target_mask_paths] != target_ids:
        raise ValueError("Target RGB/tool-mask ids are not synchronized")

    summary_path = args.m3b_dir / "render_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("method") != "M3-B_surface" or summary.get("sequence") != args.sequence:
        raise ValueError("Frozen M3B summary does not identify the requested M3B sequence")
    frame_map = {
        int(frame["sequence_index"]): int(frame["source_frame_id"])
        for frame in summary.get("frames", [])
    }
    if list(sorted(frame_map)) != list(range(len(m3b_paths))):
        raise ValueError("M3B summary indices do not cover the prediction stream")
    for index, target_id in enumerate(target_ids):
        if frame_map[index] != target_id:
            raise ValueError(
                f"M3B/target timestamp mismatch at {index:05d}: {frame_map[index]} != {target_id}"
            )

    manifest_path = args.m2_dir / "run_manifest.txt"
    if manifest_path.is_file():
        manifest = manifest_path.read_text(encoding="utf-8")
        if f"sequence={args.sequence}\n" not in manifest:
            raise ValueError(f"Method-2 manifest is not for {args.sequence}")
        if "experiment=B1_metric_l1_w5e-5\n" not in manifest:
            raise ValueError("Method-2 fallback is not the frozen submitted metric-L1 configuration")
    return summary


def _prepare_output(output: Path) -> None:
    for name in (
        "rgb",
        "renders",
        "valid_mask",
        "filled_valid_mask",
        "fallback_mask",
        "debug",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}; use a fresh directory"
        )
    evaluator = _load_evaluator(args.evaluator)
    device = torch.device(args.device)
    sequence_dir = args.data_root / args.sequence
    m2_internal_dir = _find_internal_m2_renders(args.m2_dir)
    m2_native_dir = args.m2_dir / "renders"
    m3b_paths = _five_digit_images(args.m3b_dir / "rgb")
    m2_paths = _five_digit_images(m2_internal_dir)
    target_paths = sorted((sequence_dir / "endoscope1" / "L").glob("frame_*.png"))
    target_mask_paths = sorted((sequence_dir / "endoscope1" / "toolL").glob("frame_*.png"))
    m3b_summary = _validate_alignment(
        args, m3b_paths, m2_paths, target_paths, target_mask_paths
    )

    first_m3b = _rgb(m3b_paths[0])
    height, width = first_m3b.shape[:2]
    size = (width, height)
    with Image.open(target_paths[0]) as target_first:
        target_full_size = target_first.size
    overlap = evaluator._corrected_overlap_mask(
        sequence_dir, height, width, target_full_size
    )

    # Pass 1: evaluation-only hole diagnostic. No output is created before
    # determining whether the frozen Method-2 render is better on M3B holes.
    hole_pixels = 0
    evaluation_pixels = 0
    m3b_hole_sse = 0.0
    m2_hole_sse = 0.0
    m3b_hole_absolute = 0.0
    m2_hole_absolute = 0.0
    for index, (m3b_path, m2_path, target_path, mask_path) in enumerate(
        zip(m3b_paths, m2_paths, target_paths, target_mask_paths)
    ):
        m3b = _rgb(m3b_path, size).astype(np.float32) / 255.0
        m2 = _rgb(m2_path, size).astype(np.float32) / 255.0
        target = _target_rgb(target_path, size).astype(np.float32) / 255.0
        tissue = _target_tissue(mask_path, size)
        filled_valid = _mask(
            args.m3b_dir / "filled_valid_mask" / f"{index:05d}.png", size
        )
        evaluation = overlap & tissue
        holes = evaluation & (~filled_valid)
        evaluation_pixels += int(evaluation.sum())
        count = int(holes.sum())
        hole_pixels += count
        if count:
            m3b_error = m3b[holes] - target[holes]
            m2_error = m2[holes] - target[holes]
            m3b_hole_sse += float(np.square(m3b_error.astype(np.float64)).sum())
            m2_hole_sse += float(np.square(m2_error.astype(np.float64)).sum())
            m3b_hole_absolute += float(np.abs(m3b_error).sum(dtype=np.float64))
            m2_hole_absolute += float(np.abs(m2_error).sum(dtype=np.float64))
    if hole_pixels == 0:
        raise ValueError("No evaluated frozen-M3B holes exist")

    diagnostic = {
        "evaluated_hole_pixels": hole_pixels,
        "evaluated_pixels": evaluation_pixels,
        "hole_fraction_of_evaluated_image_percent": 100.0 * hole_pixels / evaluation_pixels,
        "m3b_hole_psnr": _psnr(m3b_hole_sse, hole_pixels * 3),
        "m2_hole_psnr": _psnr(m2_hole_sse, hole_pixels * 3),
        "m3b_hole_mae": m3b_hole_absolute / (hole_pixels * 3),
        "m2_hole_mae": m2_hole_absolute / (hole_pixels * 3),
    }
    hole_improves = (
        diagnostic["m2_hole_psnr"] > diagnostic["m3b_hole_psnr"]
        and diagnostic["m2_hole_mae"] < diagnostic["m3b_hole_mae"]
    )
    diagnostic["m2_better_on_holes"] = hole_improves
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hole_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostic, indent=2), flush=True)
    if not hole_improves:
        report = {
            "sequence": args.sequence,
            "diagnostic": diagnostic,
            "F1_created": False,
            "decision": "STOP",
            "reason": "Frozen Method 2 is not better than current M3B values on evaluated M3B-hole pixels.",
        }
        (args.output_dir / "fallback_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print("[METHOD8] STOP: Method 2 is not better on evaluated M3B holes", flush=True)
        return

    _prepare_output(args.output_dir)
    m3b_psnr, m3b_ssim, f1_psnr, f1_ssim = [], [], [], []
    replaced_total = 0
    evaluated_replaced = 0
    valid_changed_total = 0
    frame_rows = []
    native_m2_paths = _five_digit_images(m2_native_dir)
    native_m3b_paths = _five_digit_images(args.m3b_dir / "renders")
    if [path.name for path in native_m2_paths] != [path.name for path in native_m3b_paths]:
        raise ValueError("Method-2 and M3B native render streams do not align")

    for index, (m3b_path, m2_path, target_path, mask_path) in enumerate(
        zip(m3b_paths, m2_paths, target_paths, target_mask_paths)
    ):
        name = f"{index:05d}.png"
        m3b = _rgb(m3b_path, size)
        m2 = _rgb(m2_path, size)
        target = _target_rgb(target_path, size)
        tissue = _target_tissue(mask_path, size)
        filled_valid = _mask(args.m3b_dir / "filled_valid_mask" / name, size)
        raw_valid = _mask(args.m3b_dir / "valid_mask" / name, size)
        # Distance is positive only inside unsupported pixels.  Threshold 0
        # exactly reproduces F1. Thresholds 1/2 preserve the immediate one/two
        # boundary layers and use Method 2 only deeper inside a hole.
        distance_from_valid = ndimage.distance_transform_edt(~filled_valid)
        fallback = (~filled_valid) & (
            distance_from_valid > float(args.boundary_exclusion_pixels)
        )
        f1 = m3b.copy()
        f1[fallback] = m2[fallback]
        changed_valid = int(np.any(f1[filled_valid] != m3b[filled_valid], axis=1).sum())
        if changed_valid != 0:
            raise AssertionError(f"CRITICAL INVARIANT FAILED at {name}: {changed_valid}")
        valid_changed_total += changed_valid
        replaced_total += int(fallback.sum())
        evaluation = overlap & tissue
        evaluated_replaced += int((fallback & evaluation).sum())

        reference_metric = _metric_pair(evaluator, m3b, target, evaluation, device)
        candidate_metric = _metric_pair(evaluator, f1, target, evaluation, device)
        m3b_psnr.append(reference_metric[0])
        m3b_ssim.append(reference_metric[1])
        f1_psnr.append(candidate_metric[0])
        f1_ssim.append(candidate_metric[1])
        frame_rows.append(
            {
                "frame": name,
                "m3b_psnr": reference_metric[0],
                "f1_psnr": candidate_metric[0],
                "delta_psnr": candidate_metric[0] - reference_metric[0],
                "m3b_ssim": reference_metric[1],
                "f1_ssim": candidate_metric[1],
                "delta_ssim": candidate_metric[1] - reference_metric[1],
                "pixels_replaced": int(fallback.sum()),
                "evaluated_pixels_replaced": int((fallback & evaluation).sum()),
                "m3b_valid_pixels_changed": changed_valid,
            }
        )
        Image.fromarray(f1, mode="RGB").save(args.output_dir / "rgb" / name)
        Image.fromarray(raw_valid.astype(np.uint8) * 255, mode="L").save(
            args.output_dir / "valid_mask" / name
        )
        final_valid = filled_valid | fallback
        Image.fromarray(final_valid.astype(np.uint8) * 255, mode="L").save(
            args.output_dir / "filled_valid_mask" / name
        )
        Image.fromarray(fallback.astype(np.uint8) * 255, mode="L").save(
            args.output_dir / "fallback_mask" / name
        )
        debug = np.concatenate(
            (
                m3b,
                np.repeat(fallback[..., None], 3, axis=2).astype(np.uint8) * 255,
                m2,
                f1,
            ),
            axis=1,
        )
        Image.fromarray(debug, mode="RGB").save(args.output_dir / "debug" / name)

        native_m3b = _rgb(native_m3b_paths[index])
        native_m2 = _rgb(native_m2_paths[index], (native_m3b.shape[1], native_m3b.shape[0]))
        native_fallback = F.interpolate(
            torch.from_numpy(np.ascontiguousarray(fallback))[None, None].to(torch.float32),
            size=native_m3b.shape[:2],
            mode="nearest",
        )[0, 0].bool().numpy()
        native_valid = F.interpolate(
            torch.from_numpy(np.ascontiguousarray(filled_valid))[None, None].to(torch.float32),
            size=native_m3b.shape[:2],
            mode="nearest",
        )[0, 0].bool().numpy()
        if bool((native_fallback & native_valid).any()):
            raise AssertionError(f"Native fallback overlaps M3B validity at {name}")
        native_f1 = native_m3b.copy()
        native_f1[native_fallback] = native_m2[native_fallback]
        if bool(np.any(native_f1[native_valid] != native_m3b[native_valid])):
            raise AssertionError(f"Native M3B-valid pixels changed at {name}")
        Image.fromarray(native_f1, mode="RGB").save(args.output_dir / "renders" / name)

    psnr_delta = np.asarray(f1_psnr) - np.asarray(m3b_psnr)
    ssim_delta = np.asarray(f1_ssim) - np.asarray(m3b_ssim)
    report = {
        "sequence": args.sequence,
        "protocol": "corrected_geometry",
        "variant": (
            "F1_strict_holes"
            if args.boundary_exclusion_pixels == 0
            else f"F2_boundary_exclusion_{args.boundary_exclusion_pixels}px"
        ),
        "boundary_exclusion_pixels": args.boundary_exclusion_pixels,
        "diagnostic": diagnostic,
        "F1_created": True,
        "m3b_overall_psnr": float(np.mean(m3b_psnr)),
        "f1_overall_psnr": float(np.mean(f1_psnr)),
        "delta_psnr": float(np.mean(f1_psnr) - np.mean(m3b_psnr)),
        "m3b_overall_ssim": float(np.mean(m3b_ssim)),
        "f1_overall_ssim": float(np.mean(f1_ssim)),
        "delta_ssim": float(np.mean(f1_ssim) - np.mean(m3b_ssim)),
        "pixels_replaced": replaced_total,
        "replacement_percentage": 100.0 * replaced_total / (len(m3b_paths) * height * width),
        "evaluated_pixels_replaced": evaluated_replaced,
        "m3b_valid_pixels_changed": valid_changed_total,
        "psnr_frame_wins": int((psnr_delta > 1.0e-7).sum()),
        "psnr_frame_losses": int((psnr_delta < -1.0e-7).sum()),
        "psnr_frame_ties": int((np.abs(psnr_delta) <= 1.0e-7).sum()),
        "ssim_frame_wins": int((ssim_delta > 1.0e-7).sum()),
        "ssim_frame_losses": int((ssim_delta < -1.0e-7).sum()),
        "ssim_frame_ties": int((np.abs(ssim_delta) <= 1.0e-7).sum()),
        "worst_frame_delta_psnr": float(psnr_delta.min()),
        "best_frame_delta_psnr": float(psnr_delta.max()),
        "decision": (
            "KEEP"
            if float(np.mean(f1_psnr) - np.mean(m3b_psnr)) >= 0.03
            else "STOP"
        ),
        "frames": frame_rows,
    }
    if valid_changed_total != 0:
        raise AssertionError("Sequence invariant failed: M3B-valid pixels changed")
    (args.output_dir / "fallback_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    output_summary = {
        **m3b_summary,
        "method": (
            "F1_M3B_plus_M2_strict_holes"
            if args.boundary_exclusion_pixels == 0
            else f"F2_M3B_plus_M2_holes_boundary{args.boundary_exclusion_pixels}"
        ),
        "frozen_m3b_reference": str(args.m3b_dir.resolve()),
        "frozen_m2_reference": str(args.m2_dir.resolve()),
        "critical_invariant": "F1 equals M3B exactly wherever M3B filled_valid_mask is true",
        "m3b_valid_pixels_changed": 0,
        "configuration": {
            "fallback": "Method2_B1_metric_l1_w5e-5",
            "fallback_mask": (
                "not M3B radius3 filled_valid_mask"
                if args.boundary_exclusion_pixels == 0
                else (
                    "not M3B radius3 filled_valid_mask AND Euclidean distance-to-valid "
                    f"> {args.boundary_exclusion_pixels} pixels"
                )
            ),
            "alpha_blending": False,
            "boundary_exclusion_pixels": args.boundary_exclusion_pixels,
        },
    }
    (args.output_dir / "render_summary.json").write_text(
        json.dumps(output_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
