#!/usr/bin/env python3
"""Aggregate frozen F1 and the two permitted F2 variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEQUENCES = (
    "session_004_scene_2_tool_1",
    "session_004_scene_6_tool_2",
    "session_005_scene_7_tool_2",
    "session_007_scene_11_tool_3",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Method-8 F1/F2")
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/summary"))
    return parser.parse_args()


def main() -> None:
    args = _args()
    variants = {
        "F1_strict_holes": args.results_root / "F1_M2_holes",
        "F2_boundary1": args.results_root / "F2_boundary1",
        # Preserve the interrupted CUDA attempts and aggregate the complete
        # CPU run from its separate directory.
        "F2_boundary2": args.results_root / "F2_boundary2_cpu",
    }
    rows = []
    summaries = {}
    for variant, root in variants.items():
        reports = []
        for sequence in SEQUENCES:
            path = root / sequence / "fallback_report.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed report: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            if int(report["m3b_valid_pixels_changed"]) != 0:
                raise AssertionError(f"Protected-pixel invariant failed: {path}")
            reports.append(report)
            rows.append(
                {
                    "sequence": sequence,
                    "method": variant,
                    "PSNR": report["f1_overall_psnr"],
                    "delta_PSNR": report["delta_psnr"],
                    "SSIM": report["f1_overall_ssim"],
                    "delta_SSIM": report["delta_ssim"],
                    "pixels_replaced": report["pixels_replaced"],
                    "evaluated_pixels_replaced": report["evaluated_pixels_replaced"],
                    "PSNR_wins": report["psnr_frame_wins"],
                    "PSNR_losses": report["psnr_frame_losses"],
                    "SSIM_wins": report["ssim_frame_wins"],
                    "SSIM_losses": report["ssim_frame_losses"],
                    "valid_m3b_pixels_changed": report["m3b_valid_pixels_changed"],
                }
            )
        summaries[variant] = {
            "mean_psnr": float(np.mean([r["f1_overall_psnr"] for r in reports])),
            "mean_delta_psnr": float(np.mean([r["delta_psnr"] for r in reports])),
            "mean_ssim": float(np.mean([r["f1_overall_ssim"] for r in reports])),
            "mean_delta_ssim": float(np.mean([r["delta_ssim"] for r in reports])),
            "sequence_psnr_wins": int(sum(r["delta_psnr"] > 0 for r in reports)),
            "sequence_psnr_losses": int(sum(r["delta_psnr"] < 0 for r in reports)),
            "frame_psnr_wins": int(sum(r["psnr_frame_wins"] for r in reports)),
            "frame_psnr_losses": int(sum(r["psnr_frame_losses"] for r in reports)),
            "frame_ssim_wins": int(sum(r["ssim_frame_wins"] for r in reports)),
            "frame_ssim_losses": int(sum(r["ssim_frame_losses"] for r in reports)),
            "valid_m3b_pixels_changed": int(
                sum(r["m3b_valid_pixels_changed"] for r in reports)
            ),
        }
    best = max(
        summaries,
        key=lambda name: (summaries[name]["mean_psnr"], summaries[name]["mean_ssim"]),
    )
    result = {
        "primary_metric": "mean PSNR across the fixed four-sequence development set",
        "secondary_metric": "mean SSIM",
        "summaries": summaries,
        "recommended_variant": best,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "method8_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "method8_ablation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
