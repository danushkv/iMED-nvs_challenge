#!/usr/bin/env python3
"""Compare frozen 50k M4-A and controlled M4-B from saved metrics only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


M4_ROOT = Path(__file__).resolve().parent
SEQUENCE = "session_004_scene_2_tool_1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare M4-A and M4-B metrics")
    parser.add_argument(
        "--m4a-root",
        type=Path,
        default=M4_ROOT / "outputs_m4a_scale_fixed" / SEQUENCE,
    )
    parser.add_argument(
        "--m4b-root",
        type=Path,
        default=M4_ROOT / "outputs_m4b_surface" / SEQUENCE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=M4_ROOT / "outputs_m4b_surface" / "m4a_vs_m4b.json",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> dict[str, tuple[float, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["frame"]] = (float(row["psnr"]), float(row["ssim"]))
    return rows


def _wins(
    baseline: dict[str, tuple[float, float]],
    candidate: dict[str, tuple[float, float]],
    metric_index: int,
) -> dict[str, int]:
    if baseline.keys() != candidate.keys():
        raise ValueError("M4-A/M4-B per-frame CSV keys do not match")
    wins = losses = ties = 0
    for frame in baseline:
        delta = candidate[frame][metric_index] - baseline[frame][metric_index]
        if delta > 1e-12:
            wins += 1
        elif delta < -1e-12:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def main() -> None:
    args = _args()
    m4a_train = _load_json(args.m4a_root / "train_summary.json")
    m4b_train = _load_json(args.m4b_root / "train_summary.json")
    m4a_source = _load_json(
        args.m4a_root / "source_reconstruction" / "source_metrics.json"
    )
    m4b_source = _load_json(
        args.m4b_root / "source_reconstruction" / "source_metrics.json"
    )
    m4a_eval = _load_json(args.m4a_root / "target_evaluation_both" / "results.json")
    m4b_eval = _load_json(args.m4b_root / "target_evaluation_both" / "results.json")
    if m4a_eval["sequence"] != m4b_eval["sequence"]:
        raise ValueError("M4-A/M4-B sequence mismatch")

    protocols: dict[str, Any] = {}
    for protocol in ("corrected_geometry", "adapted_baseline"):
        m4a_metrics = m4a_eval["protocols"][protocol]
        m4b_metrics = m4b_eval["protocols"][protocol]
        m4a_frames = _load_csv(
            args.m4a_root
            / "target_evaluation_both"
            / f"per_frame_{protocol}.csv"
        )
        m4b_frames = _load_csv(
            args.m4b_root
            / "target_evaluation_both"
            / f"per_frame_{protocol}.csv"
        )
        protocols[protocol] = {
            "m4a": {
                "psnr": m4a_metrics["mean_psnr"],
                "ssim": m4a_metrics["mean_ssim"],
            },
            "m4b": {
                "psnr": m4b_metrics["mean_psnr"],
                "ssim": m4b_metrics["mean_ssim"],
            },
            "delta": {
                "psnr": m4b_metrics["mean_psnr"] - m4a_metrics["mean_psnr"],
                "ssim": m4b_metrics["mean_ssim"] - m4a_metrics["mean_ssim"],
            },
            "per_frame_psnr": _wins(m4a_frames, m4b_frames, 0),
            "per_frame_ssim": _wins(m4a_frames, m4b_frames, 1),
        }

    report = {
        "method": "M4-B controlled comparison",
        "sequence": m4a_eval["sequence"],
        "controlled_difference": "Gaussian initialization only",
        "standard_m4a": {
            "initial_gaussians": m4a_train["initial_gaussians"],
            "final_gaussians": m4a_train["final_gaussians"],
            "training_runtime_seconds": m4a_train["training_runtime_seconds"],
            "peak_vram_bytes": m4a_train["peak_vram_bytes"],
            "source_psnr": m4a_source["psnr"],
            "source_ssim": m4a_source["ssim"],
            "source_depth_mae": m4a_source["depth_mae"],
        },
        "surface_m4b": {
            "initial_gaussians": m4b_train["initial_gaussians"],
            "final_gaussians": m4b_train["final_gaussians"],
            "training_runtime_seconds": m4b_train["training_runtime_seconds"],
            "peak_vram_bytes": m4b_train["peak_vram_bytes"],
            "source_psnr": m4b_source["psnr"],
            "source_ssim": m4b_source["ssim"],
            "source_depth_mae": m4b_source["depth_mae"],
            "initialization_statistics": m4b_train["initial_statistics"],
        },
        "source_delta": {
            "psnr": m4b_source["psnr"] - m4a_source["psnr"],
            "ssim": m4b_source["ssim"] - m4a_source["ssim"],
            "depth_mae": m4b_source["depth_mae"] - m4a_source["depth_mae"],
        },
        "runtime_delta_seconds": (
            m4b_train["training_runtime_seconds"]
            - m4a_train["training_runtime_seconds"]
        ),
        "peak_vram_delta_bytes": (
            m4b_train["peak_vram_bytes"] - m4a_train["peak_vram_bytes"]
        ),
        "protocols": protocols,
        "target_rgb_access": "offline evaluation only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    adapted = protocols["adapted_baseline"]
    markdown = f"""# Method 4B controlled result

Sequence: `{report['sequence']}`

| Metric | M4-A | M4-B | Delta |
| --- | ---: | ---: | ---: |
| Target PSNR (adapted) | {adapted['m4a']['psnr']:.6f} | {adapted['m4b']['psnr']:.6f} | {adapted['delta']['psnr']:+.6f} |
| Target SSIM (adapted) | {adapted['m4a']['ssim']:.6f} | {adapted['m4b']['ssim']:.6f} | {adapted['delta']['ssim']:+.6f} |
| Source PSNR | {m4a_source['psnr']:.6f} | {m4b_source['psnr']:.6f} | {report['source_delta']['psnr']:+.6f} |
| Source SSIM | {m4a_source['ssim']:.6f} | {m4b_source['ssim']:.6f} | {report['source_delta']['ssim']:+.6f} |
| Initial Gaussians | {m4a_train['initial_gaussians']} | {m4b_train['initial_gaussians']} | {m4b_train['initial_gaussians'] - m4a_train['initial_gaussians']:+d} |
| Final Gaussians | {m4a_train['final_gaussians']} | {m4b_train['final_gaussians']} | {m4b_train['final_gaussians'] - m4a_train['final_gaussians']:+d} |
| Training runtime (s) | {m4a_train['training_runtime_seconds']:.2f} | {m4b_train['training_runtime_seconds']:.2f} | {report['runtime_delta_seconds']:+.2f} |

Adapted PSNR wins/losses/ties: {adapted['per_frame_psnr']['wins']} / {adapted['per_frame_psnr']['losses']} / {adapted['per_frame_psnr']['ties']}

Adapted SSIM wins/losses/ties: {adapted['per_frame_ssim']['wins']} / {adapted['per_frame_ssim']['losses']} / {adapted['per_frame_ssim']['ties']}
"""
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
