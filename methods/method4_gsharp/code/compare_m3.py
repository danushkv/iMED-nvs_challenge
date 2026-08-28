#!/usr/bin/env python3
"""Compare saved M4-A metrics with the read-only Method 3 ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


M4_ROOT = Path(__file__).resolve().parent
SEQUENCES = (
    "session_004_scene_2_tool_1",
    "session_004_scene_6_tool_2",
    "session_005_scene_7_tool_2",
    "session_007_scene_11_tool_3",
)
PROTOCOLS = ("corrected_geometry", "adapted_baseline")
M3_METHODS = {
    "M3-A": "M3_A_MV1A",
    "M3-B": "M3_B_surface",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only M4-A versus M3 comparison")
    parser.add_argument(
        "--m4-root",
        type=Path,
        default=M4_ROOT / "outputs_m4a_scale_fixed",
    )
    parser.add_argument(
        "--m3-root",
        type=Path,
        default=Path("/mnt/cluster/workspaces/venkateda/method3_surface_fusion"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=M4_ROOT / "outputs_m4a_scale_fixed" / "m4a_vs_m3.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=M4_ROOT / "outputs_m4a_scale_fixed" / "m4a_vs_m3.md",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frame = row["frame"]
            if frame in rows:
                raise ValueError(f"Duplicate frame {frame} in {path}")
            rows[frame] = {
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
            }
    if len(rows) != 199:
        raise ValueError(f"Expected 199 frames in {path}, found {len(rows)}")
    return rows


def _m4_path(root: Path, sequence: str, protocol: str) -> Path:
    return root / sequence / "target_evaluation_both" / f"per_frame_{protocol}.csv"


def _m3_path(root: Path, method_dir: str, sequence: str, protocol: str) -> Path:
    return (
        root
        / "outputs"
        / method_dir
        / sequence
        / "evaluation_full"
        / f"per_frame_{protocol}.csv"
    )


def _method_summary(rows_by_sequence: dict[str, dict[str, dict[str, float]]]) -> dict:
    all_rows = [row for rows in rows_by_sequence.values() for row in rows.values()]
    return {
        "frames": len(all_rows),
        "mean_psnr": mean(row["psnr"] for row in all_rows),
        "mean_ssim": mean(row["ssim"] for row in all_rows),
        "sequences": {
            sequence: {
                "frames": len(rows),
                "mean_psnr": mean(row["psnr"] for row in rows.values()),
                "mean_ssim": mean(row["ssim"] for row in rows.values()),
            }
            for sequence, rows in rows_by_sequence.items()
        },
    }


def _comparison(
    m4_rows: dict[str, dict[str, dict[str, float]]],
    baseline_rows: dict[str, dict[str, dict[str, float]]],
) -> dict:
    counts = {
        "psnr_wins": 0,
        "psnr_losses": 0,
        "psnr_ties": 0,
        "ssim_wins": 0,
        "ssim_losses": 0,
        "ssim_ties": 0,
    }
    sequence_rows: dict[str, dict] = {}
    all_psnr_deltas: list[float] = []
    all_ssim_deltas: list[float] = []
    for sequence in SEQUENCES:
        if set(m4_rows[sequence]) != set(baseline_rows[sequence]):
            raise ValueError(f"Frame mismatch for {sequence}")
        psnr_deltas: list[float] = []
        ssim_deltas: list[float] = []
        local = {key: 0 for key in counts}
        for frame in sorted(m4_rows[sequence]):
            psnr_delta = m4_rows[sequence][frame]["psnr"] - baseline_rows[sequence][frame]["psnr"]
            ssim_delta = m4_rows[sequence][frame]["ssim"] - baseline_rows[sequence][frame]["ssim"]
            psnr_deltas.append(psnr_delta)
            ssim_deltas.append(ssim_delta)
            psnr_key = "psnr_wins" if psnr_delta > 0 else "psnr_losses" if psnr_delta < 0 else "psnr_ties"
            ssim_key = "ssim_wins" if ssim_delta > 0 else "ssim_losses" if ssim_delta < 0 else "ssim_ties"
            local[psnr_key] += 1
            local[ssim_key] += 1
            counts[psnr_key] += 1
            counts[ssim_key] += 1
        all_psnr_deltas.extend(psnr_deltas)
        all_ssim_deltas.extend(ssim_deltas)
        sequence_rows[sequence] = {
            "mean_delta_psnr": mean(psnr_deltas),
            "mean_delta_ssim": mean(ssim_deltas),
            **local,
        }
    return {
        "mean_delta_psnr": mean(all_psnr_deltas),
        "mean_delta_ssim": mean(all_ssim_deltas),
        **counts,
        "sequences": sequence_rows,
    }


def _markdown(report: dict) -> str:
    lines = ["# M4-A versus read-only Method 3", ""]
    for protocol in PROTOCOLS:
        data = report["protocols"][protocol]
        lines.extend(
            [
                f"## {protocol}",
                "",
                "| Method | PSNR | SSIM |",
                "|---|---:|---:|",
            ]
        )
        for method in ("M3-A", "M3-B", "M4-A"):
            summary = data["methods"][method]
            lines.append(f"| {method} | {summary['mean_psnr']:.5f} | {summary['mean_ssim']:.5f} |")
        lines.extend(["", "| Comparison | Delta PSNR | Delta SSIM | PSNR W/L/T | SSIM W/L/T |", "|---|---:|---:|---:|---:|"])
        for baseline in ("M3-A", "M3-B"):
            comparison = data["comparisons"][f"M4-A_vs_{baseline}"]
            lines.append(
                f"| M4-A vs {baseline} | {comparison['mean_delta_psnr']:+.5f} | "
                f"{comparison['mean_delta_ssim']:+.5f} | "
                f"{comparison['psnr_wins']}/{comparison['psnr_losses']}/{comparison['psnr_ties']} | "
                f"{comparison['ssim_wins']}/{comparison['ssim_losses']}/{comparison['ssim_ties']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _args()
    report = {
        "status": "complete",
        "sequences": list(SEQUENCES),
        "frame_count": 796,
        "method3_root": str(args.m3_root),
        "method3_access": "read-only metrics",
        "target_rgb_access": False,
        "protocols": {},
    }
    for protocol in PROTOCOLS:
        m4_rows = {
            sequence: _read_csv(_m4_path(args.m4_root, sequence, protocol))
            for sequence in SEQUENCES
        }
        baseline_sets = {
            label: {
                sequence: _read_csv(_m3_path(args.m3_root, method_dir, sequence, protocol))
                for sequence in SEQUENCES
            }
            for label, method_dir in M3_METHODS.items()
        }
        report["protocols"][protocol] = {
            "methods": {
                "M3-A": _method_summary(baseline_sets["M3-A"]),
                "M3-B": _method_summary(baseline_sets["M3-B"]),
                "M4-A": _method_summary(m4_rows),
            },
            "comparisons": {
                f"M4-A_vs_{label}": _comparison(m4_rows, rows)
                for label, rows in baseline_sets.items()
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report), end="")


if __name__ == "__main__":
    main()
