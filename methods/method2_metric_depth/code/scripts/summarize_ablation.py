#!/usr/bin/env python3
"""Summarize paired Method 2 results using the fixed sequence protocol."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=project_root / "results")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=project_root / "configs" / "method2_sequence_split.txt",
    )
    parser.add_argument("--baseline", default="B0_baseline")
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--experiments", nargs="*", default=None)
    return parser.parse_args()


def read_sequences(path: Path, include_holdout: bool) -> list[str]:
    sequences: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        role, _gpu_id, sequence = line.split()
        if role == "development" or include_holdout:
            sequences.append(sequence)
    return sequences


def read_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        raise ValueError(f"Empty metric payload: {path}")

    def iteration_key(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    method = max(payload, key=iteration_key)
    values = payload[method]
    return {
        "psnr": float(values["PSNR"]),
        "ssim": float(values["SSIM"]),
        "lpips": float(values.get("LPIPS", float("nan"))),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    sequences = read_sequences(args.split_file, args.include_holdout)
    if not sequences:
        raise SystemExit("No sequences selected by the split file")

    if args.experiments:
        experiments = list(dict.fromkeys([args.baseline, *args.experiments]))
    else:
        experiments = [args.baseline]
        experiments.extend(
            sorted(
                path.name
                for path in results_root.iterdir()
                if path.is_dir() and path.name != args.baseline
            )
        )

    baseline_by_sequence: dict[str, dict[str, float]] = {}
    for sequence in sequences:
        path = results_root / args.baseline / sequence / "results.json"
        if path.is_file():
            baseline_by_sequence[sequence] = read_metrics(path)

    if len(baseline_by_sequence) != len(sequences):
        missing = sorted(set(sequences) - set(baseline_by_sequence))
        raise SystemExit(f"Missing paired B0 result(s): {missing}")

    rows: list[dict[str, object]] = []
    for experiment in experiments:
        for sequence in sequences:
            path = results_root / experiment / sequence / "results.json"
            if not path.is_file():
                continue
            values = read_metrics(path)
            baseline = baseline_by_sequence[sequence]
            rows.append(
                {
                    "experiment": experiment,
                    "sequence": sequence,
                    **values,
                    "delta_psnr": values["psnr"] - baseline["psnr"],
                    "delta_ssim": values["ssim"] - baseline["ssim"],
                    "delta_lpips": values["lpips"] - baseline["lpips"],
                }
            )

    per_frame_path = results_root / "ablation_results.csv"
    with per_frame_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "experiment",
            "sequence",
            "psnr",
            "ssim",
            "lpips",
            "delta_psnr",
            "delta_ssim",
            "delta_lpips",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summaries: list[dict[str, object]] = []
    for experiment in experiments:
        experiment_rows = [row for row in rows if row["experiment"] == experiment]
        if not experiment_rows:
            continue
        psnr = [float(row["psnr"]) for row in experiment_rows]
        ssim = [float(row["ssim"]) for row in experiment_rows]
        delta_psnr = [float(row["delta_psnr"]) for row in experiment_rows]
        delta_ssim = [float(row["delta_ssim"]) for row in experiment_rows]
        summaries.append(
            {
                "experiment": experiment,
                "completed_sequences": len(experiment_rows),
                "expected_sequences": len(sequences),
                "mean_psnr": mean(psnr),
                "median_psnr": statistics.median(psnr),
                "mean_ssim": mean(ssim),
                "median_ssim": statistics.median(ssim),
                "mean_delta_psnr": mean(delta_psnr),
                "median_delta_psnr": statistics.median(delta_psnr),
                "std_delta_psnr": statistics.pstdev(delta_psnr),
                "mean_delta_ssim": mean(delta_ssim),
                "median_delta_ssim": statistics.median(delta_ssim),
                "psnr_wins": sum(value > 0 for value in delta_psnr),
                "psnr_degradations": sum(value < 0 for value in delta_psnr),
                "best_delta_psnr": max(delta_psnr),
                "worst_delta_psnr": min(delta_psnr),
            }
        )

    summaries.sort(
        key=lambda row: (float(row["mean_psnr"]), float(row["mean_ssim"])),
        reverse=True,
    )
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank

    summary_csv_path = results_root / "ablation_summary.csv"
    if summaries:
        fieldnames = [
            "rank",
            "experiment",
            "completed_sequences",
            "expected_sequences",
            "mean_psnr",
            "median_psnr",
            "mean_ssim",
            "median_ssim",
            "mean_delta_psnr",
            "median_delta_psnr",
            "std_delta_psnr",
            "mean_delta_ssim",
            "median_delta_ssim",
            "psnr_wins",
            "psnr_degradations",
            "best_delta_psnr",
            "worst_delta_psnr",
        ]
        with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

    summary_json_path = results_root / "ablation_summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "primary_metric": "mean_psnr",
                "secondary_metric": "mean_ssim",
                "sequences": sequences,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {per_frame_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_json_path}")
    if summaries:
        print("\nRanking (mean PSNR primary, mean SSIM secondary):")
        for row in summaries:
            print(
                f"{row['rank']:>2}. {row['experiment']:<34} "
                f"PSNR={row['mean_psnr']:.6f} "
                f"dPSNR={row['mean_delta_psnr']:+.6f} "
                f"SSIM={row['mean_ssim']:.6f} "
                f"dSSIM={row['mean_delta_ssim']:+.6f} "
                f"n={row['completed_sequences']}/{row['expected_sequences']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
