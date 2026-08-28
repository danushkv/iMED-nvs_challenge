#!/usr/bin/env python3
"""Small staged Phase-11 ablation for one public iMED-NVS sequence.

Rendering is source-only. Target RGB is opened only by the already-separated
``evaluate.py`` subprocess after each fixed prediction set has been written.
Selections are global configuration choices for the sequence, never per-frame
adaptation. A one-sequence result is explicitly labelled preliminary.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence


@dataclass(frozen=True)
class SweepConfig:
    renderer: str
    mask_source_tools: bool
    hole_fill: str = "none"
    fill_radius: int = 1
    depth_filter: str = "none"
    introduced_stage: str = "A_renderer_mask"

    @property
    def key(self) -> tuple:
        return (
            self.renderer,
            self.mask_source_tools,
            self.hole_fill,
            self.fill_radius,
            self.depth_filter,
        )

    @property
    def config_id(self) -> str:
        mask = "mask_on" if self.mask_source_tools else "mask_off"
        if self.hole_fill == "none":
            hole = "h0"
        elif self.hole_fill == "morphological":
            hole = f"h1_morph_r{self.fill_radius}"
        else:
            hole = f"h2_nearest_r{self.fill_radius}"
        return f"{self.renderer}__{mask}__{hole}__depth_{self.depth_filter}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged Method-1 Phase-11 sweep")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("adapted_baseline", "corrected_geometry"),
        default="adapted_baseline",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite_partial",
        action="store_true",
        help="Resume an incomplete run by replacing matching outputs; nothing is deleted",
    )
    return parser.parse_args()


def _run(command: Sequence[str], cwd: Path) -> None:
    print("[SWEEP RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def _load_record(config: SweepConfig, run_dir: Path, protocol: str) -> dict:
    results_path = run_dir / "evaluation" / "results.json"
    render_summary_path = run_dir / "render_summary.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    render_summary = json.loads(render_summary_path.read_text(encoding="utf-8"))
    metrics = results["protocols"][protocol]
    return {
        "config_id": config.config_id,
        **asdict(config),
        "mean_psnr": float(metrics["mean_psnr"]),
        "median_psnr": float(metrics["median_psnr"]),
        "mean_ssim": float(metrics["mean_ssim"]),
        "median_ssim": float(metrics["median_ssim"]),
        "mean_coverage_percent": float(metrics["mean_coverage_percent"]),
        "mean_raw_coverage_percent": float(metrics["mean_raw_coverage_percent"]),
        "total_sequence_runtime_seconds": float(
            render_summary["total_sequence_runtime_seconds"]
        ),
        "mean_compute_time_per_frame_seconds": float(
            render_summary["mean_compute_time_per_frame_seconds"]
        ),
        "peak_gpu_memory_mb": float(render_summary["peak_gpu_memory_mb"]),
        "prediction_dir": str(run_dir),
        "results_json": str(results_path),
    }


def _run_config(
    config: SweepConfig,
    args: argparse.Namespace,
    method_root: Path,
) -> dict:
    run_dir = args.output_root / "runs" / config.config_id
    results_path = run_dir / "evaluation" / "results.json"
    render_summary_path = run_dir / "render_summary.json"
    if results_path.is_file() and render_summary_path.is_file():
        print(f"[SWEEP REUSE] {config.config_id}", flush=True)
        return _load_record(config, run_dir, args.protocol)

    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite_partial:
        raise FileExistsError(
            f"Incomplete run exists at {run_dir}. Inspect it, then use --overwrite_partial "
            "to replace matching files without deleting anything."
        )

    render_command = [
        sys.executable,
        "render_sequence.py",
        "--data_root",
        str(args.data_root),
        "--sequence",
        args.sequence,
        "--output_dir",
        str(run_dir),
        "--renderer",
        config.renderer,
        "--hole_fill",
        config.hole_fill,
        "--fill_radius",
        str(config.fill_radius),
        "--depth_filter",
        config.depth_filter,
        "--device",
        args.device,
        "--no_debug",
        "--no_correspondence",
        "--minimal_outputs",
    ]
    if config.mask_source_tools:
        render_command.append("--mask_source_tools")
    if args.overwrite_partial:
        render_command.append("--overwrite")
    _run(render_command, method_root)

    evaluation_dir = run_dir / "evaluation"
    evaluate_command = [
        sys.executable,
        "evaluate.py",
        "--data_root",
        str(args.data_root),
        "--sequence",
        args.sequence,
        "--prediction_dir",
        str(run_dir),
        "--evaluation_dir",
        str(evaluation_dir),
        "--mask_protocol",
        args.protocol,
        "--device",
        args.device,
    ]
    if args.overwrite_partial:
        evaluate_command.append("--overwrite")
    _run(evaluate_command, method_root)
    return _load_record(config, run_dir, args.protocol)


def _best(configs: Iterable[SweepConfig], records: Dict[tuple, dict]) -> SweepConfig:
    """Apply the challenge ranking: mean PSNR primary, mean SSIM secondary."""

    return max(
        configs,
        key=lambda config: (
            float(records[config.key]["mean_psnr"]),
            float(records[config.key]["mean_ssim"]),
        ),
    )


def _write_outputs(
    args: argparse.Namespace,
    records: Dict[tuple, dict],
    best_a: SweepConfig,
    best_b: SweepConfig,
    best_final: SweepConfig,
) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        records.values(),
        key=lambda row: (-float(row["mean_psnr"]), -float(row["mean_ssim"])),
    )
    for rank, row in enumerate(ranked, start=1):
        key = (
            row["renderer"],
            row["mask_source_tools"],
            row["hole_fill"],
            row["fill_radius"],
            row["depth_filter"],
        )
        row["rank"] = rank
        row["selected_stage_a"] = key == best_a.key
        row["selected_stage_b"] = key == best_b.key
        row["selected_final"] = key == best_final.key

    csv_fields = (
        "rank",
        "config_id",
        "introduced_stage",
        "renderer",
        "mask_source_tools",
        "hole_fill",
        "fill_radius",
        "depth_filter",
        "mean_psnr",
        "median_psnr",
        "mean_ssim",
        "median_ssim",
        "mean_coverage_percent",
        "mean_raw_coverage_percent",
        "total_sequence_runtime_seconds",
        "mean_compute_time_per_frame_seconds",
        "peak_gpu_memory_mb",
        "selected_stage_a",
        "selected_stage_b",
        "selected_final",
        "prediction_dir",
        "results_json",
    )
    with (args.output_root / "ablation_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({field: row[field] for field in csv_fields})

    payload = {
        "phase": 11,
        "sequence": args.sequence,
        "protocol": args.protocol,
        "single_sequence_preliminary": True,
        "ranking": "mean PSNR primary, mean SSIM secondary",
        "rendering_uses_target_rgb": False,
        "target_rgb_access": "offline evaluate.py subprocess only after each fixed render",
        "per_frame_parameter_selection": False,
        "staged_design": {
            "stage_a": "renderer x complete source-tool-mask stream, H0, unfiltered depth",
            "stage_b": "H0, H1 morphological r1, H2 nearest r1/r2/r3 with stage-A geometry fixed",
            "stage_c": "none, median 3x3, bilateral 5x5 with stage-B geometry/fill fixed",
            "bilateral_fixed_parameters": {
                "kernel_size": 5,
                "sigma_spatial_pixels": 2.0,
                "sigma_depth_mm": 3.0,
            },
        },
        "selected_stage_a": best_a.config_id,
        "selected_stage_b": best_b.config_id,
        "selected_final": best_final.config_id,
        "configuration_count": len(ranked),
        "results": ranked,
    }
    (args.output_root / "ablation_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "configuration_count": len(ranked),
                "selected_stage_a": best_a.config_id,
                "selected_stage_b": best_b.config_id,
                "selected_final": best_final.config_id,
                "mean_psnr": records[best_final.key]["mean_psnr"],
                "mean_ssim": records[best_final.key]["mean_ssim"],
                "ablation_csv": str(args.output_root / "ablation_results.csv"),
            },
            indent=2,
        )
    )


def main() -> None:
    args = _parse_args()
    method_root = Path(__file__).resolve().parent
    args.output_root = args.output_root.resolve()
    records: Dict[tuple, dict] = {}

    stage_a = [
        SweepConfig(renderer=renderer, mask_source_tools=mask)
        for renderer in ("nearest", "bilinear", "soft_depth")
        for mask in (False, True)
    ]
    for config in stage_a:
        records[config.key] = _run_config(config, args, method_root)
    best_a = _best(stage_a, records)
    print(f"[SWEEP SELECT A] {best_a.config_id}", flush=True)

    stage_b = [
        best_a,
        SweepConfig(
            best_a.renderer,
            best_a.mask_source_tools,
            hole_fill="morphological",
            fill_radius=1,
            introduced_stage="B_holes",
        ),
        *[
            SweepConfig(
                best_a.renderer,
                best_a.mask_source_tools,
                hole_fill="nearest",
                fill_radius=radius,
                introduced_stage="B_holes",
            )
            for radius in (1, 2, 3)
        ],
    ]
    for config in stage_b:
        if config.key not in records:
            records[config.key] = _run_config(config, args, method_root)
    best_b = _best(stage_b, records)
    print(f"[SWEEP SELECT B] {best_b.config_id}", flush=True)

    stage_c = [
        best_b,
        *[
            SweepConfig(
                best_b.renderer,
                best_b.mask_source_tools,
                hole_fill=best_b.hole_fill,
                fill_radius=best_b.fill_radius,
                depth_filter=depth_filter,
                introduced_stage="C_depth",
            )
            for depth_filter in ("median3", "bilateral")
        ],
    ]
    for config in stage_c:
        if config.key not in records:
            records[config.key] = _run_config(config, args, method_root)
    best_final = _best(stage_c, records)
    print(f"[SWEEP SELECT C] {best_final.config_id}", flush=True)

    _write_outputs(args, records, best_a, best_b, best_final)


if __name__ == "__main__":
    main()
