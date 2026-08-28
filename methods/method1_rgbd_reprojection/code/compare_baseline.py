#!/usr/bin/env python3
"""Compare saved Method-1 metrics against saved Endo-4DGS metrics.

This program is deliberately read-only with respect to both methods. It never
renders, trains, or recomputes a metric. New files are written only beneath the
requested comparison output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROTOCOLS = ("corrected_geometry", "adapted_baseline")
SOURCE_PATH_RE = re.compile(r"source_path=(?:'([^']+)'|\"([^\"]+)\")")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare existing Endo-4DGS and RGB-D reprojection results"
    )
    parser.add_argument(
        "--baseline_root",
        type=Path,
        required=True,
        help="Endo-4DGS results.json, one model directory, or a root containing sequence models",
    )
    parser.add_argument(
        "--rgbd_root",
        type=Path,
        required=True,
        help="Method-1 results.json, one prediction directory, or a root containing predictions",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        required=True,
        help="The named Method-1 evaluation protocol to compare",
    )
    parser.add_argument(
        "--baseline_protocol",
        choices=PROTOCOLS,
        required=True,
        help="Explicit declaration of the mask protocol used to produce the existing baseline metrics",
    )
    parser.add_argument(
        "--baseline_method",
        help="Exact Endo-4DGS results.json method key, e.g. ours_14000; required if a file is ambiguous",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        dest="sequences",
        help="Compare only this sequence; repeat for multiple sequences",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Compare only the intersection when the two roots contain different sequence sets",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json_files(root: Path, method1: bool) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(root.rglob("results.json"))
    if method1:
        files = [path for path in files if path.parent.name == "evaluation"]
        direct = root / "evaluation" / "results.json"
        if direct.exists() and direct not in files:
            files.insert(0, direct)
    if not files:
        kind = "Method-1 evaluation" if method1 else "Endo-4DGS"
        raise FileNotFoundError(f"no {kind} results.json found under {root}")
    return files


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sequence_from_cfg(result_path: Path) -> Optional[str]:
    for directory in (result_path.parent, *result_path.parents):
        cfg_path = directory / "cfg_args"
        if not cfg_path.is_file():
            continue
        match = SOURCE_PATH_RE.search(cfg_path.read_text(encoding="utf-8"))
        if match is not None:
            source_path = match.group(1) or match.group(2)
            return Path(source_path).name
    return None


def _sequence_from_path(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        if part.startswith("session_"):
            return part
    return None


def _metric_value(record: dict, name: str, context: str) -> float:
    matching_keys = [key for key in record if key.lower() == name.lower()]
    if len(matching_keys) != 1:
        raise ValueError(f"{context} must contain exactly one {name} value")
    value = float(record[matching_keys[0]])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name} in {context}")
    return value


def _metric_candidates(data: dict, prefix: str = "") -> List[Tuple[str, dict]]:
    candidates: List[Tuple[str, dict]] = []
    lower_keys = {str(key).lower() for key in data}
    if "psnr" in lower_keys and "ssim" in lower_keys:
        candidates.append((prefix or "<root>", data))
        return candidates
    for key, value in data.items():
        if isinstance(value, dict):
            label = f"{prefix}.{key}" if prefix else str(key)
            candidates.extend(_metric_candidates(value, label))
    return candidates


def _select_baseline_record(
    data: dict,
    result_path: Path,
    requested_method: Optional[str],
) -> Tuple[str, dict]:
    candidates = _metric_candidates(data)
    if requested_method is not None:
        matches = [
            candidate
            for candidate in candidates
            if candidate[0] == requested_method or candidate[0].split(".")[-1] == requested_method
        ]
        if len(matches) != 1:
            labels = ", ".join(label for label, _ in candidates) or "none"
            raise ValueError(
                f"--baseline_method {requested_method!r} matched {len(matches)} records in {result_path}; "
                f"available: {labels}"
            )
        return matches[0]
    if len(candidates) != 1:
        labels = ", ".join(label for label, _ in candidates) or "none"
        raise ValueError(
            f"expected one metric record in {result_path}, found {len(candidates)} ({labels}); "
            "specify --baseline_method"
        )
    return candidates[0]


def _insert_unique(records: Dict[str, dict], sequence: str, record: dict, kind: str) -> None:
    if sequence in records:
        raise ValueError(
            f"multiple {kind} result files resolve to {sequence!r}; narrow the input root or remove ambiguity"
        )
    records[sequence] = record


def _load_baselines(
    root: Path,
    method: Optional[str],
    requested: Optional[Sequence[str]] = None,
) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    for path in _json_files(root, method1=False):
        data = _read_json(path)
        sequence = data.get("sequence")
        if not isinstance(sequence, str):
            sequence = _sequence_from_cfg(path) or _sequence_from_path(path)
        if sequence is None:
            raise ValueError(f"cannot determine sequence for baseline result: {path}")
        if requested and sequence not in requested:
            continue
        method_name, metrics = _select_baseline_record(data, path, method)
        _insert_unique(
            records,
            sequence,
            {
                "psnr": _metric_value(metrics, "PSNR", str(path)),
                "ssim": _metric_value(metrics, "SSIM", str(path)),
                "method": method_name,
                "result_path": str(path),
            },
            "baseline",
        )
    return records


def _load_rgbd(
    root: Path,
    protocol: str,
    requested: Optional[Sequence[str]] = None,
) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    for path in _json_files(root, method1=True):
        data = _read_json(path)
        sequence = data.get("sequence")
        if not isinstance(sequence, str):
            sequence = _sequence_from_path(path)
        if sequence is None:
            raise ValueError(f"cannot determine sequence for Method-1 result: {path}")
        if requested and sequence not in requested:
            continue
        if data.get("subset_evaluation", False):
            raise ValueError(f"refusing subset evaluation in Phase 9: {path}")
        if data.get("include_first_frame_support", False):
            raise ValueError(f"refusing method-dependent first-frame support in Phase 9: {path}")
        protocols = data.get("protocols")
        if not isinstance(protocols, dict) or protocol not in protocols:
            raise ValueError(f"Method-1 protocol {protocol!r} is absent from {path}")
        metrics = protocols[protocol]
        if not isinstance(metrics, dict):
            raise ValueError(f"invalid Method-1 protocol record in {path}")
        coverage = _metric_value(metrics, "mean_coverage_percent", str(path))
        _insert_unique(
            records,
            sequence,
            {
                "psnr": _metric_value(metrics, "mean_psnr", str(path)),
                "ssim": _metric_value(metrics, "mean_ssim", str(path)),
                "coverage": coverage,
                "result_path": str(path),
            },
            "Method-1",
        )
    return records


def _filter_requested(records: Dict[str, dict], requested: Optional[Sequence[str]], kind: str) -> Dict[str, dict]:
    if not requested:
        return records
    missing = sorted(set(requested) - set(records))
    if missing:
        raise ValueError(f"requested sequence(s) missing from {kind}: {', '.join(missing)}")
    return {sequence: records[sequence] for sequence in requested}


def _comparison_rows(
    baselines: Dict[str, dict],
    rgbd: Dict[str, dict],
    allow_partial: bool,
) -> Tuple[List[dict], List[str], List[str]]:
    baseline_only = sorted(set(baselines) - set(rgbd))
    rgbd_only = sorted(set(rgbd) - set(baselines))
    if (baseline_only or rgbd_only) and not allow_partial:
        raise ValueError(
            "sequence sets differ; baseline-only="
            f"{baseline_only}, RGBD-only={rgbd_only}. Narrow with --sequence or explicitly use --allow_partial."
        )
    common = sorted(set(baselines) & set(rgbd))
    if not common:
        raise ValueError("the baseline and Method-1 inputs have no sequences in common")
    rows = []
    for sequence in common:
        baseline = baselines[sequence]
        method1 = rgbd[sequence]
        ranking_winner, ranking_decided_by = _challenge_order_winner(
            baseline["psnr"], method1["psnr"], baseline["ssim"], method1["ssim"]
        )
        rows.append(
            {
                "sequence": sequence,
                "endo4dgs_psnr": baseline["psnr"],
                "rgbd_psnr": method1["psnr"],
                "delta_psnr": method1["psnr"] - baseline["psnr"],
                "endo4dgs_ssim": baseline["ssim"],
                "rgbd_ssim": method1["ssim"],
                "delta_ssim": method1["ssim"] - baseline["ssim"],
                "coverage_percent": method1["coverage"],
                "challenge_order_winner": ranking_winner,
                "challenge_order_decided_by": ranking_decided_by,
                "baseline_method": baseline["method"],
                "baseline_result": baseline["result_path"],
                "rgbd_result": method1["result_path"],
            }
        )
    return rows, baseline_only, rgbd_only


def _challenge_order_winner(
    endo4dgs_psnr: float,
    rgbd_psnr: float,
    endo4dgs_ssim: float,
    rgbd_ssim: float,
) -> Tuple[str, str]:
    """Rank by mean PSNR first, then mean SSIM only for an exact PSNR tie."""

    if rgbd_psnr > endo4dgs_psnr:
        return "RGBD", "mean_psnr"
    if rgbd_psnr < endo4dgs_psnr:
        return "Endo4DGS", "mean_psnr"
    if rgbd_ssim > endo4dgs_ssim:
        return "RGBD", "mean_ssim_tiebreak"
    if rgbd_ssim < endo4dgs_ssim:
        return "Endo4DGS", "mean_ssim_tiebreak"
    return "tie", "exact_tie"


def _win_counts(values: Iterable[float]) -> dict:
    values = list(values)
    return {
        "rgbd_wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "endo4dgs_wins": sum(value < 0 for value in values),
    }


def _extreme(rows: Sequence[dict], key: str, best: bool) -> dict:
    row = (max if best else min)(rows, key=lambda item: item[key])
    return {"sequence": row["sequence"], key: row[key]}


def _build_summary(
    rows: Sequence[dict],
    protocol: str,
    baseline_protocol: str,
    baseline_only: Sequence[str],
    rgbd_only: Sequence[str],
) -> dict:
    delta_psnr = [row["delta_psnr"] for row in rows]
    delta_ssim = [row["delta_ssim"] for row in rows]
    mean_endo4dgs_psnr = mean(row["endo4dgs_psnr"] for row in rows)
    mean_rgbd_psnr = mean(row["rgbd_psnr"] for row in rows)
    mean_endo4dgs_ssim = mean(row["endo4dgs_ssim"] for row in rows)
    mean_rgbd_ssim = mean(row["rgbd_ssim"] for row in rows)
    overall_winner, overall_decided_by = _challenge_order_winner(
        mean_endo4dgs_psnr,
        mean_rgbd_psnr,
        mean_endo4dgs_ssim,
        mean_rgbd_ssim,
    )
    return {
        "protocol": protocol,
        "baseline_protocol_declaration": baseline_protocol,
        "ranking_rule": "higher arithmetic mean per-frame PSNR; mean per-frame SSIM is the secondary tie-breaker",
        "aggregation": (
            "direct single-sequence means"
            if len(rows) == 1
            else "unweighted mean of per-sequence means"
        ),
        "sequence_count": len(rows),
        "challenge_order_result": {
            "winner": overall_winner,
            "decided_by": overall_decided_by,
            "endo4dgs_mean_psnr": mean_endo4dgs_psnr,
            "rgbd_mean_psnr": mean_rgbd_psnr,
            "endo4dgs_mean_ssim": mean_endo4dgs_ssim,
            "rgbd_mean_ssim": mean_rgbd_ssim,
        },
        "sequence_wins": {
            "psnr": _win_counts(delta_psnr),
            "ssim": _win_counts(delta_ssim),
            "rgbd_wins_both": sum(
                row["delta_psnr"] > 0 and row["delta_ssim"] > 0 for row in rows
            ),
            "challenge_order": {
                "rgbd_wins": sum(row["challenge_order_winner"] == "RGBD" for row in rows),
                "ties": sum(row["challenge_order_winner"] == "tie" for row in rows),
                "endo4dgs_wins": sum(
                    row["challenge_order_winner"] == "Endo4DGS" for row in rows
                ),
            },
        },
        "mean_delta_psnr": mean(delta_psnr),
        "median_delta_psnr": median(delta_psnr),
        "mean_delta_ssim": mean(delta_ssim),
        "median_delta_ssim": median(delta_ssim),
        "best_delta_psnr": _extreme(rows, "delta_psnr", best=True),
        "worst_delta_psnr": _extreme(rows, "delta_psnr", best=False),
        "best_delta_ssim": _extreme(rows, "delta_ssim", best=True),
        "worst_delta_ssim": _extreme(rows, "delta_ssim", best=False),
        "mean_rgbd_coverage_percent": mean(row["coverage_percent"] for row in rows),
        "baseline_only_sequences": list(baseline_only),
        "rgbd_only_sequences": list(rgbd_only),
        "rows": list(rows),
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = (
        "sequence",
        "endo4dgs_psnr",
        "rgbd_psnr",
        "delta_psnr",
        "endo4dgs_ssim",
        "rgbd_ssim",
        "delta_ssim",
        "coverage_percent",
        "challenge_order_winner",
        "challenge_order_decided_by",
        "baseline_method",
        "baseline_result",
        "rgbd_result",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: Sequence[dict]) -> str:
    lines = [
        "| Sequence | Endo4DGS PSNR | RGBD PSNR | ΔPSNR | Endo4DGS SSIM | RGBD SSIM | ΔSSIM | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sequence']} | {row['endo4dgs_psnr']:.3f} | {row['rgbd_psnr']:.3f} | "
            f"{row['delta_psnr']:+.3f} | {row['endo4dgs_ssim']:.4f} | {row['rgbd_ssim']:.4f} | "
            f"{row['delta_ssim']:+.4f} | {row['coverage_percent']:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    if args.protocol != args.baseline_protocol:
        raise ValueError(
            f"protocol mismatch: Method-1={args.protocol}, baseline declaration={args.baseline_protocol}; "
            "scores computed with different masks are not comparable"
        )
    output_paths = [
        args.output_dir / "comparison.csv",
        args.output_dir / "comparison.md",
        args.output_dir / "comparison_results.json",
    ]
    conflicts = [path for path in output_paths if path.exists()]
    if conflicts and not args.overwrite:
        raise FileExistsError(f"comparison output exists: {conflicts[0]}; use --overwrite")

    baselines = _filter_requested(
        _load_baselines(args.baseline_root, args.baseline_method, args.sequences),
        args.sequences,
        "baseline",
    )
    rgbd = _filter_requested(
        _load_rgbd(args.rgbd_root, args.protocol, args.sequences),
        args.sequences,
        "Method-1",
    )
    rows, baseline_only, rgbd_only = _comparison_rows(baselines, rgbd, args.allow_partial)
    summary = _build_summary(rows, args.protocol, args.baseline_protocol, baseline_only, rgbd_only)
    summary["baseline_root"] = str(args.baseline_root)
    summary["rgbd_root"] = str(args.rgbd_root)
    summary["baseline_method_request"] = args.baseline_method

    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown = _markdown(rows)
    _write_csv(output_paths[0], rows)
    output_paths[1].write_text(markdown, encoding="utf-8")
    output_paths[2].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(markdown, end="")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
