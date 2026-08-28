#!/usr/bin/env python3
"""Source-only challenge wrapper for the accepted Method-2 configuration."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO = Path("/workspace/Endo-4DGS")
RUNNER = REPO / "imed_nvs_baseline.py"
ITERATIONS = 1000
COARSE_ITERATIONS = 300
DEPTH_LOSS = "metric_l1"
DEPTH_WEIGHT = "5e-5"


def is_sequence_dir(path: Path) -> bool:
    """Recognize a sequence without probing any Endoscope-1 path."""

    return (
        path.is_dir()
        and (path / "K.txt").is_file()
        and (path / "pose.txt").is_file()
        and (path / "endoscope2" / "L").is_dir()
        and (path / "endoscope2" / "depthL").is_dir()
        and (path / "endoscope2" / "toolL").is_dir()
    )


def discover_sequences(input_root: Path) -> list[Path]:
    if is_sequence_dir(input_root):
        return [input_root]
    return sorted(path for path in input_root.rglob("*") if is_sequence_dir(path))


def source_frame_names(sequence: Path) -> list[str]:
    names = sorted(path.name for path in (sequence / "endoscope2" / "L").glob("frame_*.png"))
    if not names:
        raise FileNotFoundError(f"No Endoscope-2 RGB frames in {sequence / 'endoscope2' / 'L'}")
    return names


def requested_target_names(sequence: Path) -> list[str]:
    """Use permitted metadata, otherwise the synchronized source frame list."""

    source_names = source_frame_names(sequence)
    target_list = sequence / "target_frames.txt"
    if target_list.is_file():
        names = [line.strip() for line in target_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not names:
            raise ValueError(f"Target frame list is empty: {target_list}")
        normalized = [name if Path(name).suffix else f"{name}.png" for name in names]
        if [Path(name).stem for name in normalized] != [Path(name).stem for name in source_names]:
            raise ValueError(
                "Method 2 requires synchronized one-to-one source/target frame ids; "
                f"target_frames.txt does not match Endoscope-2 RGB in {sequence}"
            )
        return normalized
    return source_names


def safe_symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite internal input view: {destination}")
    os.symlink(source, destination, target_is_directory=source.is_dir())


def make_source_only_view(sequence: Path, staging_root: Path) -> Path:
    """Expose only permitted inputs to the existing Endo-4DGS adapter."""

    staged = staging_root / sequence.name
    staged.mkdir(parents=True)
    safe_symlink(sequence / "K.txt", staged / "K.txt")
    safe_symlink(sequence / "pose.txt", staged / "pose.txt")

    staged_source = staged / "endoscope2"
    staged_source.mkdir()
    for stream in ("L", "depthL", "toolL"):
        safe_symlink(sequence / "endoscope2" / stream, staged_source / stream)

    # The loader needs target camera records but not target pixels. An empty
    # synthetic Endoscope-1 directory plus frame-name metadata produces camera
    # objects from K1_L and camera id 1 without exposing target RGB/depth/masks.
    (staged / "endoscope1").mkdir()
    (staged / "target_frames.txt").write_text(
        "".join(f"{name}\n" for name in requested_target_names(sequence)),
        encoding="utf-8",
    )
    return staged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Method-2 iMED-NVS submission")
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sequences = discover_sequences(args.input)
    if not sequences:
        raise SystemExit(f"No iMED-NVS sequences found under {args.input}")

    print(
        "[METHOD2] "
        f"sequences={len(sequences)} iterations={ITERATIONS} "
        f"coarse_iterations={COARSE_ITERATIONS} depth_loss={DEPTH_LOSS} "
        f"depth_weight={DEPTH_WEIGHT} appearance_correction=disabled "
        "additional_source_dssim=disabled modified_tool_mask_loss=disabled "
        "confidence_losses=enabled metrics=disabled",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="method2-source-only-", dir="/tmp") as temporary:
        staged_root = Path(temporary)
        for sequence in sequences:
            started = time.perf_counter()
            frame_count = len(requested_target_names(sequence))
            output_path = args.output / sequence.name
            if output_path.exists() and any(output_path.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite a non-empty sequence output: {output_path}"
                )
            staged_sequence = make_source_only_view(sequence, staged_root)
            command = [
                sys.executable,
                str(RUNNER),
                "--repo",
                str(REPO),
                "run-sequence",
                "--sequence",
                str(staged_sequence),
                "--output",
                str(output_path),
                "--iterations",
                str(ITERATIONS),
                "--coarse-iterations",
                str(COARSE_ITERATIONS),
                "--depth-loss",
                DEPTH_LOSS,
                "--depth-weight",
                DEPTH_WEIGHT,
                "--no-metrics",
            ]
            print(
                f"[METHOD2] sequence={sequence.name} frames={frame_count} "
                f"output={output_path}",
                flush=True,
            )
            subprocess.run(command, cwd=REPO, check=True)
            elapsed = time.perf_counter() - started
            print(
                f"[METHOD2] completed sequence={sequence.name} frames={frame_count} "
                f"runtime_seconds={elapsed:.3f} output={output_path / 'renders'}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
