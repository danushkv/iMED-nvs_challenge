#!/usr/bin/env python3
"""Challenge entrypoint for the frozen Method 4B initialization ablation.

M4-B is an explicit reproducibility profile. It enumerates only calibration
and Endoscope2 source inputs and never opens Endoscope1 RGB, depth, or masks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image


WORKSPACE = Path("/workspace")
METHOD_ROOT = WORKSPACE / "method4_gsharp"
GSPLAT_COMMIT_FILE = METHOD_ROOT / "GSHARP_GSPLAT_COMMIT.txt"
EXPECTED_GSPLAT_COMMIT = "846c07932a77a901b474c40dd7fbfe42965ab354"
EXPECTED_ARCHITECTURES = "8.0;8.6"
IMAGE_VARIANT = "m4b"


def _source_rgb_files(sequence: Path) -> list[Path]:
    return sorted((sequence / "endoscope2" / "L").glob("frame_*.png"))


def _is_legal_sequence(sequence: Path) -> bool:
    """Validate one path using only calibration and Endoscope2 source data."""

    return (
        sequence.is_dir()
        and (sequence / "K.txt").is_file()
        and (sequence / "pose.txt").is_file()
        and bool(_source_rgb_files(sequence))
        and bool(sorted((sequence / "endoscope2" / "depthL").glob("frame_*.npy")))
        and bool(sorted((sequence / "endoscope2" / "toolL").glob("frame_*.png")))
    )


def _discover_sequences(data_root: Path, selected: list[str] | None) -> list[Path]:
    """Discover direct children only; never recursively walk a sequence."""

    if not data_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {data_root}")
    if selected:
        sequences = [data_root / name for name in selected]
        invalid_names = [name for name in selected if Path(name).name != name]
        if invalid_names:
            raise ValueError(f"Sequence names must be direct child names: {invalid_names}")
    elif _is_legal_sequence(data_root):
        sequences = [data_root]
    else:
        sequences = sorted(path for path in data_root.iterdir() if _is_legal_sequence(path))
    invalid = [str(path) for path in sequences if not _is_legal_sequence(path)]
    if invalid:
        raise FileNotFoundError(f"Invalid or incomplete legal source sequence(s): {invalid}")
    if not sequences:
        raise FileNotFoundError(f"No legal iMED source sequences directly under {data_root}")
    return sequences


def _run(command: list[str]) -> None:
    print("[M4-B RUN]", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(WORKSPACE), str(METHOD_ROOT / "third_party" / "gsplat"))
    )
    subprocess.run(command, cwd=str(WORKSPACE), env=environment, check=True)


def _audit_environment() -> None:
    import gsplat

    commit = GSPLAT_COMMIT_FILE.read_text(encoding="utf-8").strip()
    if commit != EXPECTED_GSPLAT_COMMIT:
        raise RuntimeError(f"Unexpected gsplat commit record: {commit}")
    if torch.__version__ != "2.9.1+cu126" or torch.version.cuda != "12.6":
        raise RuntimeError(
            f"Unexpected torch/CUDA build: {torch.__version__}, {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    audit = {
        "method_variant": IMAGE_VARIANT,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch_architectures": torch.cuda.get_arch_list(),
        "compiled_architecture_request": EXPECTED_ARCHITECTURES,
        "gsplat": str(Path(gsplat.__file__).resolve()),
        "gsplat_commit": commit,
        "target_rgb_access": False,
    }
    print("[M4-B ENV] " + json.dumps(audit, sort_keys=True), flush=True)


def _assert_output_available(sequence_output: Path) -> None:
    if sequence_output.exists() and any(sequence_output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {sequence_output}")


def _export_renders(source: Path, destination: Path, expected_count: int) -> None:
    render_paths = sorted(source.glob("*.png"))
    expected_names = [f"{index:05d}.png" for index in range(expected_count)]
    actual_names = [path.name for path in render_paths]
    if actual_names != expected_names:
        raise RuntimeError(
            f"Render contract mismatch: expected {expected_count} contiguous files; "
            f"found {len(render_paths)}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in render_paths:
        destination_path = destination / source_path.name
        shutil.copy2(source_path, destination_path)
        with Image.open(destination_path) as image:
            if image.mode != "RGB":
                raise RuntimeError(f"Non-RGB challenge render: {destination_path}")


def _run_sequence(sequence: Path, output_root: Path, temporary_root: Path) -> None:
    sequence_name = sequence.name
    source_frames = _source_rgb_files(sequence)
    with Image.open(source_frames[0]) as image:
        native_size = image.size
    output_sequence = output_root / sequence_name
    _assert_output_available(output_sequence)

    work_dir = Path(tempfile.mkdtemp(prefix=f"{sequence_name}-", dir=temporary_root))
    train_root = work_dir / "train"
    render_root = work_dir / "target"
    checkpoint = train_root / sequence_name / "checkpoints" / "m4b_final.pt"
    started = time.perf_counter()
    try:
        _run(
            [
                sys.executable,
                str(METHOD_ROOT / "train_imed_m4b.py"),
                "--data-root",
                str(sequence.parent),
                "--sequence",
                sequence_name,
                "--output-dir",
                str(train_root),
                "--coarse-steps",
                "500",
                "--fine-steps",
                "3000",
                "--init-max-points",
                "50000",
                "--voxel-size",
                "0.5",
                "--thickness-ratio",
                "0.2",
                "--candidate-multiplier",
                "4.0",
                "--world-scale",
                "1.0",
                "--seed",
                "42",
                "--device",
                "cuda",
            ]
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training did not produce checkpoint: {checkpoint}")
        _run(
            [
                sys.executable,
                str(METHOD_ROOT / "render_imed.py"),
                "--view",
                "target",
                "--data-root",
                str(sequence.parent),
                "--sequence",
                sequence_name,
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(render_root),
                "--device",
                "cuda",
            ]
        )
        source_render_dir = render_root / sequence_name / "renders"
        _export_renders(
            source_render_dir,
            output_sequence / "renders",
            expected_count=len(source_frames),
        )
        first_output = output_sequence / "renders" / "00000.png"
        with Image.open(first_output) as image:
            if image.size != native_size:
                raise RuntimeError(
                    f"Native output size mismatch: {image.size} versus {native_size}"
                )
        print(
            "[M4-B COMPLETE] "
            + json.dumps(
                {
                    "sequence": sequence_name,
                    "renders": len(source_frames),
                    "resolution_wh": list(native_size),
                    "runtime_seconds": time.perf_counter() - started,
                    "target_rgb_access": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        resolved_work = work_dir.resolve()
        resolved_temp = temporary_root.resolve()
        if resolved_work.parent == resolved_temp and resolved_work.name.startswith(
            f"{sequence_name}-"
        ):
            shutil.rmtree(resolved_work, ignore_errors=False)
        else:
            raise RuntimeError(f"Refusing unsafe temporary cleanup: {resolved_work}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen iMED NVS Method 4B reproducibility profile"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_dataset = subparsers.add_parser("run-dataset")
    run_dataset.add_argument("--data-root", type=Path, default=Path("/input"))
    run_dataset.add_argument("--output-root", type=Path, default=Path("/output"))
    run_dataset.add_argument(
        "--sequence",
        action="append",
        help="Optional direct-child sequence name; repeat to select multiple.",
    )
    run_dataset.add_argument("--max-sequences", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    _audit_environment()
    sequences = _discover_sequences(args.data_root, args.sequence)
    if args.max_sequences is not None:
        if args.max_sequences <= 0:
            raise ValueError("--max-sequences must be positive")
        sequences = sequences[: args.max_sequences]
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path("/tmp/method4")
    temporary_root.mkdir(parents=True, exist_ok=True)
    print(f"[M4-B] discovered {len(sequences)} legal sequence(s)", flush=True)
    for sequence in sequences:
        _run_sequence(sequence, args.output_root, temporary_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
