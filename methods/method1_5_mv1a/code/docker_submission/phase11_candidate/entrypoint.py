#!/usr/bin/env python3
"""Container entrypoint implementing the official iMED-NVS I/O contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from nvs_method import render_target_views


def is_sequence_dir(path: Path) -> bool:
    """Return whether ``path`` contains the permitted NVS inference inputs."""

    return (
        path.is_dir()
        and (path / "endoscope2").is_dir()
        and (path / "K.txt").is_file()
        and (path / "pose.txt").is_file()
    )


def discover_sequences(input_root: Path) -> list[Path]:
    if is_sequence_dir(input_root):
        return [input_root]
    return sorted(path for path in input_root.rglob("*") if is_sequence_dir(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Method-1 RGB-D reprojection submission")
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--max-sequences", type=int, default=None, help="Local smoke tests only")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(args.input)
    if args.max_sequences is not None:
        if args.max_sequences <= 0:
            raise ValueError("--max-sequences must be positive")
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise SystemExit(f"No iMED-NVS sequence directories found under {args.input}")

    print(f"[METHOD1] Discovered {len(sequences)} sequence(s)", flush=True)
    for sequence in sequences:
        render_target_views(sequence, args.output / sequence.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
