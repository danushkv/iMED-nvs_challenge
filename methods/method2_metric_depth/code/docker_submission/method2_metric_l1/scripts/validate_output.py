#!/usr/bin/env python3
"""Read-only validation of the iMED-NVS output contract."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image


OUTPUT_NAME = re.compile(r"\d{5}\.png")


def is_sequence_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "K.txt").is_file()
        and (path / "pose.txt").is_file()
        and (path / "endoscope2" / "L").is_dir()
    )


def discover_sequences(input_root: Path) -> list[Path]:
    if is_sequence_dir(input_root):
        return [input_root]
    return sorted(path for path in input_root.rglob("*") if is_sequence_dir(path))


def expected_count(sequence: Path) -> int:
    target_list = sequence / "target_frames.txt"
    if target_list.is_file():
        count = sum(1 for line in target_list.read_text(encoding="utf-8").splitlines() if line.strip())
    else:
        count = len(list((sequence / "endoscope2" / "L").glob("frame_*.png")))
    if count <= 0:
        raise ValueError(f"No requested frames found for {sequence.name}")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-width", type=int, default=1280)
    parser.add_argument("--expected-height", type=int, default=1024)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    failures: list[str] = []
    sequences = discover_sequences(args.input_root)
    if not sequences:
        raise SystemExit(f"No sequences found under {args.input_root}")

    for sequence in sequences:
        count = expected_count(sequence)
        expected_names = [f"{index:05d}.png" for index in range(count)]
        render_dir = args.output_root / sequence.name / "renders"
        if not render_dir.is_dir():
            failures.append(f"{sequence.name}: missing directory {render_dir}")
            continue

        entries = sorted(path.name for path in render_dir.iterdir())
        invalid_names = [name for name in entries if not OUTPUT_NAME.fullmatch(name)]
        if invalid_names:
            failures.append(f"{sequence.name}: unexpected render entries {invalid_names[:5]}")
        if entries != expected_names:
            missing = sorted(set(expected_names) - set(entries))
            extra = sorted(set(entries) - set(expected_names))
            failures.append(
                f"{sequence.name}: filename/count mismatch; "
                f"expected={count}, actual={len(entries)}, missing={missing[:5]}, extra={extra[:5]}"
            )
            continue

        for name in expected_names:
            path = render_dir / name
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.format != "PNG":
                        failures.append(f"{path}: format={image.format}, expected PNG")
                    if image.mode != "RGB":
                        failures.append(f"{path}: mode={image.mode}, expected RGB")
                    if image.size != (args.expected_width, args.expected_height):
                        failures.append(
                            f"{path}: size={image.size}, expected "
                            f"{args.expected_width}x{args.expected_height}"
                        )
            except Exception as error:
                failures.append(f"{path}: cannot decode PNG: {error}")

        if not failures:
            print(
                f"[OK] {sequence.name}: {count} sequential RGB PNGs at "
                f"{args.expected_width}x{args.expected_height}"
            )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] Method-2 NVS output contract passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
