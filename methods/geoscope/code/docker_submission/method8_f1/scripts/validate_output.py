#!/usr/bin/env python3
"""Read-only validation of the Method-8 challenge output contract."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image


OUTPUT_NAME = re.compile(r"\d{5}\.png")


def _is_sequence(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "K.txt").is_file()
        and (path / "pose.txt").is_file()
        and (path / "endoscope2" / "L").is_dir()
        and (path / "endoscope2" / "depthL").is_dir()
        and (path / "endoscope2" / "toolL").is_dir()
    )


def _sequences(input_root: Path) -> list[Path]:
    if _is_sequence(input_root):
        return [input_root]
    return sorted(path for path in input_root.rglob("*") if _is_sequence(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-width", type=int, default=1280)
    parser.add_argument("--expected-height", type=int, default=1024)
    return parser


def main() -> int:
    args = _parser().parse_args()
    failures: list[str] = []
    sequences = _sequences(args.input_root)
    if not sequences:
        raise SystemExit(f"No sequences found under {args.input_root}")

    expected_sequence_names = {sequence.name for sequence in sequences}
    actual_sequence_names = {path.name for path in args.output_root.iterdir() if path.is_dir()}
    if actual_sequence_names != expected_sequence_names:
        failures.append(
            f"sequence directories mismatch expected={sorted(expected_sequence_names)} "
            f"actual={sorted(actual_sequence_names)}"
        )

    for sequence in sequences:
        source_frames = sorted((sequence / "endoscope2" / "L").glob("frame_*.png"))
        expected = [f"{index:05d}.png" for index in range(len(source_frames))]
        sequence_output = args.output_root / sequence.name
        render_dir = sequence_output / "renders"
        if not render_dir.is_dir():
            failures.append(f"{sequence.name}: missing {render_dir}")
            continue
        extra_sequence_entries = sorted(path.name for path in sequence_output.iterdir() if path.name != "renders")
        if extra_sequence_entries:
            failures.append(f"{sequence.name}: unexpected output entries {extra_sequence_entries}")
        actual = sorted(path.name for path in render_dir.iterdir())
        invalid = [name for name in actual if not OUTPUT_NAME.fullmatch(name)]
        if invalid:
            failures.append(f"{sequence.name}: unexpected render entries {invalid[:5]}")
        if actual != expected:
            failures.append(
                f"{sequence.name}: filename/count mismatch expected={len(expected)} actual={len(actual)}"
            )
            continue
        for name in expected:
            path = render_dir / name
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.format != "PNG" or image.mode != "RGB":
                        failures.append(f"{path}: expected RGB PNG, got {image.mode} {image.format}")
                    if image.size != (args.expected_width, args.expected_height):
                        failures.append(
                            f"{path}: size={image.size}, expected={(args.expected_width, args.expected_height)}"
                        )
            except Exception as error:
                failures.append(f"{path}: cannot decode PNG: {error}")
        print(f"[CHECKED] {sequence.name}: {len(expected)} frames")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] Method-8 F1 output contract passed; output contains final RGB renders only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
