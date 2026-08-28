#!/usr/bin/env python3
"""Read-only validation of Method-3 challenge outputs."""

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

    for sequence in sequences:
        source_frames = sorted((sequence / "endoscope2" / "L").glob("frame_*.png"))
        expected = [f"{index:05d}.png" for index in range(len(source_frames))]
        render_dir = args.output_root / sequence.name / "renders"
        if not source_frames:
            failures.append(f"{sequence.name}: no source frames")
            continue
        if not render_dir.is_dir():
            failures.append(f"{sequence.name}: missing {render_dir}")
            continue

        actual = sorted(path.name for path in render_dir.iterdir())
        invalid = [name for name in actual if not OUTPUT_NAME.fullmatch(name)]
        if invalid:
            failures.append(f"{sequence.name}: unexpected render entries {invalid[:5]}")
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            failures.append(
                f"{sequence.name}: filename/count mismatch expected={len(expected)} "
                f"actual={len(actual)} missing={missing[:5]} extra={extra[:5]}"
            )
            continue

        for name in expected:
            path = render_dir / name
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.format != "PNG":
                        failures.append(f"{path}: format={image.format}, expected PNG")
                    if image.mode != "RGB":
                        failures.append(f"{path}: mode={image.mode}, expected RGB")
                    expected_size = (args.expected_width, args.expected_height)
                    if image.size != expected_size:
                        failures.append(f"{path}: size={image.size}, expected={expected_size}")
            except Exception as error:
                failures.append(f"{path}: cannot decode PNG: {error}")

        if not any(item.startswith(f"{sequence.name}:") for item in failures):
            print(
                f"[OK] {sequence.name}: {len(expected)} sequential RGB PNGs at "
                f"{args.expected_width}x{args.expected_height}"
            )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] Method-3 NVS output contract passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
