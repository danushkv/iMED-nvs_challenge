#!/usr/bin/env python3
"""Source-only validation of the Method-1 Docker output contract."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def is_sequence_dir(path: Path) -> bool:
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


def source_frames(sequence: Path) -> list[Path]:
    source_dir = sequence / "endoscope2" / "L"
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: check_outputs.py INPUT_ROOT OUTPUT_ROOT")

    input_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])
    failures: list[str] = []

    for sequence in discover_sequences(input_root):
        sources = source_frames(sequence)
        expected_names = [f"{index:05d}.png" for index in range(len(sources))]
        render_dir = output_root / sequence.name / "renders"
        produced = sorted(path.name for path in render_dir.glob("*.png")) if render_dir.is_dir() else []
        if produced != expected_names:
            missing = sorted(set(expected_names) - set(produced))
            extra = sorted(set(produced) - set(expected_names))
            failures.append(
                f"{sequence.name}: filename mismatch, missing={missing[:3]}, extra={extra[:3]}"
            )
            continue
        if not sources:
            failures.append(f"{sequence.name}: no source RGB frames")
            continue

        with Image.open(sources[0]) as source_image:
            expected_size = source_image.size
        for output_name in expected_names:
            output_path = render_dir / output_name
            try:
                with Image.open(output_path) as prediction:
                    if prediction.format != "PNG":
                        failures.append(f"{output_path}: format is {prediction.format}, expected PNG")
                    if prediction.mode != "RGB":
                        failures.append(f"{output_path}: mode is {prediction.mode}, expected RGB")
                    if prediction.size != expected_size:
                        failures.append(
                            f"{output_path}: size is {prediction.size}, expected {expected_size}"
                        )
            except Exception as error:
                failures.append(f"{output_path}: cannot read image: {error}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1
    print("[OK] Method-1 NVS output contract passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
