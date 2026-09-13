#!/usr/bin/env python3
"""Compare Docker and saved F1 output as exact decoded RGB bytes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9].png"))


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    prediction = _paths(args.prediction)
    reference = _paths(args.reference)
    if [path.name for path in prediction] != [path.name for path in reference]:
        raise AssertionError("Prediction/reference frame streams differ")
    if not prediction:
        raise AssertionError("No frames to compare")

    differing_frames = 0
    differing_bytes = 0
    maximum_error = 0
    for predicted_path, reference_path in zip(prediction, reference):
        predicted = _rgb(predicted_path)
        expected = _rgb(reference_path)
        if predicted.shape != expected.shape:
            raise AssertionError(
                f"Shape mismatch at {predicted_path.name}: {predicted.shape} != {expected.shape}"
            )
        difference = np.abs(predicted.astype(np.int16) - expected.astype(np.int16))
        if bool(difference.any()):
            differing_frames += 1
            differing_bytes += int((difference != 0).sum())
            maximum_error = max(maximum_error, int(difference.max()))
    if differing_frames:
        raise AssertionError(
            f"F1 mismatch: frames={differing_frames}/{len(prediction)} "
            f"bytes={differing_bytes} maximum_byte_error={maximum_error}"
        )
    print(f"EXACT F1 PIXEL MATCH: frames={len(prediction)} differing_bytes=0")


if __name__ == "__main__":
    main()
