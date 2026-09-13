#!/usr/bin/env python3
"""Independently assert that Docker F1 preserves every valid M3B byte."""

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


def _native_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        values = np.asarray(mask, dtype=np.uint8)
        if not np.all(np.isin(np.unique(values), (0, 255))):
            raise AssertionError(f"Non-binary mask: {path}")
        if mask.size != size:
            resampling = getattr(Image, "Resampling", Image)
            mask = mask.resize(size, resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 127


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--m3b-renders", type=Path, required=True)
    parser.add_argument("--m3b-filled-mask", type=Path, required=True)
    args = parser.parse_args()
    prediction = _paths(args.prediction)
    m3b = _paths(args.m3b_renders)
    masks = _paths(args.m3b_filled_mask)
    expected_names = [path.name for path in prediction]
    if not prediction or [path.name for path in m3b] != expected_names or [path.name for path in masks] != expected_names:
        raise AssertionError("Prediction, M3B, and M3B-mask streams do not align")

    valid_pixels = 0
    changed_valid_pixels = 0
    changed_valid_bytes = 0
    maximum_valid_byte_error = 0
    for prediction_path, m3b_path, mask_path in zip(prediction, m3b, masks):
        final = _rgb(prediction_path)
        frozen = _rgb(m3b_path)
        if final.shape != frozen.shape:
            raise AssertionError(f"RGB shape mismatch at {prediction_path.name}")
        valid = _native_mask(mask_path, (final.shape[1], final.shape[0]))
        valid_pixels += int(valid.sum())
        valid_difference = np.abs(final[valid].astype(np.int16) - frozen[valid].astype(np.int16))
        changed_valid_pixels += int(np.any(valid_difference != 0, axis=1).sum())
        changed_valid_bytes += int((valid_difference != 0).sum())
        if valid_difference.size:
            maximum_valid_byte_error = max(
                maximum_valid_byte_error,
                int(valid_difference.max()),
            )
    if changed_valid_pixels:
        percentage = 100.0 * changed_valid_pixels / max(valid_pixels, 1)
        raise AssertionError(
            "Saved-M3B comparison differs: "
            f"changed_valid_pixels={changed_valid_pixels}/{valid_pixels} "
            f"({percentage:.8f}%) changed_valid_bytes={changed_valid_bytes} "
            f"maximum_valid_byte_error={maximum_valid_byte_error}. "
            "This comparison cannot distinguish cross-run M3B variation from fusion changes; "
            "the container separately asserts against its own in-memory M3B."
        )
    print(
        f"M3B VALID-PIXEL INVARIANT PASSED: frames={len(prediction)} "
        f"valid_pixels={valid_pixels} changed_valid_pixels=0"
    )


if __name__ == "__main__":
    main()
