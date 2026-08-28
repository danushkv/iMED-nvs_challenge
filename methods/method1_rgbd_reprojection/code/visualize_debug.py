"""Array-only visualization helpers.

This module never locates or opens target data. Evaluation-only callers may
pass already-loaded target arrays to ``save_evaluation_debug``; inference
callers use the source-only helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


def depth_to_rgb(depth: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Percentile-normalized grayscale depth for visualization only."""

    depth = np.asarray(depth)
    if valid is None:
        valid = np.isfinite(depth) & (depth > 0)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0)
    gray = np.zeros(depth.shape, dtype=np.float32)
    if valid.any():
        low, high = np.percentile(depth[valid], [2, 98])
        if high <= low:
            high = low + 1.0
        gray[valid] = np.clip((depth[valid] - low) / (high - low), 0, 1)
    # Near is bright, far is dark, invalid remains black.
    gray[valid] = 1.0 - gray[valid]
    return np.repeat((gray[..., None] * 255).round().astype(np.uint8), 3, axis=2)


def scalar_to_rgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    # Compact blue->cyan->yellow heat map without a matplotlib dependency.
    red = np.clip(2.0 * values - 0.5, 0, 1)
    green = np.clip(2.0 - np.abs(4.0 * values - 2.0), 0, 1)
    blue = np.clip(1.5 - 2.0 * values, 0, 1)
    return (np.stack((red, green, blue), axis=-1) * 255).round().astype(np.uint8)


def absolute_error_to_rgb(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Visualize mean absolute RGB error with excluded pixels left black."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    error = np.mean(np.abs(prediction[..., :3] - target[..., :3]), axis=2)
    error = np.where(mask, error, 0.0)
    return scalar_to_rgb(error)


def red_cyan_alignment_overlay(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Make aligned structure white and misaligned edges red/cyan.

    Target luminance is placed in red; prediction luminance is placed in green
    and blue. This is intended only for evaluation-time geometric diagnosis.
    """

    prediction = np.asarray(prediction, dtype=np.float32)[..., :3]
    target = np.asarray(target, dtype=np.float32)[..., :3]
    mask = np.asarray(mask, dtype=bool)
    prediction_luma = 0.299 * prediction[..., 0] + 0.587 * prediction[..., 1] + 0.114 * prediction[..., 2]
    target_luma = 0.299 * target[..., 0] + 0.587 * target[..., 1] + 0.114 * target[..., 2]
    overlay = np.stack((target_luma, prediction_luma, prediction_luma), axis=-1)
    overlay[~mask] = 0.0
    return (np.clip(overlay, 0, 1) * 255).round().astype(np.uint8)


def _uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image[..., :3]
    return (np.clip(image[..., :3], 0, 1) * 255).round().astype(np.uint8)


def save_inference_debug(
    output_path: Path,
    source_rgb: np.ndarray,
    source_depth: np.ndarray,
    projected_rgb: np.ndarray,
    projected_depth: np.ndarray,
    valid_mask: np.ndarray,
    confidence: np.ndarray,
) -> None:
    """Save the Phase-5 six-panel visualization without target information."""

    panels = [
        ("Source Endoscope2 RGB", _uint8_rgb(source_rgb)),
        ("Source depth (visual only)", depth_to_rgb(source_depth)),
        ("Projected Endoscope1 RGB", _uint8_rgb(projected_rgb)),
        ("Projected Endoscope1 depth", depth_to_rgb(projected_depth, valid_mask)),
        ("Confidence", scalar_to_rgb(confidence)),
        ("Displayed validity mask", np.repeat(np.asarray(valid_mask, dtype=np.uint8)[..., None] * 255, 3, axis=2)),
    ]
    height, width = panels[0][1].shape[:2]
    label_height = 28
    canvas = Image.new("RGB", (width * 2, (height + label_height) * 3), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        row, col = divmod(index, 2)
        x = col * width
        y = row * (height + label_height)
        draw.text((x + 6, y + 6), label, fill=(255, 255, 255))
        canvas.paste(Image.fromarray(array, mode="RGB"), (x, y + label_height))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_hole_fill_debug(
    output_path: Path,
    raw_rgb: np.ndarray,
    raw_depth: np.ndarray,
    raw_valid_mask: np.ndarray,
    filled_rgb: np.ndarray,
    filled_depth: np.ndarray,
    filled_confidence: np.ndarray,
    fill_mask: np.ndarray,
    filled_valid_mask: np.ndarray,
) -> None:
    """Save an H0-versus-filled comparison with explicit fill provenance."""

    panels = [
        ("H0 raw projected RGB", _uint8_rgb(raw_rgb)),
        ("H0 raw projected depth", depth_to_rgb(raw_depth, raw_valid_mask)),
        ("Selected filled RGB", _uint8_rgb(filled_rgb)),
        ("Selected filled depth", depth_to_rgb(filled_depth, filled_valid_mask)),
        ("Fill mask only", np.repeat(np.asarray(fill_mask, dtype=np.uint8)[..., None] * 255, 3, axis=2)),
        ("Filled confidence", scalar_to_rgb(filled_confidence)),
    ]
    height, width = panels[0][1].shape[:2]
    label_height = 28
    canvas = Image.new("RGB", (width * 2, (height + label_height) * 3), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        row, col = divmod(index, 2)
        x = col * width
        y = row * (height + label_height)
        draw.text((x + 6, y + 6), label, fill=(255, 255, 255))
        canvas.paste(Image.fromarray(array, mode="RGB"), (x, y + label_height))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_evaluation_debug(
    output_path: Path,
    source_rgb: np.ndarray,
    source_depth: np.ndarray,
    projected_rgb: np.ndarray,
    projected_depth: np.ndarray,
    confidence: np.ndarray,
    raw_valid_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    target_rgb: np.ndarray,
    title: str,
) -> None:
    """Save the requested Phase-10 evaluation-only nine-panel montage."""

    source_rgb = _uint8_rgb(source_rgb)
    projected_rgb_u8 = _uint8_rgb(projected_rgb)
    target_rgb_u8 = _uint8_rgb(target_rgb)
    height, width = projected_rgb_u8.shape[:2]

    def resize_rgb(array: np.ndarray, nearest: bool = False) -> np.ndarray:
        image = Image.fromarray(_uint8_rgb(array), mode="RGB")
        if image.size != (width, height):
            resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
            image = image.resize((width, height), resampling)
        return np.asarray(image)

    source_rgb = resize_rgb(source_rgb)
    source_depth_rgb = resize_rgb(depth_to_rgb(source_depth))
    projected_depth_rgb = resize_rgb(depth_to_rgb(projected_depth, raw_valid_mask))
    confidence_rgb = resize_rgb(scalar_to_rgb(confidence))

    valid = np.asarray(raw_valid_mask, dtype=bool)
    evaluation = np.asarray(evaluation_mask, dtype=bool)
    support = np.zeros((height, width, 3), dtype=np.uint8)
    support[..., 0] = evaluation.astype(np.uint8) * 255
    support[..., 1] = valid.astype(np.uint8) * 255
    # Red-only is evaluation support, green-only is raw projection support,
    # and their intersection appears yellow.

    prediction_float = projected_rgb_u8.astype(np.float32) / 255.0
    target_float = target_rgb_u8.astype(np.float32) / 255.0
    panels = [
        ("Source Endoscope2 RGB", source_rgb),
        ("Source metric depth", source_depth_rgb),
        ("Projected Endoscope1 RGB", projected_rgb_u8),
        ("Projected Endoscope1 depth", projected_depth_rgb),
        ("Projection confidence", confidence_rgb),
        ("Mask: eval=red valid=green overlap=yellow", support),
        ("GT Endoscope1 RGB (evaluation only)", target_rgb_u8),
        ("Absolute RGB error (evaluation mask)", absolute_error_to_rgb(prediction_float, target_float, evaluation)),
        ("Red=GT, cyan=prediction; white=aligned", red_cyan_alignment_overlay(prediction_float, target_float, evaluation)),
    ]

    label_height = 32
    title_height = 34
    canvas = Image.new(
        "RGB",
        (width * 3, title_height + (height + label_height) * 3),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), title, fill=(255, 255, 255))
    for index, (label, array) in enumerate(panels):
        row, col = divmod(index, 3)
        x = col * width
        y = title_height + row * (height + label_height)
        draw.text((x + 6, y + 8), label, fill=(255, 255, 255))
        canvas.paste(Image.fromarray(array, mode="RGB"), (x, y + label_height))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
