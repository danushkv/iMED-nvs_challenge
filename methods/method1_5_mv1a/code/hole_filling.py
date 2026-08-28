"""Conservative Phase-6 hole filling with explicit provenance masks.

These functions operate only on a rendered prediction and its confidence. They
never read target RGB, target depth, or target masks. Raw geometric validity is
preserved separately from the pixels synthesized by a fill strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class HoleFillResult:
    rgb: torch.Tensor  # [H,W,3]
    depth: torch.Tensor  # [H,W]
    confidence: torch.Tensor  # [H,W]
    raw_valid_mask: torch.Tensor  # [H,W], immutable geometric support
    fill_mask: torch.Tensor  # [H,W], pixels synthesized by this strategy
    filled_valid_mask: torch.Tensor  # raw_valid_mask | fill_mask
    method: str


def _validate_inputs(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must have shape [H,W,3], got {tuple(rgb.shape)}")
    if depth.shape != rgb.shape[:2] or valid_mask.shape != rgb.shape[:2] or confidence.shape != rgb.shape[:2]:
        raise ValueError("depth, valid_mask, and confidence must match rgb spatial dimensions")


def no_fill(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: torch.Tensor,
) -> HoleFillResult:
    """H0: return raw geometric output without synthesizing pixels."""

    _validate_inputs(rgb, depth, valid_mask, confidence)
    raw_valid = valid_mask.bool().clone()
    return HoleFillResult(
        rgb=rgb.clone(),
        depth=depth.clone(),
        confidence=confidence.clone(),
        raw_valid_mask=raw_valid,
        fill_mask=torch.zeros_like(raw_valid),
        filled_valid_mask=raw_valid.clone(),
        method="none",
    )


def _conv_sum(values_nchw: torch.Tensor, channels: int) -> torch.Tensor:
    kernel = torch.ones((channels, 1, 3, 3), dtype=values_nchw.dtype, device=values_nchw.device)
    return F.conv2d(values_nchw, kernel, padding=1, groups=channels)


def morphological_fill(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: torch.Tensor,
    iterations: int = 1,
    min_valid_neighbors: int = 7,
    min_neighbor_confidence: float = 0.1,
) -> HoleFillResult:
    """H1: fill only 1-2 pixel holes that are almost completely surrounded.

    A candidate needs at least ``min_valid_neighbors`` of its eight neighbors.
    With the default of seven, boundaries of large disocclusions cannot grow
    inward. The operation is vectorized; the only loop is over at most three
    morphological iterations, never over image pixels.
    """

    _validate_inputs(rgb, depth, valid_mask, confidence)
    if iterations not in (1, 2, 3):
        raise ValueError("morphological iterations must be 1, 2, or 3")
    if not 1 <= min_valid_neighbors <= 8:
        raise ValueError("min_valid_neighbors must be in [1,8]")
    if not 0 <= min_neighbor_confidence <= 1:
        raise ValueError("min_neighbor_confidence must be in [0,1]")

    raw_valid = valid_mask.bool().clone()
    current_valid = raw_valid.clone()
    filled_rgb = rgb.clone()
    filled_depth = depth.clone()
    filled_confidence = confidence.clone()
    fill_mask = torch.zeros_like(raw_valid)

    for iteration in range(iterations):
        valid_f = current_valid.to(torch.float32)[None, None]
        # Exclude the center so the threshold explicitly counts 8 neighbors.
        neighbor_count = _conv_sum(valid_f, 1)[0, 0] - valid_f[0, 0]
        confidence_weight = (filled_confidence * current_valid).clamp_min(0)[None, None]
        weight_sum = _conv_sum(confidence_weight, 1)[0, 0] - confidence_weight[0, 0]
        mean_neighbor_confidence = weight_sum / neighbor_count.clamp_min(1)
        candidates = (
            (~current_valid)
            & (neighbor_count >= min_valid_neighbors)
            & (mean_neighbor_confidence >= min_neighbor_confidence)
            & (weight_sum > 1e-8)
        )
        if not bool(candidates.any()):
            break

        rgb_nchw = filled_rgb.permute(2, 0, 1)[None]
        rgb_weighted = rgb_nchw * confidence_weight
        rgb_sum = _conv_sum(rgb_weighted, 3)[0].permute(1, 2, 0) - filled_rgb * confidence_weight[0, 0, :, :, None]
        depth_weighted = (filled_depth[None, None] * confidence_weight)
        depth_sum = _conv_sum(depth_weighted, 1)[0, 0] - filled_depth * confidence_weight[0, 0]

        filled_rgb[candidates] = rgb_sum[candidates] / weight_sum[candidates, None]
        filled_depth[candidates] = depth_sum[candidates] / weight_sum[candidates]
        # Confidence decays for each synthesized layer and stays below its
        # supporting neighborhood's mean confidence.
        decay = 0.75 ** (iteration + 1)
        filled_confidence[candidates] = (mean_neighbor_confidence[candidates] * decay).clamp(0, 1)
        current_valid |= candidates
        fill_mask |= candidates

    return HoleFillResult(
        rgb=filled_rgb,
        depth=filled_depth,
        confidence=filled_confidence,
        raw_valid_mask=raw_valid,
        fill_mask=fill_mask,
        filled_valid_mask=current_valid,
        method="morphological",
    )


def _nearest_source_indices(valid_mask: torch.Tensor, radius: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find nearest raw-valid source index within a Euclidean pixel radius."""

    height, width = valid_mask.shape
    device = valid_mask.device
    source_indices = torch.arange(height * width, dtype=torch.int64, device=device).reshape(height, width)
    best_index = torch.full((height, width), -1, dtype=torch.int64, device=device)
    best_distance = torch.full((height, width), float("inf"), dtype=torch.float32, device=device)
    offsets = [
        (dy * dy + dx * dx, dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if (dy != 0 or dx != 0) and dy * dy + dx * dx <= radius * radius
    ]
    offsets.sort(key=lambda item: (item[0], abs(item[1]) + abs(item[2]), item[1], item[2]))
    raw_invalid = ~valid_mask
    for distance_sq, dy, dx in offsets:
        if dy >= 0:
            dst_y = slice(0, height - dy)
            src_y = slice(dy, height)
        else:
            dst_y = slice(-dy, height)
            src_y = slice(0, height + dy)
        if dx >= 0:
            dst_x = slice(0, width - dx)
            src_x = slice(dx, width)
        else:
            dst_x = slice(-dx, width)
            src_x = slice(0, width + dx)
        available = raw_invalid[dst_y, dst_x] & (best_index[dst_y, dst_x] < 0) & valid_mask[src_y, src_x]
        best_index[dst_y, dst_x] = torch.where(
            available,
            source_indices[src_y, src_x],
            best_index[dst_y, dst_x],
        )
        best_distance[dst_y, dst_x] = torch.where(
            available,
            torch.full_like(best_distance[dst_y, dst_x], float(distance_sq) ** 0.5),
            best_distance[dst_y, dst_x],
        )
    return best_index, best_distance


def nearest_valid_fill(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: torch.Tensor,
    radius: int = 1,
) -> HoleFillResult:
    """H2: propagate the nearest raw-valid sample by at most 1-3 pixels."""

    _validate_inputs(rgb, depth, valid_mask, confidence)
    if radius not in (1, 2, 3):
        raise ValueError("nearest-fill radius must be 1, 2, or 3")
    raw_valid = valid_mask.bool().clone()
    best_index, best_distance = _nearest_source_indices(raw_valid, radius)
    fill_mask = (~raw_valid) & (best_index >= 0)
    filled_rgb = rgb.clone()
    filled_depth = depth.clone()
    filled_confidence = confidence.clone()
    if bool(fill_mask.any()):
        flat_index = best_index[fill_mask]
        filled_rgb[fill_mask] = rgb.reshape(-1, 3)[flat_index]
        filled_depth[fill_mask] = depth.reshape(-1)[flat_index]
        distance_decay = torch.exp(-best_distance[fill_mask] / float(radius))
        filled_confidence[fill_mask] = (confidence.reshape(-1)[flat_index] * distance_decay).clamp(0, 1)
    return HoleFillResult(
        rgb=filled_rgb,
        depth=filled_depth,
        confidence=filled_confidence,
        raw_valid_mask=raw_valid,
        fill_mask=fill_mask,
        filled_valid_mask=raw_valid | fill_mask,
        method="nearest",
    )


def small_hole_inpaint(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: torch.Tensor,
    radius: int = 1,
    max_component_area: int = 25,
) -> HoleFillResult:
    """H3: OpenCV Telea RGB inpainting for small invalid components only.

    Large exterior regions, disocclusions, and tool-shaped components are never
    selected. Depth/confidence use radius-limited nearest propagation; no target
    signal is used.
    """

    _validate_inputs(rgb, depth, valid_mask, confidence)
    if radius not in (1, 2, 3):
        raise ValueError("inpaint radius must be 1, 2, or 3")
    if max_component_area < 1:
        raise ValueError("max_component_area must be positive")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "H3 requires optional OpenCV: python -m pip install opencv-python-headless"
        ) from error

    raw_valid = valid_mask.bool().clone()
    invalid_u8 = (~raw_valid).detach().cpu().numpy().astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(invalid_u8, connectivity=8)
    selected = np.zeros_like(invalid_u8, dtype=bool)
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area <= max_component_area:
            selected |= labels == component_id
    selected_t = torch.from_numpy(selected).to(device=rgb.device)

    nearest = nearest_valid_fill(rgb, depth, raw_valid, confidence, radius=radius)
    fill_mask = selected_t & nearest.fill_mask
    filled_rgb = rgb.clone()
    filled_depth = depth.clone()
    filled_confidence = confidence.clone()
    if bool(fill_mask.any()):
        rgb_u8 = (rgb.detach().cpu().numpy().clip(0, 1) * 255.0).round().astype(np.uint8)
        inpaint_mask = fill_mask.detach().cpu().numpy().astype(np.uint8) * 255
        inpainted_u8 = cv2.inpaint(rgb_u8, inpaint_mask, float(radius), cv2.INPAINT_TELEA)
        inpainted = torch.from_numpy(inpainted_u8.astype(np.float32) / 255.0).to(rgb.device)
        filled_rgb[fill_mask] = inpainted[fill_mask]
        filled_depth[fill_mask] = nearest.depth[fill_mask]
        filled_confidence[fill_mask] = nearest.confidence[fill_mask]
    return HoleFillResult(
        rgb=filled_rgb,
        depth=filled_depth,
        confidence=filled_confidence,
        raw_valid_mask=raw_valid,
        fill_mask=fill_mask,
        filled_valid_mask=raw_valid | fill_mask,
        method="inpaint",
    )


def apply_hole_fill(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: torch.Tensor,
    method: str = "none",
    radius: int = 1,
    morph_min_neighbors: int = 7,
    morph_min_confidence: float = 0.1,
    inpaint_max_area: int = 25,
) -> HoleFillResult:
    """Dispatch one explicitly named Phase-6 strategy."""

    if method == "none":
        return no_fill(rgb, depth, valid_mask, confidence)
    if method == "morphological":
        return morphological_fill(
            rgb,
            depth,
            valid_mask,
            confidence,
            iterations=radius,
            min_valid_neighbors=morph_min_neighbors,
            min_neighbor_confidence=morph_min_confidence,
        )
    if method == "nearest":
        return nearest_valid_fill(rgb, depth, valid_mask, confidence, radius=radius)
    if method == "inpaint":
        return small_hole_inpaint(
            rgb,
            depth,
            valid_mask,
            confidence,
            radius=radius,
            max_component_area=inpaint_max_area,
        )
    raise ValueError(f"unsupported hole-fill method: {method}")
