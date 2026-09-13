"""The complete frozen F1 fusion rule, isolated for independent testing."""

from __future__ import annotations

import numpy as np


def strict_hole_only_fusion(
    m3b_rgb: np.ndarray,
    method2_rgb: np.ndarray,
    m3b_valid: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Replace only pixels outside the native M3B filled-valid mask."""

    if m3b_rgb.dtype != np.uint8 or method2_rgb.dtype != np.uint8:
        raise TypeError("M3B and Method-2 RGB must be uint8")
    if m3b_rgb.shape != method2_rgb.shape or m3b_rgb.ndim != 3 or m3b_rgb.shape[2] != 3:
        raise ValueError("M3B and Method-2 RGB must have identical HxWx3 shapes")
    valid = np.asarray(m3b_valid, dtype=np.bool_)
    if valid.shape != m3b_rgb.shape[:2]:
        raise ValueError("M3B validity must have the same HxW shape as RGB")

    fallback = ~valid
    final = m3b_rgb.copy()
    final[fallback] = method2_rgb[fallback]
    if not np.array_equal(final[valid], m3b_rgb[valid]):
        raise AssertionError("Critical F1 invariant failed: an M3B-valid byte changed")
    return final, int(fallback.sum())
