#!/usr/bin/env python3
"""Independent synthetic tests of the frozen F1 fusion rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION_ROOT))
from f1_fusion import strict_hole_only_fusion  # noqa: E402


def main() -> None:
    generator = np.random.default_rng(20260831)
    m3b = generator.integers(0, 256, size=(31, 47, 3), dtype=np.uint8)
    method2 = generator.integers(0, 256, size=m3b.shape, dtype=np.uint8)
    valid = generator.random(m3b.shape[:2]) > 0.23
    final, fallback_count = strict_hole_only_fusion(m3b, method2, valid)

    assert fallback_count == int((~valid).sum())
    assert np.array_equal(final[valid], m3b[valid])
    assert np.array_equal(final[~valid], method2[~valid])
    assert not np.shares_memory(final, m3b)

    all_valid, count = strict_hole_only_fusion(m3b, method2, np.ones_like(valid))
    assert count == 0 and np.array_equal(all_valid, m3b)
    all_holes, count = strict_hole_only_fusion(m3b, method2, np.zeros_like(valid))
    assert count == valid.size and np.array_equal(all_holes, method2)
    print("F1 INVARIANT TEST PASSED: valid=M3B exactly, holes=Method2 exactly")


if __name__ == "__main__":
    main()
