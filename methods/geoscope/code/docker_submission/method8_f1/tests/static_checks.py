#!/usr/bin/env python3
"""Static audit of the isolated Method-8 build context."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "method2/train.py": "adcba1b40e0c76e6ae39cd3e81b92edfe36480e1874f4e42c67e6602979443c3",
    "method2/arguments/__init__.py": "0049d6239f2c939b775ce27fc0f1738c3fa29f3190ea43a522827c4419a50cc2",
    "method2/scene/imed_loader.py": "76f64bf45fd77a041a0aee84fde97332861ac3f904063cf3c01bfb9673ed89c3",
    "method2/imed_nvs_baseline.py": "d363c2984290b93f3daedd17538b174aa1cff03c96f81568c86be2032f835fde",
    "m3b/confidence.py": "708ce410ee9c38f9b58a4951f2ef84abbaf07da2484fc64853faf15a8bbc6e8e",
    "m3b/surface_geometry.py": "f678b71f5fd1606fa29b5fbfa673e83442d91229c4f40e4236a36ab841b3f8c1",
    "m3b/surface_splat.py": "aae69eb981cf85d9bc697332920c8182332ff069c07d2acbec383ab2b309d2da",
    "m3b/configs/m3_surface_default.json": "7bfd3b2195a69105184cad020a9f7788d6263b88dcfb46d981431143d9f7327c",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for relative, expected in EXPECTED_HASHES.items():
        actual = _sha256(ROOT / relative)
        assert actual == expected, f"Frozen source changed: {relative}: {actual} != {expected}"

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (ROOT / "combined_entrypoint.py").read_text(encoding="utf-8")
    fusion = (ROOT / "f1_fusion.py").read_text(encoding="utf-8")
    assert 'METHOD3_RENDERER=surface' in dockerfile
    assert 'TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"' in dockerfile
    assert 'METHOD2_ITERATIONS = 1000' in entrypoint
    assert 'METHOD2_COARSE_ITERATIONS = 300' in entrypoint
    assert 'METHOD2_DEPTH_LOSS = "metric_l1"' in entrypoint
    assert 'METHOD2_DEPTH_WEIGHT = "5e-5"' in entrypoint
    assert 'confidence_aware=False' in entrypoint
    assert 'FILL_RADIUS = 3' in entrypoint
    assert 'fallback = ~valid' in fusion
    assert "alpha" not in fusion.lower()
    assert "blend" not in fusion.lower()

    forbidden_names = {
        "hole_fallback.py",
        "aggregate_fallbacks.py",
        "evaluate.py",
        "metrics.py",
        "METHOD8_FINAL_REPORT.md",
        "method8_summary.json",
        "method8_ablation.csv",
    }
    packaged = {path.name for path in ROOT.rglob("*") if path.is_file()}
    assert packaged.isdisjoint(forbidden_names), sorted(packaged & forbidden_names)
    for path in ROOT.rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("STATIC CHECKS PASSED: frozen hashes/configuration and leakage allowlist")


if __name__ == "__main__":
    main()
