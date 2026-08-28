#!/usr/bin/env python3
"""Minimal gsplat RGB+ED CUDA forward/backward smoke test.

This does not read challenge data or start training. It renders one synthetic
Gaussian into a 64x64 image and verifies finite forward and backward results.
"""

from __future__ import annotations

import torch
from gsplat import rasterization


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this test on the A100 node")

    device = torch.device("cuda")

    means = torch.tensor(
        [[0.0, 0.0, 2.0]], dtype=torch.float32, device=device, requires_grad=True
    )
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    scales = torch.full(
        (1, 3), 0.15, dtype=torch.float32, device=device, requires_grad=True
    )
    opacities = torch.tensor(
        [0.9], dtype=torch.float32, device=device, requires_grad=True
    )
    colors = torch.tensor(
        [[1.0, 0.2, 0.1]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    viewmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    intrinsics = torch.tensor(
        [
            [
                [50.0, 0.0, 32.0],
                [0.0, 50.0, 32.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    renders, alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=64,
        height=64,
        packed=True,
        sh_degree=None,
        render_mode="RGB+ED",
    )

    expected_shape = (1, 64, 64, 4)
    if tuple(renders.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected render shape: {tuple(renders.shape)}; expected {expected_shape}"
        )
    if not bool(torch.isfinite(renders).all()):
        raise RuntimeError("Render contains NaN or infinity")
    if not bool(torch.isfinite(alphas).all()):
        raise RuntimeError("Alpha image contains NaN or infinity")

    (renders.sum() + alphas.sum()).backward()

    parameters = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }
    for name, parameter in parameters.items():
        if parameter.grad is None:
            raise RuntimeError(f"{name}: missing gradient")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"{name}: gradient contains NaN or infinity")

    print("gsplat RGB+ED forward: OK")
    print("gsplat backward:       OK")
    print(f"render shape:           {tuple(renders.shape)}")
    print(f"alpha maximum:          {alphas.detach().max().item():.6f}")
    print(f"torch:                  {torch.__version__}")
    print(f"CUDA runtime:           {torch.version.cuda}")
    print(f"GPU:                    {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
