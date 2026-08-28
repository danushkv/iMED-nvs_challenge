#!/usr/bin/env python3
"""Source-to-source geometry gate. This script never opens Endoscope 1 RGB."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from camera import backproject_depth, project_points
from inspect_dataset import collect_source_frames, load_calibration, load_source_frame, scale_intrinsics
from metrics_compat import masked_psnr, masked_ssim
from reprojection import render_nearest_zbuffer


def _select_frame(frames, requested_id):
    if requested_id is None:
        return frames[0]
    matches = [frame for frame in frames if frame.frame_id == requested_id]
    if not matches:
        raise ValueError(f"frame_{requested_id:06d} is not present")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Endoscope 2 -> Endoscope 2 identity reprojection gate")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame_id", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mask_source_tools", action="store_true")
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--max_mean_error", type=float, default=1e-4)
    parser.add_argument("--min_coverage", type=float, default=99.99)
    parser.add_argument("--min_psnr", type=float, default=70.0)
    parser.add_argument("--min_ssim", type=float, default=0.9999)
    args = parser.parse_args()

    sequence_dir = args.data_root / args.sequence
    frame = _select_frame(collect_source_frames(sequence_dir), args.frame_id)
    rgb_np, depth_np, tissue_np, (scale_x, scale_y) = load_source_frame(
        frame, load_tool_mask=args.mask_source_tools
    )
    calibration = load_calibration(sequence_dir)
    K2 = scale_intrinsics(calibration.K2_L_full, scale_x, scale_y)
    device = torch.device(args.device)
    rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).to(device)
    depth = torch.from_numpy(np.ascontiguousarray(depth_np)).to(device)
    tissue = None if tissue_np is None else torch.from_numpy(np.ascontiguousarray(tissue_np)).to(device)
    K = torch.from_numpy(K2.astype(np.float32)).to(device)
    identity = torch.eye(4, dtype=torch.float32, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = render_nearest_zbuffer(rgb, depth, K, K, identity, depth.shape, source_valid_mask=tissue)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    points = backproject_depth(depth, K)
    uv = project_points(points, K)
    height, width = depth.shape
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    valid_source = torch.isfinite(depth) & (depth > 0)
    if tissue is not None:
        valid_source &= tissue.bool()
    errors = torch.linalg.vector_norm(uv - torch.stack((u, v), dim=-1), dim=-1)[valid_source]

    pred = result.rgb.permute(2, 0, 1).unsqueeze(0)
    gt = rgb.permute(2, 0, 1).unsqueeze(0)
    metric_mask = result.valid_mask.unsqueeze(0).unsqueeze(0).float()
    psnr = float(masked_psnr(pred, gt, metric_mask).item())
    ssim = float(masked_ssim(pred, gt, metric_mask).item())
    report = {
        "status": "PASS",
        "sequence": args.sequence,
        "frame": f"frame_{frame.frame_id:06d}",
        "device": str(device),
        "mean_reprojection_error_pixels": float(errors.mean().item()),
        "median_reprojection_error_pixels": float(errors.median().item()),
        "valid_pixel_percentage": float(valid_source.float().mean().item() * 100.0),
        "rendered_coverage_percentage": float(result.valid_mask.float().mean().item() * 100.0),
        "psnr": psnr,
        "ssim": ssim,
        "time_seconds": elapsed,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
        ),
        **result.stats,
    }
    failures = []
    if report["mean_reprojection_error_pixels"] > args.max_mean_error:
        failures.append("mean reprojection error")
    expected_coverage = args.min_coverage if tissue is None else 0.0
    if report["rendered_coverage_percentage"] < expected_coverage:
        failures.append("coverage")
    if psnr < args.min_psnr:
        failures.append("PSNR")
    if ssim < args.min_ssim:
        failures.append("SSIM")
    if failures:
        report["status"] = "FAIL"
        report["failed_checks"] = failures

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
