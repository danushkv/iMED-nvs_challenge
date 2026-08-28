#!/usr/bin/env python3
"""Render legal iMED source or target views from an M4-A checkpoint.

Source mode evaluates the source-reconstruction gate using Endoscope2 RGB-D.
Target mode uses Endoscope1 calibration and timestamps only. This script never
discovers or loads Endoscope1 imagery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
from PIL import Image


M4_ROOT = Path(__file__).resolve().parent
REPO_ROOT = M4_ROOT.parent
GSPLAT_ROOT = M4_ROOT / "third_party" / "gsplat"
for path in (REPO_ROOT, GSPLAT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.dynamic_surgical_trainer import (  # noqa: E402
    Config,
    _deform_with_mask,
    build_deform_modules,
)
from gsplat import rasterization  # noqa: E402
from gsplat.losses import masked_ssim  # noqa: E402
from method4_gsharp.datasets.imed_nvs import IMEDNVSDataset, IMEDNVSParser  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M4-A source-view reconstruction sanity evaluation"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view", choices=("source", "target"), default="source")
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Target render size; omit for native 1280x1024 challenge output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow target render PNGs to be replaced. Source artifacts are resumable.",
    )
    parser.add_argument(
        "--frame-indices",
        type=int,
        nargs="*",
        default=None,
        help="Ordered dataset indices to evaluate; omit to use all source frames.",
    )
    return parser.parse_args()


def _load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    dict[str, object],
    nn.ParameterDict,
    nn.Module,
    nn.Module,
    torch.Tensor,
]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "m4a-gsharp-imed-v1":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format')!r}")

    config_values = dict(checkpoint["config"])
    config_values["device"] = str(device)
    cfg = Config(**config_values)
    params = nn.ParameterDict(
        {
            key: nn.Parameter(value.to(device), requires_grad=False)
            for key, value in checkpoint["params"].items()
        }
    )
    hexplane, deform_net, _, _ = build_deform_modules(
        cfg,
        device,
        bounds_override=float(checkpoint["derived_hex_bounds"]),
    )
    hexplane.load_state_dict(checkpoint["hexplane"], strict=True)
    deform_net.load_state_dict(checkpoint["deform_net"], strict=True)
    hexplane.eval().requires_grad_(False)
    deform_net.eval().requires_grad_(False)

    dynamic_mask = checkpoint["dynamic_mask"].to(device=device, dtype=torch.bool)
    if dynamic_mask.shape != (len(params["means"]),):
        raise ValueError(
            "Checkpoint dynamic mask does not match final Gaussian count: "
            f"{tuple(dynamic_mask.shape)} versus {len(params['means'])}"
        )
    return checkpoint, params, hexplane, deform_net, dynamic_mask


@torch.inference_mode()
def _render_camera(
    camera: dict[str, object],
    height: int,
    width: int,
    params: nn.ParameterDict,
    hexplane: nn.Module,
    deform_net: nn.Module,
    dynamic_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    camtoworld = camera["camtoworld"].to(device).unsqueeze(0)
    K = camera["K"].to(device).unsqueeze(0)
    time_value = camera["time"].to(device).view(1)

    means = params["means"]
    quats = params["quats"]
    opacity_logits = params["opacities"]
    t_per_gaussian = time_value.expand(len(means)).unsqueeze(-1)
    means_d, quats_d, opacity_logits_d = _deform_with_mask(
        means=means,
        quats=quats,
        opacities_logit=opacity_logits,
        t_per_g=t_per_gaussian,
        dynamic_mask=dynamic_mask,
        hexplane=hexplane,
        deform_net=deform_net,
    )
    rendered, alpha, _ = rasterization(
        means=means_d,
        quats=quats_d,
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(opacity_logits_d).squeeze(-1),
        colors=params["colors"],
        viewmats=torch.linalg.inv(camtoworld),
        Ks=K,
        width=width,
        height=height,
        sh_degree=None,
        render_mode="RGB+ED",
        packed=True,
    )
    rgb = rendered[0, ..., :3].clamp(0.0, 1.0)
    depth = rendered[0, ..., 3]
    return rgb, depth, alpha[0, ..., 0]


def _save_rgb(path: Path, image: np.ndarray) -> None:
    encoded = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(path)


def _save_gray(path: Path, image: np.ndarray) -> None:
    encoded = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="L").save(path)


def _depth_visual(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(depth.shape, dtype=np.float32)
    visual = (np.nan_to_num(depth, nan=low, posinf=high, neginf=low) - low) / (
        high - low
    )
    return np.clip(visual, 0.0, 1.0).astype(np.float32)


def _prepare_output(root: Path) -> dict[str, Path]:
    directories = {
        name: root / name
        for name in (
            "gt_source_rgb",
            "rendered_source_rgb",
            "gt_source_depth",
            "rendered_source_depth",
            "rgb_absolute_error",
            "depth_absolute_error",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _render_target_views(
    args: argparse.Namespace,
    parser: IMEDNVSParser,
    indices: list[int],
    checkpoint: dict[str, object],
    params: nn.ParameterDict,
    hexplane: nn.Module,
    deform_net: nn.Module,
    dynamic_mask: torch.Tensor,
    device: torch.device,
) -> None:
    sequence_root = args.output_dir / args.sequence
    renders_dir = sequence_root / "renders"
    rgb_dir = sequence_root / "rgb"
    valid_mask_dir = sequence_root / "valid_mask"
    alpha_dir = sequence_root / "alpha"
    existing = [renders_dir / f"{index:05d}.png" for index in indices]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} target renders already exist under {renders_dir}; "
            "use a fresh --output-dir or explicitly pass --overwrite"
        )
    renders_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    valid_mask_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    frame_rows: list[dict[str, object]] = []
    for index in tqdm.tqdm(indices, desc="M4-A legal target rendering"):
        target = parser.target_camera(index)
        rendered_rgb, _, rendered_alpha = _render_camera(
            target,
            height=int(target["height"]),
            width=int(target["width"]),
            params=params,
            hexplane=hexplane,
            deform_net=deform_net,
            dynamic_mask=dynamic_mask,
            device=device,
        )
        filename = f"{index:05d}.png"
        render_path = renders_dir / filename
        rgb_path = rgb_dir / filename
        _save_rgb(render_path, rendered_rgb.cpu().numpy())
        if rgb_path.exists():
            if not args.overwrite:
                raise FileExistsError(rgb_path)
            rgb_path.unlink()
        os.link(render_path, rgb_path)

        alpha_np = rendered_alpha.clamp(0.0, 1.0).cpu().numpy()
        _save_gray(alpha_dir / filename, alpha_np)
        # This mask is diagnostic only. The frozen M3 evaluator does not add
        # prediction validity to its PSNR/SSIM mask, so uncovered pixels are
        # still penalized identically for all methods.
        valid = (alpha_np > 1e-4).astype(np.uint8) * 255
        Image.fromarray(valid, mode="L").save(valid_mask_dir / filename)
        frame_rows.append(
            {
                "frame_index": index,
                "frame_id": int(target["frame_id"]),
                "render": str(renders_dir / filename),
            }
        )

    torch.cuda.synchronize(device)
    summary = {
        "status": "target_render_complete",
        "sequence": args.sequence,
        "checkpoint": str(args.checkpoint),
        "frames_rendered": len(indices),
        "frame_indices": indices,
        "resolution_wh": [parser.target_width, parser.target_height],
        "gaussians": len(params["means"]),
        "render_runtime_seconds": time.perf_counter() - start,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "target_camera_input": "Endoscope1/L intrinsics and extrinsics only",
        "target_rgb_access": False,
        "validity_convention": "rendered alpha > 1e-4 (diagnostic coverage only)",
        "challenge_render_directory": str(renders_dir),
        "frames": frame_rows,
        "gsplat_commit": checkpoint["gsplat_commit"],
    }
    (sequence_root / "target_render_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    compatibility_summary = {
        "sequence": args.sequence,
        "target_size_hw": [parser.target_height, parser.target_width],
        "filename_convention": "zero-based sorted-stream index, five digits",
        "selected_frame_count": len(indices),
        "target_rgb_access": False,
        "renderer": "M4-A G-SHARP-iMED",
        "validity_convention": "rendered alpha > 1e-4 (diagnostic coverage only)",
        "frames": [
            {
                "sequence_index": row["frame_index"],
                "source_frame_id": row["frame_id"],
            }
            for row in frame_rows
        ],
    }
    (sequence_root / "render_summary.json").write_text(
        json.dumps(compatibility_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "frames"},
            indent=2,
        )
    )


def main() -> None:
    args = _args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M4-A rendering requires CUDA")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    checkpoint, params, hexplane, deform_net, dynamic_mask = _load_checkpoint(
        args.checkpoint, device
    )
    if checkpoint["sequence"] != args.sequence:
        raise ValueError(
            f"Checkpoint sequence {checkpoint['sequence']!r} != {args.sequence!r}"
        )
    world_scale = float(checkpoint["adapter"]["world_scale"])
    target_size_wh = None
    if args.target_size is not None:
        if args.view != "target":
            raise ValueError("--target-size is valid only with --view target")
        if min(args.target_size) <= 0:
            raise ValueError("--target-size dimensions must be positive")
        target_size_wh = (int(args.target_size[0]), int(args.target_size[1]))
    parser = IMEDNVSParser(
        args.data_root / args.sequence,
        world_scale=world_scale,
        target_size_wh=target_size_wh,
    )
    dataset = IMEDNVSDataset(parser)
    indices = list(range(len(dataset))) if args.frame_indices is None else args.frame_indices
    if not indices:
        raise ValueError("No source frames selected")
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError(f"Frame index outside [0, {len(dataset) - 1}]")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate frame indices are not allowed")

    if args.view == "target":
        _render_target_views(
            args=args,
            parser=parser,
            indices=indices,
            checkpoint=checkpoint,
            params=params,
            hexplane=hexplane,
            deform_net=deform_net,
            dynamic_mask=dynamic_mask,
            device=device,
        )
        return

    result_root = args.output_dir / args.sequence / "source_reconstruction"
    paths = _prepare_output(result_root)
    frame_rows: list[dict[str, object]] = []
    rgb_squared_error_sum = 0.0
    rgb_value_count = 0
    depth_absolute_error_sum = 0.0
    depth_gt_count = 0
    depth_pair_absolute_error_sum = 0.0
    depth_pair_count = 0
    start = time.perf_counter()

    for index in tqdm.tqdm(indices, desc="M4-A source reconstruction"):
        item = dataset[index]
        rendered_rgb, rendered_depth, _ = _render_camera(
            item,
            height=int(item["image"].shape[0]),
            width=int(item["image"].shape[1]),
            params=params,
            hexplane=hexplane,
            deform_net=deform_net,
            dynamic_mask=dynamic_mask,
            device=device,
        )
        gt_rgb = item["image"].to(device)
        gt_depth = item["depth"].to(device)
        tissue = item["mask"].to(device) > 0.5

        rgb_diff = rendered_rgb - gt_rgb
        rgb_mask = tissue.unsqueeze(-1).expand_as(rgb_diff)
        frame_rgb_count = int(rgb_mask.sum().item())
        frame_rgb_sse = float(rgb_diff.square()[rgb_mask].sum().item())
        frame_mse = frame_rgb_sse / max(frame_rgb_count, 1)
        frame_psnr = -10.0 * math.log10(max(frame_mse, 1e-12))
        frame_ssim = 1.0 - float(
            masked_ssim(
                rendered_rgb.permute(2, 0, 1).unsqueeze(0),
                gt_rgb.permute(2, 0, 1).unsqueeze(0),
                tissue.unsqueeze(0).unsqueeze(0),
            ).item()
        )
        rgb_squared_error_sum += frame_rgb_sse
        rgb_value_count += frame_rgb_count

        valid_gt_depth = tissue & torch.isfinite(gt_depth) & (gt_depth > 0.0)
        valid_render_depth = torch.isfinite(rendered_depth) & (rendered_depth > 0.0)
        valid_pair = valid_gt_depth & valid_render_depth
        safe_render_depth = torch.nan_to_num(rendered_depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_abs = (safe_render_depth - gt_depth).abs()
        frame_depth_count = int(valid_gt_depth.sum().item())
        frame_pair_count = int(valid_pair.sum().item())
        frame_depth_error = float(depth_abs[valid_gt_depth].sum().item())
        frame_pair_error = float(depth_abs[valid_pair].sum().item())
        frame_depth_mae = frame_depth_error / max(frame_depth_count, 1)
        frame_visible_depth_mae = frame_pair_error / max(frame_pair_count, 1)
        frame_depth_coverage = frame_pair_count / max(frame_depth_count, 1)
        depth_absolute_error_sum += frame_depth_error
        depth_gt_count += frame_depth_count
        depth_pair_absolute_error_sum += frame_pair_error
        depth_pair_count += frame_pair_count

        gt_rgb_np = gt_rgb.cpu().numpy()
        render_rgb_np = rendered_rgb.cpu().numpy()
        gt_depth_np = gt_depth.cpu().numpy().astype(np.float32, copy=False)
        render_depth_np = safe_render_depth.cpu().numpy().astype(np.float32, copy=False)
        depth_abs_np = depth_abs.cpu().numpy().astype(np.float32, copy=False)
        tissue_np = tissue.cpu().numpy()
        stem = f"{index:05d}"
        _save_rgb(paths["gt_source_rgb"] / f"{stem}.png", gt_rgb_np)
        _save_rgb(paths["rendered_source_rgb"] / f"{stem}.png", render_rgb_np)
        _save_gray(
            paths["rgb_absolute_error"] / f"{stem}.png",
            np.abs(render_rgb_np - gt_rgb_np).mean(axis=-1),
        )
        np.save(paths["gt_source_depth"] / f"{stem}.npy", gt_depth_np)
        np.save(paths["rendered_source_depth"] / f"{stem}.npy", render_depth_np)
        np.save(paths["depth_absolute_error"] / f"{stem}.npy", depth_abs_np)

        valid_depth_values = gt_depth_np[tissue_np & np.isfinite(gt_depth_np) & (gt_depth_np > 0)]
        if valid_depth_values.size:
            depth_low, depth_high = np.percentile(valid_depth_values, [1.0, 99.0])
        else:
            depth_low, depth_high = 0.0, 1.0
        valid_errors = depth_abs_np[tissue_np & np.isfinite(depth_abs_np)]
        error_high = float(np.percentile(valid_errors, 99.0)) if valid_errors.size else 1.0
        error_high = max(error_high, 1e-6)
        _save_gray(
            paths["gt_source_depth"] / f"{stem}.png",
            _depth_visual(gt_depth_np, float(depth_low), float(depth_high)),
        )
        _save_gray(
            paths["rendered_source_depth"] / f"{stem}.png",
            _depth_visual(render_depth_np, float(depth_low), float(depth_high)),
        )
        _save_gray(
            paths["depth_absolute_error"] / f"{stem}.png",
            np.clip(depth_abs_np / error_high, 0.0, 1.0),
        )

        frame_rows.append(
            {
                "frame_index": index,
                "frame_id": int(item["frame_id"].item()),
                "psnr": frame_psnr,
                "ssim": frame_ssim,
                "depth_mae": frame_depth_mae,
                "visible_depth_mae": frame_visible_depth_mae,
                "depth_coverage": frame_depth_coverage,
                "depth_visual_min": float(depth_low),
                "depth_visual_max": float(depth_high),
                "depth_error_visual_max": error_high,
            }
        )

    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - start
    overall_mse = rgb_squared_error_sum / max(rgb_value_count, 1)
    summary = {
        "status": "source_reconstruction_complete",
        "sequence": args.sequence,
        "checkpoint": str(args.checkpoint),
        "frames_evaluated": len(indices),
        "frame_indices": indices,
        "gaussians": len(params["means"]),
        "psnr": -10.0 * math.log10(max(overall_mse, 1e-12)),
        "ssim": float(np.mean([row["ssim"] for row in frame_rows])),
        "depth_mae": depth_absolute_error_sum / max(depth_gt_count, 1),
        "visible_depth_mae": depth_pair_absolute_error_sum / max(depth_pair_count, 1),
        "depth_coverage": depth_pair_count / max(depth_gt_count, 1),
        "depth_units": checkpoint["adapter"]["depth_units"],
        "render_runtime_seconds": runtime_seconds,
        "target_rgb_access": False,
        "artifact_directory": str(result_root),
        "frames": frame_rows,
    }
    (result_root / "source_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
