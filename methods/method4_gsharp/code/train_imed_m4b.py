#!/usr/bin/env python3
"""Train controlled Method 4B with M3-derived surface initialization."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm


M4_ROOT = Path(__file__).resolve().parent
REPO_ROOT = M4_ROOT.parent
GSPLAT_ROOT = M4_ROOT / "third_party" / "gsplat"
for path in (REPO_ROOT, GSPLAT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.dynamic_surgical_trainer import (  # noqa: E402
    Config,
    build_deform_modules,
    train_step,
)
from gsplat.contrib.dynamic import DynamicStrategy  # noqa: E402
from gsplat.training import TwoStageScheduler  # noqa: E402
from method4_gsharp.datasets.imed_nvs import IMEDNVSDataset, IMEDNVSParser  # noqa: E402
from method4_gsharp.surface_initialization import (  # noqa: E402
    SurfaceInitialization,
    build_surface_initialization,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Method 4B geometry-bootstrapped G-SHARP-iMED"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coarse-steps", type=int, default=500)
    parser.add_argument("--fine-steps", type=int, default=3000)
    parser.add_argument("--init-max-points", type=int, default=50_000)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--thickness-ratio", type=float, default=0.2)
    parser.add_argument("--candidate-multiplier", type=float, default=4.0)
    parser.add_argument("--world-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _jsonable_config(cfg: Config) -> dict[str, object]:
    result = asdict(cfg)
    result["data_dir"] = str(result["data_dir"])
    result["output_dir"] = str(result["output_dir"])
    return result


def _build_splats(
    initialization: SurfaceInitialization,
    cfg: Config,
) -> tuple[nn.ParameterDict, dict[str, torch.optim.Optimizer]]:
    count = int(initialization.means.shape[0])
    if count < 4:
        raise RuntimeError(f"Only {count} valid M4-B initialization points")
    opacity_logit = math.log(cfg.init_opacity / (1.0 - cfg.init_opacity))
    params = nn.ParameterDict(
        {
            "means": nn.Parameter(initialization.means.contiguous()),
            "quats": nn.Parameter(initialization.quats.contiguous()),
            "scales": nn.Parameter(initialization.log_scales.contiguous()),
            "opacities": nn.Parameter(
                torch.full(
                    (count, 1),
                    opacity_logit,
                    dtype=initialization.means.dtype,
                    device=initialization.means.device,
                )
            ),
            "colors": nn.Parameter(initialization.colors.contiguous()),
        }
    )
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=cfg.lr_means),
        "quats": torch.optim.Adam([params["quats"]], lr=cfg.lr_quats),
        "scales": torch.optim.Adam([params["scales"]], lr=cfg.lr_scales),
        "opacities": torch.optim.Adam([params["opacities"]], lr=cfg.lr_opacities),
        "colors": torch.optim.Adam([params["colors"]], lr=cfg.lr_colors),
    }
    return params, optimizers


def main() -> None:
    args = _args()
    output_dir = args.output_dir / args.sequence
    checkpoint_path = output_dir / "checkpoints" / "m4b_final.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M4-B training requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    sequence_dir = args.data_root / args.sequence
    parser = IMEDNVSParser(sequence_dir, world_scale=args.world_scale)
    dataset = IMEDNVSDataset(parser)
    cfg = Config(
        data_dir=sequence_dir,
        output_dir=output_dir,
        coarse_steps=args.coarse_steps,
        fine_steps=args.fine_steps,
        init_max_points=args.init_max_points,
        seed=args.seed,
        device=str(device),
    )

    start = time.perf_counter()
    initialization = build_surface_initialization(
        dataset=dataset,
        device=device,
        max_points=cfg.init_max_points,
        voxel_size=args.voxel_size,
        thickness_ratio=args.thickness_ratio,
        candidate_multiplier=args.candidate_multiplier,
        seed=args.seed,
    )
    init_statistics = initialization.statistics
    initial_metadata = {
        "geometry_confidence": initialization.confidence.detach().cpu(),
        "source_time": initialization.times.detach().cpu(),
        "surface_normal_world": initialization.normals.detach().cpu(),
    }
    params, optimizers = _build_splats(initialization, cfg)
    del initialization

    with torch.no_grad():
        initial_abs_max = float(params["means"].abs().max().item())
    derived_bounds = max(cfg.hex_bounds, initial_abs_max * 2.0)
    hexplane, deform_net, hex_optimizer, deform_optimizer = build_deform_modules(
        cfg, device, bounds_override=derived_bounds
    )
    strategy = DynamicStrategy()
    strategy.check_sanity(params, optimizers)
    strategy_scene_scale = float(init_statistics["scene_extent_max"])
    if not math.isfinite(strategy_scene_scale) or strategy_scene_scale <= 0.0:
        raise RuntimeError(
            f"Invalid M4-B initialization scene extent: {strategy_scene_scale}"
        )
    print(
        "[strategy_scene_scale] "
        f"source=initial_xyz_max_axis_extent, value={strategy_scene_scale:.6f}"
    )
    print("[m4b_initialization] " + json.dumps(init_statistics, sort_keys=True))
    strategy_state = strategy.initialize_state(
        scene_scale=strategy_scene_scale,
        num_gaussians=len(params["means"]),
        device=device,
    )
    scheduler = TwoStageScheduler(cfg.coarse_steps, cfg.fine_steps)
    total_steps = cfg.coarse_steps + cfg.fine_steps
    curves_path = output_dir / "training_curve.jsonl"
    with curves_path.open("w", encoding="utf-8") as curves:
        progress = tqdm.tqdm(range(total_steps), desc="M4-B surface G-SHARP")
        for step in progress:
            schedule = scheduler.step(step, len(dataset))
            losses = train_step(
                cfg=cfg,
                item=dataset[schedule.frame_index],
                params=params,
                optimizers=optimizers,
                hexplane=hexplane,
                deform_net=deform_net,
                hex_optimizer=hex_optimizer,
                deform_optimizer=deform_optimizer,
                strategy=strategy,
                state=strategy_state,
                step=step,
            )
            row = {
                "step": step,
                "stage": schedule.stage,
                "frame_index": schedule.frame_index,
                "gaussians": len(params["means"]),
                **losses,
            }
            curves.write(json.dumps(row) + "\n")
            if step % 50 == 0:
                curves.flush()
                progress.set_postfix(
                    loss=f"{losses['loss']:.5f}", n=len(params["means"])
                )

    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - start
    peak_vram_bytes = int(torch.cuda.max_memory_allocated(device))
    commit = (M4_ROOT / "GSHARP_GSPLAT_COMMIT.txt").read_text().strip()
    checkpoint = {
        # Keep the renderer-compatible base schema so M4-A and M4-B use the
        # exact same frozen rendering path. method_variant distinguishes it.
        "format": "m4a-gsharp-imed-v1",
        "method_variant": "M4-B geometry-bootstrapped G-SHARP",
        "sequence": args.sequence,
        "gsplat_commit": commit,
        "config": _jsonable_config(cfg),
        "adapter": parser.summary(),
        "initialization": {
            "type": "m3_source_surface_voxel_fusion",
            "voxel_size": args.voxel_size,
            "thickness_ratio": args.thickness_ratio,
            "candidate_multiplier": args.candidate_multiplier,
            "opacity": cfg.init_opacity,
        },
        "derived_hex_bounds": derived_bounds,
        "strategy_scene_scale": strategy_scene_scale,
        "strategy_scene_scale_source": "initial_xyz_max_axis_extent",
        "initial_statistics": init_statistics,
        "initial_surface_metadata": initial_metadata,
        "initial_gaussians": init_statistics["count"],
        "final_gaussians": len(params["means"]),
        "training_runtime_seconds": runtime_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "params": {key: value.detach().cpu() for key, value in params.items()},
        "hexplane": {
            key: value.detach().cpu() for key, value in hexplane.state_dict().items()
        },
        "deform_net": {
            key: value.detach().cpu() for key, value in deform_net.state_dict().items()
        },
        "dynamic_mask": strategy_state["dynamic_mask"].detach().cpu(),
        "optimizer_states": {
            key: value.state_dict() for key, value in optimizers.items()
        },
        "hex_optimizer_state": hex_optimizer.state_dict(),
        "deform_optimizer_state": deform_optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "status": "trained_not_yet_evaluated",
        "method": "M4-B geometry-bootstrapped G-SHARP",
        "controlled_difference_from_m4a": "Gaussian initialization only",
        "sequence": args.sequence,
        "gsplat_commit": commit,
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "adapter": parser.summary(),
        "initialization": checkpoint["initialization"],
        "initial_statistics": init_statistics,
        "hex_bounds": derived_bounds,
        "strategy_scene_scale": strategy_scene_scale,
        "strategy_scene_scale_source": "initial_xyz_max_axis_extent",
        "training_steps": total_steps,
        "coarse_steps": cfg.coarse_steps,
        "fine_steps": cfg.fine_steps,
        "initial_gaussians": init_statistics["count"],
        "final_gaussians": len(params["means"]),
        "training_runtime_seconds": runtime_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "checkpoint": str(checkpoint_path),
        "endoscope1_rgb_access": False,
        "target_camera_access_during_initialization": False,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
