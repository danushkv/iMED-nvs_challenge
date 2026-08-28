#!/usr/bin/env python3
"""iMED-NVS Endo-4DGS baseline entrypoint."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


IMED_CONFIG = """ModelParams = dict(
    camera_extent=10,
    use_pretrain=True
)

OptimizationParams = dict(
    coarse_iterations={coarse_iterations},
    deformation_lr_init=0.00016,
    deformation_lr_final=0.0000016,
    deformation_lr_delay_mult=0.01,
    grid_lr_init=0.0016,
    grid_lr_final=0.000016,
    iterations={iterations},
    percent_dense=0.01,
    render_process=True,
    densify_until_iter=3000,
    pruning_from_iter=500,
    densify_from_iter=500,
    densification_interval=100,
    pruning_interval=100,
    opacity_reset_interval=9000,
    # ===== METHOD2: SOURCE DSSIM =====
    lambda_dssim={ssim_weight},
)

ModelHiddenParams = dict(
    kplanes_config={{
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 75]
    }},
    multires=[1, 2],
    defor_depth=0,
    net_width=64,
    plane_tv_weight=0.0001,
    time_smoothness_weight=0.01,
    l1_time_planes=0.0001,
    weight_decay_iteration=0,
    bounds=1.6,
    pool_list=[2],
    multi_scale=False,
    # ===== METHOD2: METRIC DEPTH =====
    depth_loss='{depth_loss}',
    primary_depth_weight={depth_weight},
    depth_huber_beta={depth_huber_beta},
    depth_diagnostics_interval={depth_diagnostics_interval},
    # ===== METHOD2: TOOL MASK =====
    tool_aware_loss={tool_aware_loss},
    tool_mask_dilation={tool_mask_dilation},
    # ===== METHOD2: APPEARANCE CORRECTION =====
    appearance_correction={appearance_correction},
    appearance_lr={appearance_lr},
    appearance_reg_weight={appearance_reg_weight},
    appearance_diagnostics_interval={appearance_diagnostics_interval}
)

PipelineParams = dict(
    use_depth=True,
    use_smooth=True,
    use_normal=True,
    use_confidence={use_confidence}
)
"""


def run(cmd: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _image_files(path: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def sequence_name(sequence: Path) -> str:
    return sequence.resolve().name


def is_sequence_dir(path: Path) -> bool:
    return (
        (path / "pose.txt").is_file()
        and (path / "K.txt").is_file()
        and (path / "endoscope1").is_dir()
        and (path / "endoscope2").is_dir()
    )


def discover_sequences(data_root: Path) -> list[Path]:
    if is_sequence_dir(data_root):
        return [data_root]
    return sorted(path for path in data_root.rglob("*") if path.is_dir() and is_sequence_dir(path))


def write_config(
    output_dir: Path,
    iterations: int,
    coarse_iterations: int,
    depth_loss: str = "normalized",
    depth_weight: float = 0.01,
    depth_huber_beta: float = 5.0,
    depth_diagnostics_interval: int = 0,
    tool_aware_loss: bool = False,
    tool_mask_dilation: int = 0,
    appearance_correction: bool = False,
    appearance_lr: float = 0.001,
    appearance_reg_weight: float = 0.01,
    appearance_diagnostics_interval: int = 0,
    ssim_weight: float = 0.0,
    use_confidence: bool = True,
) -> Path:
    config_path = output_dir / "imed_runtime_config.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        IMED_CONFIG.format(
            iterations=iterations,
            coarse_iterations=coarse_iterations,
            depth_loss=depth_loss,
            depth_weight=depth_weight,
            depth_huber_beta=depth_huber_beta,
            depth_diagnostics_interval=depth_diagnostics_interval,
            tool_aware_loss=tool_aware_loss,
            tool_mask_dilation=tool_mask_dilation,
            appearance_correction=appearance_correction,
            appearance_lr=appearance_lr,
            appearance_reg_weight=appearance_reg_weight,
            appearance_diagnostics_interval=appearance_diagnostics_interval,
            ssim_weight=ssim_weight,
            use_confidence=use_confidence,
        ),
        encoding="utf-8",
    )
    return config_path


def _safe_link(src: Path, dst: Path) -> None:
    """Create a symlink for immutable input data without copying large frames."""

    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        raise FileExistsError(f"Refusing to overwrite existing work item: {dst}")

    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_file():
            shutil.copy2(src, dst)
            return
        raise


def prepare_writable_sequence_view(sequence: Path, output: Path) -> Path:
    """Expose a read-only iMED sequence through a writable work directory."""

    work_sequence = output / "_input_sequence"
    work_sequence.mkdir(parents=True, exist_ok=True)
    for name in ("pose.txt", "K.txt", "endoscope1", "endoscope2"):
        _safe_link(sequence / name, work_sequence / name)
    target_frames = sequence / "target_frames.txt"
    if target_frames.is_file():
        _safe_link(target_frames, work_sequence / "target_frames.txt")
    return work_sequence


def _infer_target_size(sequence: Path) -> tuple[int, int]:
    source_frames = _image_files(sequence / "endoscope2" / "L")
    if not source_frames:
        raise FileNotFoundError(f"No source RGB frames found under {sequence / 'endoscope2' / 'L'}")
    with Image.open(source_frames[0]) as image:
        return image.size


def _latest_render_dir(output: Path) -> Path:
    def iteration_key(path: Path) -> int:
        try:
            return int(path.parent.name.split("_")[-1])
        except (IndexError, ValueError):
            return -1

    candidates = sorted((output / "test").glob("ours_*/renders"), key=iteration_key)
    if not candidates:
        raise FileNotFoundError(f"No Endo-4DGS test render directory found under {output / 'test'}")
    return candidates[-1]


def export_submission_renders(sequence: Path, output: Path) -> Path:
    """Convert Endo-4DGS native renders into the challenge output contract."""

    source_render_dir = _latest_render_dir(output)
    source_frames = _image_files(source_render_dir)
    if not source_frames:
        raise FileNotFoundError(f"No rendered images found in {source_render_dir}")

    target_size = _infer_target_size(sequence)
    submission_render_dir = output / "renders"
    if submission_render_dir.exists():
        shutil.rmtree(submission_render_dir)
    submission_render_dir.mkdir(parents=True, exist_ok=True)

    for index, src in enumerate(source_frames):
        with Image.open(src) as image:
            image = image.convert("RGB")
            if image.size != target_size:
                image = image.resize(target_size, Image.BILINEAR)
            image.save(submission_render_dir / f"{index:05d}.png")

    print(
        f"[EXPORT] Wrote {len(source_frames)} challenge render(s) to "
        f"{submission_render_dir} at {target_size[0]}x{target_size[1]}",
        flush=True,
    )
    return submission_render_dir


def run_sequence(
    sequence: Path,
    output: Path,
    repo: Path,
    iterations: int = 1000,
    coarse_iterations: int = 300,
    port: int = 6017,
    render: bool = True,
    metrics: bool = True,
    depth_loss: str = "normalized",
    depth_weight: float = 0.01,
    depth_huber_beta: float = 5.0,
    depth_diagnostics_interval: int = 0,
    tool_aware_loss: bool = False,
    tool_mask_dilation: int = 0,
    appearance_correction: bool = False,
    appearance_lr: float = 0.001,
    appearance_reg_weight: float = 0.01,
    appearance_diagnostics_interval: int = 0,
    ssim_weight: float = 0.0,
    use_confidence: bool = True,
) -> Path:
    sequence = sequence.resolve()
    output = output.resolve()
    repo = repo.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if depth_loss not in {"normalized", "metric_l1", "metric_huber"}:
        raise ValueError(f"Unsupported depth loss: {depth_loss}")
    if depth_weight < 0:
        raise ValueError("depth_weight must be non-negative")
    if depth_huber_beta <= 0:
        raise ValueError("depth_huber_beta must be positive")
    if depth_diagnostics_interval < 0:
        raise ValueError("depth_diagnostics_interval must be non-negative")
    if tool_mask_dilation not in {0, 3, 5}:
        raise ValueError("tool_mask_dilation must be one of: 0, 3, 5")
    if appearance_lr <= 0:
        raise ValueError("appearance_lr must be positive")
    if appearance_reg_weight < 0:
        raise ValueError("appearance_reg_weight must be non-negative")
    if appearance_diagnostics_interval < 0:
        raise ValueError("appearance_diagnostics_interval must be non-negative")
    if not 0.0 <= ssim_weight <= 1.0:
        raise ValueError("ssim_weight must be in [0, 1]")
    if metrics and not render:
        raise ValueError("metrics require rendering; combine --no-render with --no-metrics")

    config = write_config(
        output,
        iterations=iterations,
        coarse_iterations=coarse_iterations,
        depth_loss=depth_loss,
        depth_weight=depth_weight,
        depth_huber_beta=depth_huber_beta,
        depth_diagnostics_interval=depth_diagnostics_interval,
        tool_aware_loss=tool_aware_loss,
        tool_mask_dilation=tool_mask_dilation,
        appearance_correction=appearance_correction,
        appearance_lr=appearance_lr,
        appearance_reg_weight=appearance_reg_weight,
        appearance_diagnostics_interval=appearance_diagnostics_interval,
        ssim_weight=ssim_weight,
        use_confidence=use_confidence,
    )
    work_sequence = prepare_writable_sequence_view(sequence, output)

    train_cmd = [
        sys.executable,
        "train.py",
        "-s",
        str(work_sequence),
        "--model_path",
        str(output),
        "--expname",
        f"imed_nvs/{sequence_name(sequence)}",
        "--port",
        str(port),
        "--configs",
        str(config),
    ]
    run(train_cmd, cwd=repo)

    if render:
        render_cmd = [
            sys.executable,
            "render.py",
            "--model_path",
            str(output),
            "--skip_train",
            "--skip_video",
            "--configs",
            str(config),
        ]
        run(render_cmd, cwd=repo)

    if metrics:
        metrics_cmd = [sys.executable, "metrics.py", "-m", str(output)]
        run(metrics_cmd, cwd=repo)

    if render:
        return export_submission_renders(sequence, output)
    # ===== METHOD2: DIAGNOSTIC RUN SUPPORT =====
    # B1 magnitude checks intentionally stop after training. The original
    # adapter attempted to export nonexistent renders in this mode.
    return output


def new_view(sequence_path: str, output_dir: str, **kwargs) -> str:
    """Train on endoscope2 and render held-out endoscope1 views for one sequence."""

    repo = Path(kwargs.pop("repo", os.environ.get("ENDO4DGS_REPO", "/workspace/Endo-4DGS")))
    test_dir = run_sequence(Path(sequence_path), Path(output_dir), repo=repo, **kwargs)
    return str(test_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="iMED-NVS Endo-4DGS baseline")
    parser.add_argument("--repo", default=os.environ.get("ENDO4DGS_REPO", "/workspace/Endo-4DGS"))
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("run-sequence", help="Train/render/evaluate one iMED-NVS sequence")
    one.add_argument("--sequence", required=True)
    one.add_argument("--output", required=True)
    one.add_argument("--iterations", type=int, default=1000)
    one.add_argument("--coarse-iterations", type=int, default=300)
    one.add_argument("--port", type=int, default=6017)
    one.add_argument("--no-render", action="store_true")
    one.add_argument("--no-metrics", action="store_true")
    # ===== METHOD2: METRIC DEPTH =====
    one.add_argument(
        "--depth-loss",
        choices=("normalized", "metric_l1", "metric_huber"),
        default="normalized",
    )
    one.add_argument("--depth-weight", type=float, default=0.01)
    one.add_argument("--depth-huber-beta", type=float, default=5.0)
    one.add_argument("--depth-diagnostics-interval", type=int, default=0)
    # ===== METHOD2: TOOL MASK =====
    one.add_argument("--tool-aware-loss", action="store_true")
    one.add_argument("--tool-mask-dilation", type=int, choices=(0, 3, 5), default=0)
    # ===== METHOD2: APPEARANCE CORRECTION =====
    one.add_argument("--appearance-correction", action="store_true")
    one.add_argument("--appearance-lr", type=float, default=0.001)
    one.add_argument("--appearance-reg-weight", type=float, default=0.01)
    one.add_argument("--appearance-diagnostics-interval", type=int, default=0)
    # ===== METHOD2: SOURCE DSSIM / CONFIDENCE ABLATION =====
    one.add_argument("--ssim-weight", type=float, default=0.0)
    one.add_argument("--disable-confidence", action="store_true")

    many = sub.add_parser("run-dataset", help="Run all detected iMED-NVS sequences under a data root")
    many.add_argument("--data-root", required=True)
    many.add_argument("--output-root", required=True)
    many.add_argument("--iterations", type=int, default=1000)
    many.add_argument("--coarse-iterations", type=int, default=300)
    many.add_argument("--max-sequences", type=int, default=None)
    many.add_argument("--no-render", action="store_true")
    many.add_argument("--no-metrics", action="store_true")
    # ===== METHOD2: METRIC DEPTH =====
    many.add_argument(
        "--depth-loss",
        choices=("normalized", "metric_l1", "metric_huber"),
        default="normalized",
    )
    many.add_argument("--depth-weight", type=float, default=0.01)
    many.add_argument("--depth-huber-beta", type=float, default=5.0)
    many.add_argument("--depth-diagnostics-interval", type=int, default=0)
    # ===== METHOD2: TOOL MASK =====
    many.add_argument("--tool-aware-loss", action="store_true")
    many.add_argument("--tool-mask-dilation", type=int, choices=(0, 3, 5), default=0)
    # ===== METHOD2: APPEARANCE CORRECTION =====
    many.add_argument("--appearance-correction", action="store_true")
    many.add_argument("--appearance-lr", type=float, default=0.001)
    many.add_argument("--appearance-reg-weight", type=float, default=0.01)
    many.add_argument("--appearance-diagnostics-interval", type=int, default=0)
    # ===== METHOD2: SOURCE DSSIM / CONFIDENCE ABLATION =====
    many.add_argument("--ssim-weight", type=float, default=0.0)
    many.add_argument("--disable-confidence", action="store_true")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.repo)

    if args.command == "run-sequence":
        run_sequence(
            Path(args.sequence),
            Path(args.output),
            repo=repo,
            iterations=args.iterations,
            coarse_iterations=args.coarse_iterations,
            port=args.port,
            render=not args.no_render,
            metrics=not args.no_metrics,
            depth_loss=args.depth_loss,
            depth_weight=args.depth_weight,
            depth_huber_beta=args.depth_huber_beta,
            depth_diagnostics_interval=args.depth_diagnostics_interval,
            tool_aware_loss=args.tool_aware_loss,
            tool_mask_dilation=args.tool_mask_dilation,
            appearance_correction=args.appearance_correction,
            appearance_lr=args.appearance_lr,
            appearance_reg_weight=args.appearance_reg_weight,
            appearance_diagnostics_interval=args.appearance_diagnostics_interval,
            ssim_weight=args.ssim_weight,
            use_confidence=not args.disable_confidence,
        )
        return 0

    sequences = discover_sequences(Path(args.data_root))
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise SystemExit(f"No iMED-NVS sequence folders found under {args.data_root}")

    for index, sequence in enumerate(sequences):
        output = Path(args.output_root) / sequence_name(sequence)
        run_sequence(
            sequence,
            output,
            repo=repo,
            iterations=args.iterations,
            coarse_iterations=args.coarse_iterations,
            port=6017 + index,
            render=not args.no_render,
            metrics=not args.no_metrics,
            depth_loss=args.depth_loss,
            depth_weight=args.depth_weight,
            depth_huber_beta=args.depth_huber_beta,
            depth_diagnostics_interval=args.depth_diagnostics_interval,
            tool_aware_loss=args.tool_aware_loss,
            tool_mask_dilation=args.tool_mask_dilation,
            appearance_correction=args.appearance_correction,
            appearance_lr=args.appearance_lr,
            appearance_reg_weight=args.appearance_reg_weight,
            appearance_diagnostics_interval=args.appearance_diagnostics_interval,
            ssim_weight=args.ssim_weight,
            use_confidence=not args.disable_confidence,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
