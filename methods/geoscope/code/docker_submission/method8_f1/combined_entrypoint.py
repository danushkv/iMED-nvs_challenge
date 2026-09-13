#!/usr/bin/env python3
"""Frozen Method 8: M3B with strict Method-2 fallback on M3B holes only."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from f1_fusion import strict_hole_only_fusion


M3B_ROOT = Path("/app/method8_m3b")
METHOD2_ROOT = Path("/workspace/Endo-4DGS")
METHOD2_RUNNER = METHOD2_ROOT / "imed_nvs_baseline.py"
sys.path.insert(0, str(M3B_ROOT))

from hole_filling import nearest_valid_fill  # noqa: E402
from imed_io import (  # noqa: E402
    collect_source_frames,
    load_calibration,
    load_source_frame,
    scale_intrinsics,
)
from surface_splat import render_surface_splat  # noqa: E402


# Frozen M3B configuration.
VISIBILITY_TOLERANCE_MM = 1.0
VISIBILITY_RELATIVE = 0.01
DEPTH_SOFTNESS = 8.0
FILL_RADIUS = 3
FOOTPRINT_SCALE = 1.0
RADIUS_MIN = 0.5
RADIUS_MAX = 2.0
SURFACE_SUPPORT = 5
MAHALANOBIS_CUTOFF = 9.0
DEPTH_DISCONTINUITY_MM = 2.0
DEPTH_DISCONTINUITY_RELATIVE = 0.02
CHUNK_SIZE = 65536

# Frozen accepted Method-2 configuration.
METHOD2_ITERATIONS = 1000
METHOD2_COARSE_ITERATIONS = 300
METHOD2_DEPTH_LOSS = "metric_l1"
METHOD2_DEPTH_WEIGHT = "5e-5"


def _is_sequence_dir(path: Path) -> bool:
    """Recognize a sequence using permitted source-side inputs only."""

    return (
        path.is_dir()
        and (path / "K.txt").is_file()
        and (path / "pose.txt").is_file()
        and (path / "endoscope2" / "L").is_dir()
        and (path / "endoscope2" / "depthL").is_dir()
        and (path / "endoscope2" / "toolL").is_dir()
    )


def _discover_sequences(input_root: Path) -> list[Path]:
    """Discover sequences while explicitly pruning every target-view tree."""

    if _is_sequence_dir(input_root):
        return [input_root]
    discovered: list[Path] = []
    pending = [input_root]
    while pending:
        directory = pending.pop()
        if _is_sequence_dir(directory):
            discovered.append(directory)
            continue
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name, reverse=True)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for entry in entries:
            if entry.name == "endoscope1":
                continue
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
    return sorted(discovered)


def _source_frame_names(sequence: Path) -> list[str]:
    names = sorted(path.name for path in (sequence / "endoscope2" / "L").glob("frame_*.png"))
    if not names:
        raise FileNotFoundError(f"No Endoscope-2 RGB frames in {sequence / 'endoscope2' / 'L'}")
    return names


def _requested_target_names(sequence: Path) -> list[str]:
    """Use permitted frame-name metadata, otherwise synchronized source ids."""

    source_names = _source_frame_names(sequence)
    target_list = sequence / "target_frames.txt"
    if not target_list.is_file():
        return source_names
    names = [line.strip() for line in target_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Target frame list is empty: {target_list}")
    normalized = [name if Path(name).suffix else f"{name}.png" for name in names]
    if [Path(name).stem for name in normalized] != [Path(name).stem for name in source_names]:
        raise ValueError(
            "Method 2 requires synchronized one-to-one source/target frame ids; "
            f"target_frames.txt does not match Endoscope-2 RGB in {sequence}"
        )
    return normalized


def _safe_symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite internal input view: {destination}")
    os.symlink(source, destination, target_is_directory=source.is_dir())


def _make_source_only_view(sequence: Path, staging_root: Path) -> Path:
    """Expose only legal source inputs to the unchanged Method-2 runner."""

    staged = staging_root / sequence.name
    staged.mkdir(parents=True)
    _safe_symlink(sequence / "K.txt", staged / "K.txt")
    _safe_symlink(sequence / "pose.txt", staged / "pose.txt")
    staged_source = staged / "endoscope2"
    staged_source.mkdir()
    for stream in ("L", "depthL", "toolL"):
        _safe_symlink(sequence / "endoscope2" / stream, staged_source / stream)

    # Method 2 needs target camera records, never target pixels. The accepted
    # source-only adapter constructs those records from K1_L/camera id 1.
    (staged / "endoscope1").mkdir()
    (staged / "target_frames.txt").write_text(
        "".join(f"{name}\n" for name in _requested_target_names(sequence)),
        encoding="utf-8",
    )
    return staged


def _to_native_rgb(rgb_internal: torch.Tensor, native_size_wh: tuple[int, int]) -> np.ndarray:
    """Frozen M3B bilinear export path."""

    native_width, native_height = native_size_wh
    rgb_nchw = rgb_internal.permute(2, 0, 1).unsqueeze(0)
    if tuple(rgb_internal.shape[:2]) != (native_height, native_width):
        rgb_nchw = F.interpolate(
            rgb_nchw,
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )
    return (
        rgb_nchw[0]
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


@torch.inference_mode()
def _render_m3b(sequence: Path, private_root: Path, device: torch.device) -> tuple[list[str], tuple[int, int]]:
    """Render frozen M3B and retain its true radius-3 support mask privately."""

    frames = collect_source_frames(sequence)
    calibration = load_calibration(sequence)
    first = load_source_frame(frames[0])
    depth_height, depth_width = first.depth.shape
    native_size_wh = first.native_rgb_size_wh
    internal_size_wh = (depth_width, depth_height)
    expected_names = [f"{index:05d}.png" for index in range(len(frames))]

    render_dir = private_root / "renders"
    mask_dir = private_root / "filled_valid_mask_internal"
    render_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    K_source = torch.from_numpy(
        scale_intrinsics(calibration.K_source_full, native_size_wh, internal_size_wh).astype(np.float32)
    ).to(device)
    K_target = torch.from_numpy(
        scale_intrinsics(calibration.K_target_full, native_size_wh, internal_size_wh).astype(np.float32)
    ).to(device)
    T_target_source = torch.from_numpy(calibration.T_target_source.astype(np.float32)).to(device)

    started = time.perf_counter()
    filled_coverage_sum = 0.0
    for index, frame in enumerate(frames):
        loaded = first if index == 0 else load_source_frame(frame)
        if loaded.depth.shape != (depth_height, depth_width):
            raise ValueError(f"Depth resolution changed at {frame.depth_path}")
        if loaded.native_rgb_size_wh != native_size_wh:
            raise ValueError(f"Source RGB resolution changed at {frame.rgb_path}")

        source_rgb = torch.from_numpy(np.ascontiguousarray(loaded.rgb_at_depth_resolution)).to(
            device=device, dtype=torch.float32
        )
        source_depth = torch.from_numpy(np.ascontiguousarray(loaded.depth)).to(
            device=device, dtype=torch.float32
        )
        surface_output = render_surface_splat(
            source_rgb=source_rgb,
            source_depth=source_depth,
            K_source=K_source,
            K_target=K_target,
            T_target_source=T_target_source,
            target_size=(depth_height, depth_width),
            source_valid_mask=None,
            visibility_tolerance_mm=VISIBILITY_TOLERANCE_MM,
            visibility_relative=VISIBILITY_RELATIVE,
            depth_softness=DEPTH_SOFTNESS,
            depth_discontinuity_mm=DEPTH_DISCONTINUITY_MM,
            depth_discontinuity_relative=DEPTH_DISCONTINUITY_RELATIVE,
            footprint_scale=FOOTPRINT_SCALE,
            radius_min=RADIUS_MIN,
            radius_max=RADIUS_MAX,
            support_size=SURFACE_SUPPORT,
            mahalanobis_cutoff=MAHALANOBIS_CUTOFF,
            chunk_size=CHUNK_SIZE,
            confidence_aware=False,
        )
        raw = surface_output.result
        filled = nearest_valid_fill(
            raw.rgb,
            raw.depth,
            raw.valid_mask,
            raw.confidence,
            radius=FILL_RADIUS,
        )
        name = expected_names[index]
        Image.fromarray(_to_native_rgb(filled.rgb, native_size_wh), mode="RGB").save(render_dir / name)
        mask = filled.filled_valid_mask.to(dtype=torch.uint8).mul(255).cpu().numpy()
        Image.fromarray(mask, mode="L").save(mask_dir / name)
        filled_coverage_sum += float(filled.filled_valid_mask.float().mean().item() * 100.0)

    elapsed = time.perf_counter() - started
    print(
        f"[METHOD8:M3B] sequence={sequence.name} frames={len(frames)} "
        f"internal_size={depth_width}x{depth_height} "
        f"native_size={native_size_wh[0]}x{native_size_wh[1]} "
        f"mean_filled_coverage={filled_coverage_sum / len(frames):.2f}% "
        f"runtime_seconds={elapsed:.3f}",
        flush=True,
    )
    return expected_names, native_size_wh


def _run_method2(sequence: Path, staging_root: Path, private_output: Path) -> None:
    staged_sequence = _make_source_only_view(sequence, staging_root)
    command = [
        sys.executable,
        str(METHOD2_RUNNER),
        "--repo",
        str(METHOD2_ROOT),
        "run-sequence",
        "--sequence",
        str(staged_sequence),
        "--output",
        str(private_output),
        "--iterations",
        str(METHOD2_ITERATIONS),
        "--coarse-iterations",
        str(METHOD2_COARSE_ITERATIONS),
        "--depth-loss",
        METHOD2_DEPTH_LOSS,
        "--depth-weight",
        METHOD2_DEPTH_WEIGHT,
        "--no-metrics",
    ]
    started = time.perf_counter()
    subprocess.run(command, cwd=METHOD2_ROOT, check=True)
    print(
        f"[METHOD8:M2] sequence={sequence.name} runtime_seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )


def _five_digit_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9].png"))


def _load_rgb(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        if image.size != expected_size:
            raise ValueError(f"Expected RGB size {expected_size}, found {image.size}: {path}")
        return np.asarray(image, dtype=np.uint8).copy()


def _load_internal_mask(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        values = np.asarray(opened.convert("L"), dtype=np.uint8)
    if not np.all(np.isin(np.unique(values), (0, 255))):
        raise ValueError(f"M3B support mask is not binary: {path}")
    return values > 127


def _native_valid_mask(mask_internal: np.ndarray, native_size_wh: tuple[int, int]) -> np.ndarray:
    native_width, native_height = native_size_wh
    return (
        F.interpolate(
            torch.from_numpy(np.ascontiguousarray(mask_internal))[None, None].to(torch.float32),
            size=(native_height, native_width),
            mode="nearest",
        )[0, 0]
        .bool()
        .numpy()
    )


def _fuse_strict_f1(
    sequence: Path,
    expected_names: list[str],
    native_size_wh: tuple[int, int],
    m3b_root: Path,
    method2_root: Path,
    private_final_root: Path,
) -> int:
    """Copy Method 2 only where the resized M3B support mask is false."""

    m3b_paths = _five_digit_paths(m3b_root / "renders")
    method2_paths = _five_digit_paths(method2_root / "renders")
    mask_paths = _five_digit_paths(m3b_root / "filled_valid_mask_internal")
    for label, paths in (("M3B", m3b_paths), ("Method 2", method2_paths), ("M3B mask", mask_paths)):
        names = [path.name for path in paths]
        if names != expected_names:
            raise ValueError(
                f"{label} stream mismatch for {sequence.name}: "
                f"expected={expected_names[:3]}... ({len(expected_names)}), "
                f"found={names[:3]}... ({len(names)})"
            )

    final_render_dir = private_final_root / "renders"
    final_render_dir.mkdir(parents=True)
    replaced_total = 0
    for name, m3b_path, method2_path, mask_path in zip(
        expected_names, m3b_paths, method2_paths, mask_paths
    ):
        m3b = _load_rgb(m3b_path, native_size_wh)
        method2 = _load_rgb(method2_path, native_size_wh)
        valid = _native_valid_mask(_load_internal_mask(mask_path), native_size_wh)
        final, fallback_count = strict_hole_only_fusion(m3b, method2, valid)

        # Critical F1 invariant: no valid M3B pixel may change by even one byte.
        if not np.array_equal(final[valid], m3b[valid]):
            raise AssertionError(f"M3B-valid bytes changed at {sequence.name}/{name}")
        changed = np.any(final != m3b, axis=2)
        if bool((changed & valid).any()):
            raise AssertionError(f"A changed pixel overlaps M3B validity at {sequence.name}/{name}")
        replaced_total += fallback_count
        Image.fromarray(final, mode="RGB").save(final_render_dir / name)

    return replaced_total


def _publish_sequence(private_final: Path, public_sequence: Path) -> None:
    if public_sequence.exists():
        if any(public_sequence.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {public_sequence}")
        public_sequence.rmdir()
    shutil.copytree(private_final, public_sequence)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen Method-8 F1 iMED-NVS submission")
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sequences = _discover_sequences(args.input)
    if not sequences:
        raise SystemExit(f"No iMED-NVS sequences found under {args.input}")
    if not torch.cuda.is_available():
        raise RuntimeError("Method 8 requires a CUDA GPU")
    device = torch.device("cuda")
    print(
        "[METHOD8] variant=F1_strict_holes "
        f"sequences={len(sequences)} device={torch.cuda.get_device_name(device)} "
        "m3b_renderer=surface m3b_fill_radius=3 "
        f"method2_iterations={METHOD2_ITERATIONS} "
        f"method2_coarse_iterations={METHOD2_COARSE_ITERATIONS} "
        f"method2_depth_loss={METHOD2_DEPTH_LOSS} "
        f"method2_depth_weight={METHOD2_DEPTH_WEIGHT} metrics=disabled",
        flush=True,
    )

    total_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="method8-f1-", dir="/tmp") as temporary:
        private_root = Path(temporary)
        for sequence in sequences:
            sequence_started = time.perf_counter()
            work = private_root / sequence.name
            m3b_root = work / "m3b"
            method2_root = work / "method2"
            staging_root = work / "source_only"
            private_final = work / "final"
            work.mkdir(parents=True)

            expected_names, native_size_wh = _render_m3b(sequence, m3b_root, device)
            torch.cuda.empty_cache()
            _run_method2(sequence, staging_root, method2_root)
            replaced = _fuse_strict_f1(
                sequence,
                expected_names,
                native_size_wh,
                m3b_root,
                method2_root,
                private_final,
            )
            _publish_sequence(private_final, args.output / sequence.name)
            elapsed = time.perf_counter() - sequence_started
            print(
                f"[METHOD8] completed sequence={sequence.name} frames={len(expected_names)} "
                f"fallback_pixels={replaced} m3b_valid_pixels_changed=0 "
                f"runtime_seconds={elapsed:.3f} output={args.output / sequence.name / 'renders'}",
                flush=True,
            )
            # Remove checkpoints and all private M3B/Method-2 intermediates as
            # soon as the final sequence output has been safely published.
            shutil.rmtree(work)

    print(
        f"[METHOD8] total_runtime_seconds={time.perf_counter() - total_started:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
