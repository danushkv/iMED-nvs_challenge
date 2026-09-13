#!/usr/bin/env python3
"""Create a compact, deterministic prediction-only GIF from frozen PNG renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--stop", type=int, default=180)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-bytes", type=int, default=9_500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input.glob("[0-9][0-9][0-9][0-9][0-9].png"))
    if not paths:
        raise FileNotFoundError(f"No five-digit PNG renders under {args.input}")
    expected = [f"{index:05d}.png" for index in range(len(paths))]
    if [path.name for path in paths] != expected:
        raise ValueError("Render stream is not contiguous and zero-based")
    if not (0 <= args.start < args.stop <= len(paths)):
        raise ValueError(f"Invalid range [{args.start}, {args.stop}) for {len(paths)} frames")
    if args.stride < 1 or args.width < 1 or args.fps <= 0:
        raise ValueError("stride, width, and fps must be positive")

    selected = paths[args.start : args.stop : args.stride]
    frames: list[Image.Image] = []
    source_size = None
    output_size = None
    for path in selected:
        with Image.open(path) as opened:
            rgb = opened.convert("RGB")
            if source_size is None:
                source_size = rgb.size
                height = round(rgb.height * args.width / rgb.width)
                output_size = (args.width, height)
            elif rgb.size != source_size:
                raise ValueError(f"Resolution changed at {path}: {rgb.size} != {source_size}")
            resized = rgb.resize(output_size, Image.Resampling.LANCZOS)
            frames.append(resized.quantize(colors=256, method=Image.Quantize.MEDIANCUT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000.0 / args.fps))
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_bytes = args.output.stat().st_size
    if size_bytes > args.max_bytes:
        args.output.unlink()
        raise RuntimeError(
            f"GIF would be {size_bytes} bytes (limit {args.max_bytes}); "
            "increase stride or reduce width"
        )
    metadata = {
        "label": args.label,
        "content": "GeoSCOPE prediction only; no Endoscope1 ground truth",
        "source_frame_count": len(paths),
        "selected_indices": list(range(args.start, args.stop, args.stride)),
        "source_size": source_size,
        "output_size": output_size,
        "fps": args.fps,
        "duration_ms_per_frame": duration_ms,
        "gif_bytes": size_bytes,
        "color_or_geometry_postprocessing": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} ({size_bytes / (1024 * 1024):.2f} MiB)")


if __name__ == "__main__":
    main()
