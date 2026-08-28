# Method 1: RGB-D reprojection

Method 1 is a standalone, training-free NVS baseline. It backprojects the
Endoscope2/L RGB-D observation, applies the supplied rigid source-to-target
camera transform, and forward-splats samples into Endoscope1/L.

## What this folder represents

This folder preserves the full research path—nearest projection, bilinear
splatting, depth-aware soft splatting, source-mask experiments, depth filters,
and bounded hole-handling variants. The submitted Phase-11 refinement is
documented separately as [MV1A](../method1_5_mv1a/README.md).

Historical hidden result for the initial RGB-D method:

```text
PSNR: 19.247
SSIM: 0.581
```

## Environment with uv

The research code needs NumPy, Pillow, SciPy, and PyTorch. Create an isolated
environment rather than modifying an Endo-4DGS environment:

```bash
cd methods/method1_rgbd_reprojection/code
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

For exact MV1A parity, prefer its Docker image with PyTorch 2.1.2 and CUDA
11.8. The broad research requirement `torch>=2.0` is convenient but not a
bit-exact lock.

## Geometry gate

Run the source-to-source identity test before target projection:

```bash
python identity_test.py \
  --data_root /path/to/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --frame_id 2 \
  --output_json outputs/identity/frame_000002.json
```

Expected gate: reprojection error below `1e-4` pixel, coverage above 99.99%,
PSNR above 70 dB, and SSIM above 0.9999.

## One-frame render

```bash
python render_sequence.py \
  --data_root /path/to/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --frame_id 2 \
  --output_dir outputs/one_frame/soft_depth \
  --renderer soft_depth
```

The original detailed README, dataset inspection notes, experiment scripts,
and evaluation tool are copied into `code/` by the release assembly script.

## Scientific boundary

Inference reads Endoscope2/L RGB-D and supplied calibration only. Endoscope1
RGB and masks belong exclusively to offline evaluation after rendering.
