# Method 3: Surface Fusion

`method3_surface_fusion` is an independent experimental method for the MICCAI
2026 iMED Novel View Synthesis challenge. Its goal is to preserve the PSNR of
the confirmed MV1A submission while testing whether a small surface-aware
extension improves coverage and SSIM.

## Current status

The narrow surface experiment has completed four-sequence validation:

- `M3-A`: an exact wrapper around the confirmed read-only MV1A modules;
- `M3-B`: a bounded projected-surface footprint with exact MV1A fallback for
  pixels whose local normal is unreliable;
- `M3-C`: M3-B plus source-only geometry confidence, with exact MV1A fallback
  below a fixed confidence threshold;
- source-only normal, footprint, coverage, confidence, timing, and memory
  diagnostics.

Stereo, temporal fusion, Endo-4DGS fallback, and target-aware processing are
not implemented.

On four fixed public sequences, M3-B improved corrected-geometry PSNR by
`+0.07898 dB` and SSIM by `+0.01156` relative to exact M3-A. See
`METHOD3_NOTES.md` for the baseline-adapted result and protocol caveat.

The exact MV1A reference is confirmed by the shared Docker image ID:

```text
method1-rgbd-reprojection:phase11-candidate
docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest
image ID: aaa952c110d1
```

The inspected MV1A configuration is:

```text
renderer: soft_depth
visibility tolerance: 1.0 mm + 0.01 * nearest target depth
depth softness: 8.0
source tool masking: disabled
hole handling: nearest-valid propagation
fill radius: 3 pixels on the 640x512 internal grid
depth filtering: none
temporal fusion: none
stereo: none
surface normals: none
```

The initial RGB-D image is distinct:

```text
method1-rgbd-reprojection:dev
image ID: dfea0cf2696f
```

## Development decision

Endoscope2/R will not be used. Although right RGB and `K2_R` are available,
the inspected input has no right depth and no explicit left-to-right rigid
transform. Consequently, `T_target_2R` cannot be computed without inventing or
estimating missing calibration.

The first planned experiment is therefore:

```text
M3-A: exact MV1A reference
M3-B: MV1A with only its four-pixel bilinear footprint replaced by a bounded
      projected elliptical surfel footprint
```

M3-B must retain MV1A's camera geometry, target-depth visibility test,
radius-3 nearest fill, export behavior, and output naming. It must not include
stereo, temporal fusion, new confidence weighting, or Endo-4DGS fallback.

See `METHOD3_NOTES.md` for the complete inspection record and proposed minimal
extension.

## Environment

Use the working Method-1/MV1A Python environment with PyTorch 2.1.2 and CUDA
11.8. Method 3 imports the confirmed MV1A geometry and inference loaders from
their read-only source directory. It does not modify those files.

If the environment is already active, first perform the syntax/import-free
compilation gate from the Method-3 root:

```bash
python -m py_compile \
  confidence.py \
  surface_geometry.py \
  surface_splat.py \
  render_sequence.py
```

This checks Python syntax only; it does not render data.

## Required one-frame gates

Do not begin with a full sequence. First render the established frame with
exact MV1A mode:

```bash
python render_sequence.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --output_dir outputs/one_frame/M3_A_MV1A \
  --renderer mv1a \
  --frame_id 2 \
  --device cuda
```

Then render the same source frame with the single surface-footprint change:

```bash
python render_sequence.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --output_dir outputs/one_frame/M3_B_surface \
  --renderer surface \
  --frame_id 2 \
  --footprint_scale 1.0 \
  --radius_min 0.5 \
  --radius_max 2.0 \
  --surface_support 5 \
  --depth_discontinuity_mm 2.0 \
  --depth_discontinuity_relative 0.02 \
  --device cuda
```

Each command writes the challenge-sized RGB prediction under `renders/` and
the internal-grid diagnostics to separate subdirectories. Inspect
`render_summary.json` before advancing.

## One-frame official-compatible evaluation

Target-view RGB and tool masks are accessed only by this offline evaluation
step, after both predictions are frozen. The renderer itself never imports or
opens them.

Evaluate M3-A with both fixed protocols:

```bash
python /mnt/cluster/workspaces/venkateda/Endo-4DGS/method1_rgbd_reprojection/evaluate.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --prediction_dir outputs/one_frame/M3_A_MV1A \
  --evaluation_dir outputs/one_frame/M3_A_MV1A/evaluation \
  --mask_protocol both \
  --allow_subset \
  --device cuda
```

Evaluate M3-B identically:

```bash
python /mnt/cluster/workspaces/venkateda/Endo-4DGS/method1_rgbd_reprojection/evaluate.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --prediction_dir outputs/one_frame/M3_B_surface \
  --evaluation_dir outputs/one_frame/M3_B_surface/evaluation \
  --mask_protocol both \
  --allow_subset \
  --device cuda
```

Do not use these one-frame target metrics for a broad parameter search. They
are a correctness and regression gate before a fixed multi-frame comparison.

## Full-sequence commands

Only after both one-frame gates succeed, run the exact reference:

```bash
bash scripts/run_mv1a_reference.sh \
  /mnt/cluster/datasets/iMED_NVS \
  session_004_scene_2_tool_1
```

Then run M3-B:

```bash
bash scripts/run_surface.sh \
  /mnt/cluster/datasets/iMED_NVS \
  session_004_scene_2_tool_1
```

Run the fixed M3-C confidence experiment separately:

```bash
bash scripts/run_confidence.sh \
  /mnt/cluster/datasets/iMED_NVS \
  session_004_scene_2_tool_1
```

The scripts intentionally omit `--overwrite`. Existing predictions are never
silently replaced.

## Outputs

```text
<output>/
├── renders/              # 1280x1024 challenge-style RGB PNG
├── rgb/                  # filled 640x512 RGB
├── depth/                # filled target-camera Z, NPY
├── valid_mask/           # raw geometric support
├── filled_valid_mask/
├── fill_mask/
├── confidence/           # NPY and PNG
├── normals/              # M3-B source-grid normal visualization
├── surface_radius/       # M3-B NPY and PNG
├── surface_validity/     # M3-B source-grid validity
├── surface_used/         # source surfels accepted after M3-C confidence gate
├── surface_recovered/    # target pixels outside MV1A bilinear support
├── geometry_confidence/  # source-only combined confidence, NPY and PNG
├── depth_confidence/     # source-only confidence component
├── tangent_confidence/   # source-only confidence component
├── viewing_confidence/   # source-only confidence component
└── render_summary.json
```

The full-stream filenames remain zero-based, sequential, and five-digit. A
one-frame/subset run preserves the frame's full-stream index so it cannot be
mistaken for a complete challenge output.

## Scientific constraints

- Do not modify the original Endo-4DGS or MV1A implementation.
- Never use Endoscope1 RGB for inference, tuning, camera adjustment,
  appearance correction, confidence estimation, or component selection.
- Endoscope1 RGB is evaluation-only.
- Do not invent a right-camera depth stream or stereo extrinsic.
- Keep local/public evaluation separate from hidden-leaderboard results.
- PSNR is the primary selection metric and SSIM is secondary.
