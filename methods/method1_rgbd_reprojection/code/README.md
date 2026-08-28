# Method 1: RGB-D reprojection

This is a standalone, training-free experimental method for the MICCAI 2026
iMED Novel View Synthesis challenge. It backprojects Endoscope 2 left RGB-D,
applies the supplied static camera transform, and forward-splats into Endoscope
1 left-camera pixels.

The surrounding Endo-4DGS repository is a read-only reference. This directory
does not import or modify it. No Method-1 inference path opens Endoscope 1 RGB,
depth, or masks.

## Current correctness-gate status

Phases 0-7 are implemented:

- read-only dataset inspection and explicit calibration parsing;
- NumPy/PyTorch camera geometry;
- source-to-source identity test with PSNR/SSIM;
- nearest-pixel z-buffer;
- bilinear and depth-aware soft splatting;
- opt-in source tool removal;
- one-frame source-to-target output and leakage-safe debug montage.
- H0 raw output, H1 conservative local morphology, H2 radius-limited nearest
  propagation, and optional H3 small-component OpenCV inpainting.
- full sorted-stream rendering with one-frame/subset modes, per-frame
  provenance, sampled debug montages, and sequence performance summaries.

Target-GT evaluation, baseline comparison, and parameter sweeps are
intentionally not yet enabled. Continue only after reviewing one complete
sequence's Phase-7 summary and sampled debug outputs.

## Environment

From this directory, in an environment with a compatible PyTorch installation:

```bash
python -m pip install -r requirements.txt
```

Installing dependencies is not required if the existing environment already
provides NumPy, Pillow, and PyTorch >= 2.0.

## 1. Inspect the dataset

One sequence, representative first/middle/last depth scan:

```bash
python inspect_dataset.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1
```

All sequences and every Endoscope 2 depth pixel (slow on network storage):

```bash
python inspect_dataset.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --scan_all_depths
```

## 2. Run the identity sanity gate

The default uses no source tool mask so every dense source pixel should return
to the same pixel:

```bash
python identity_test.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --frame_id 2 \
  --output_json outputs/identity/session_004_scene_2_tool_1_frame_000002.json
```

Or use the wrapper from this directory:

```bash
bash scripts/test_identity.sh \
  /mnt/cluster/datasets/iMED_NVS \
  session_004_scene_2_tool_1 \
  2
```

The script exits nonzero and reports `FAIL` if mean reprojection error exceeds
`1e-4` pixel, coverage falls below `99.99%`, PSNR is below 70 dB, or SSIM is
below 0.9999. Do not proceed to cross-camera rendering on failure.

## 3. Render one source-to-target frame

Nearest z-buffer geometry check:

```bash
python render_sequence.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --frame_id 2 \
  --output_dir outputs/one_frame/nearest_no_tool_mask \
  --renderer nearest
```

Bilinear splatting:

```bash
python render_sequence.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --frame_id 2 \
  --output_dir outputs/one_frame/bilinear_no_tool_mask \
  --renderer bilinear
```

Depth-aware soft splatting with source tool removal:

```bash
python render_sequence.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --frame_id 2 \
  --output_dir outputs/one_frame/soft_depth_tool_mask \
  --renderer soft_depth \
  --mask_source_tools
```

The corresponding no-mask run should be saved separately; never compare modes
after overwriting outputs. `session_006_scene_7_tool_3` has no tool masks and
must be rendered without `--mask_source_tools`.

For the first frame above, the debug visualization is saved at:

```text
outputs/one_frame/<variant>/debug/00000.png
```

No GT panel or target RGB is included at this stage. Output structure is:

```text
outputs/one_frame/<variant>/
├── rgb/00000.png
├── depth/00000.npy
├── depth/00000.png
├── valid_mask/00000.png
├── confidence/00000.npy
├── confidence/00000.png
├── correspondence/00000.npy
├── correspondence/00000_source_to_target.npy
└── debug/
    ├── 00000.png
    ├── 00000.json
    └── 00000_hit_count.npy
```

`rgb/00000.png` follows the adapted baseline's sequential zero-based filename
convention. The JSON records its original dataset frame (`frame_000002.png`).

## 4. Phase 6: isolated hole-handling runs

Use exactly the same renderer and source-mask setting for every run so the hole
strategy is the only changing variable. The commands below use the validated
depth-aware renderer with source tool masking.

### H0 — no filling

```bash
python render_sequence.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --frame_id 2 \
  --output_dir outputs/phase6/h0_none \
  --renderer soft_depth \
  --mask_source_tools \
  --hole_fill none
```

### H1 — conservative surrounded-pixel morphology

```bash
python render_sequence.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --frame_id 2 \
  --output_dir outputs/phase6/h1_morphological_r1 \
  --renderer soft_depth \
  --mask_source_tools \
  --hole_fill morphological \
  --fill_radius 1 \
  --morph_min_neighbors 7 \
  --morph_min_confidence 0.1
```

The default requires at least seven of eight neighbours to be valid, preventing
the edge of a large disocclusion from growing inward. Radius means at most 1-3
such iterations, not unrestricted morphological closing.

### H2 — nearest valid propagation, radii 1-3

Run each radius into a different directory:

```bash
for radius in 1 2 3; do
  python render_sequence.py \
    --data_root /mnt/cluster/datasets/iMED_NVS \
    --sequence session_004_scene_2_tool_1 \
    --frame_id 2 \
    --output_dir "outputs/phase6/h2_nearest_r${radius}" \
    --renderer soft_depth \
    --mask_source_tools \
    --hole_fill nearest \
    --fill_radius "$radius"
done
```

H2 copies only from raw geometrically valid pixels. It cannot propagate farther
than the specified Euclidean pixel radius and does not chain synthesized pixels.

### H3 — optional small-component inpainting

H3 requires OpenCV only for this mode:

```bash
python -m pip install opencv-python-headless

python render_sequence.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --frame_id 2 \
  --output_dir outputs/phase6/h3_inpaint_r2 \
  --renderer soft_depth \
  --mask_source_tools \
  --hole_fill inpaint \
  --fill_radius 2 \
  --inpaint_max_area 25
```

Only invalid connected components with at most 25 pixels are eligible. Large
exterior regions, disocclusions, and the source-tool-shaped region are excluded.
RGB uses OpenCV Telea inpainting; depth and confidence use bounded nearest-valid
propagation.

### Phase-6 output provenance

For every strategy:

```text
<output_dir>/
├── rgb/                    # selected H0/H1/H2/H3 prediction
├── depth/                  # selected prediction
├── confidence/             # selected prediction
├── valid_mask/             # raw geometric support; never expanded by filling
├── fill_mask/              # pixels synthesized by H1/H2/H3
├── filled_valid_mask/      # valid_mask OR fill_mask
├── raw/
│   ├── rgb/                # preserved H0 RGB
│   ├── depth/              # preserved H0 depth
│   ├── confidence/         # preserved H0 confidence
│   └── valid_mask/         # preserved H0 support
├── correspondence/         # raw geometry only; no invented fill correspondence
└── debug/
    ├── 00000.png
    ├── 00000_hole_fill.png # direct H0-versus-filled comparison
    └── 00000.json          # raw, filled, timing, and memory statistics
```

H0 must always be reported separately. Filled coverage must never be presented
as geometric coverage; both values are recorded independently in JSON.

## 5. Phase 7: render one complete sequence

The `--frame_id` argument is now optional. Omitting it processes every sorted,
synchronized Endoscope 2 timestamp and produces `00000.png`, `00001.png`, ...,
matching the adapted baseline's zero-based five-digit convention.

Render raw H0 with the validated depth-aware renderer and source tool masking:

```bash
python render_sequence.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --output_dir outputs/sequences/soft_depth_masked_h0/session_004_scene_2_tool_1 \
  --renderer soft_depth \
  --mask_source_tools \
  --hole_fill none
```

Equivalent wrapper command:

```bash
bash scripts/run_one_sequence.sh \
  /mnt/cluster/datasets/iMED_NVS \
  session_004_scene_2_tool_1 \
  outputs/sequences/soft_depth_masked_h0 \
  --renderer soft_depth \
  --mask_source_tools \
  --hole_fill none
```

For a short non-evaluation smoke subset, use sorted-stream indices:

```bash
python render_sequence.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --output_dir outputs/subset/session_004_scene_2_tool_1 \
  --start_index 0 \
  --max_frames 5 \
  --renderer soft_depth \
  --mask_source_tools \
  --hole_fill none
```

This preserves the global stems `00000` through `00004`. `--end_index` is
exclusive. `--frame_id` still selects one original dataset ID.

By default, full correspondence arrays are saved for every frame and montages
are saved for the first, middle, and last selected frames. Useful controls:

```text
--debug_every N       save a montage every N selected frames plus the last
--no_debug            skip montage images (per-frame JSON is retained)
--no_correspondence   skip large correspondence arrays to reduce disk use
--overwrite           overwrite matching outputs without deleting anything
```

Every sequence writes `render_summary.json`, containing:

- total sequence runtime;
- mean/median wall and compute time per frame;
- peak GPU memory;
- total source-valid and projected pixels;
- mean/median raw and filled target coverage;
- calibration and transform matrices;
- per-frame provenance and performance records.

When hole filling is `none`, primary outputs are already H0, so a duplicate
`raw/` tree is not written. For H1-H3, the separate raw tree is retained.

### Full dataset

Render all 20 sequences sequentially without source masking:

```bash
bash scripts/run_all.sh \
  /mnt/cluster/datasets/iMED_NVS \
  outputs/all_sequences/soft_depth_unmasked_h0 \
  --renderer soft_depth \
  --hole_fill none
```

This unmasked command includes `session_006_scene_7_tool_3`, whose tool-mask
directories are empty. A globally masked run will intentionally stop on that
sequence rather than synthesize or silently ignore its missing mask. Run that
sequence separately without `--mask_source_tools` if mixed mask availability is
acceptable under the challenge rules.

### Phase-7 output structure

```text
<output_dir>/
├── rgb/
├── depth/
├── valid_mask/          # raw geometric support
├── filled_valid_mask/
├── fill_mask/
├── confidence/
├── correspondence/
├── debug/               # per-frame JSON + sampled montages/hit counts
└── render_summary.json
```

## 6. Phase 8: evaluate one sequence

`evaluate.py` is evaluation-only: it is the only Method-1 program that opens
Endoscope 1 RGB or target `toolL`. It cannot alter calibration, predictions, or
renderer parameters. Run it on the completed H0 sequence:

```bash
python evaluate.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --prediction_dir outputs/sequences/soft_depth_masked_h0/session_004_scene_2_tool_1 \
  --mask_protocol both
```

The default reports two explicitly named protocols:

- `corrected_geometry`: scales K2 to the depth grid and K1 to the output grid;
- `adapted_baseline`: faithfully reproduces the inspected adapted `metrics.py`
  unscaled-K, provisional-size, resize, and morphology behavior.

This is not a GT-based protocol selection. The adapted baseline behavior has a
known resolution inconsistency, so neither result is labelled organizer-
official until the organizers clarify it. PSNR and SSIM themselves exactly
match the challenge-adapted formulas.

In both protocols the per-frame evaluation mask is:

```text
target tissue-valid mask AND global geometric overlap mask
```

Prediction validity is deliberately not added to that mask: black invalid
prediction pixels inside the legitimate evaluation region are penalized.
Coverage separately reports the fraction of the evaluation mask reached by the
raw and selected (possibly filled) prediction.

Outputs are written beneath `<prediction_dir>/evaluation/`:

```text
evaluation/
├── results.json
├── per_frame_corrected_geometry.csv
├── per_frame_adapted_baseline.csv
├── corrected_geometry/
│   ├── global_overlap_mask.png
│   └── masks/
└── adapted_baseline/
    ├── global_overlap_mask.png
    └── masks/
```

Each CSV has exactly `frame,psnr,ssim,coverage`. `results.json` contains mean
and median PSNR, SSIM, filled coverage, raw geometric coverage, mask sizes, and
per-frame provenance.

The adapted baseline has a first-run-only, method-dependent behavior caused by
loading `overlap_mask.png` before overwriting it with its geometric mask. It is
disabled by default. Reproduce it only for an explicit diagnostic:

```bash
python evaluate.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --prediction_dir outputs/sequences/soft_depth_masked_h0/session_004_scene_2_tool_1 \
  --mask_protocol adapted_baseline \
  --include_first_frame_support \
  --evaluation_dir outputs/evaluation_diagnostics/first_frame_support
```

For all already-rendered sequences:

```bash
bash scripts/evaluate_all.sh \
  /mnt/cluster/datasets/iMED_NVS \
  outputs/all_sequences/soft_depth_unmasked_h0 \
  --mask_protocol both
```

Full-sequence evaluation is enforced by default. `--allow_subset` exists only
for debugging an intentionally rendered subset. Use `--overwrite` to replace
an existing evaluation result; no prediction is deleted or modified.

## 7. Phase 9: compare existing Endo-4DGS results

`compare_baseline.py` reads already-computed `results.json` files. It does not
train, render, or recompute metrics for Endo-4DGS or Method 1. No baseline
result file is currently present in this workspace, so supply the path to your
existing Endo-4DGS result or model directory.

For the sequence evaluated above, compare using the adapted-baseline protocol:

```bash
python compare_baseline.py \
  --baseline_root /path/to/endo4dgs/session_004_scene_2_tool_1/results.json \
  --rgbd_root outputs/sequences/soft_depth_masked_h0/session_004_scene_2_tool_1 \
  --sequence session_004_scene_2_tool_1 \
  --protocol adapted_baseline \
  --baseline_protocol adapted_baseline \
  --baseline_method ours_14000 \
  --output_dir outputs/comparisons/adapted_baseline/session_004_scene_2_tool_1
```

Replace `ours_14000` with the exact key in the existing baseline
`results.json`. If that file contains exactly one PSNR/SSIM record,
`--baseline_method` may be omitted. The sequence is discovered from the
baseline `cfg_args` `source_path`, a `session_*` path component, or an explicit
`sequence` field.

The protocol declarations must match. In particular, do not compare Method
1's `corrected_geometry` score against an Endo-4DGS score generated with the
adapted mask. A corrected comparison is allowed only when the saved baseline
metric was also computed with the corrected mask:

```text
--protocol corrected_geometry --baseline_protocol corrected_geometry
```

For roots containing one result per sequence:

```bash
python compare_baseline.py \
  --baseline_root /path/to/existing/endo4dgs_outputs \
  --rgbd_root outputs/all_sequences/soft_depth_unmasked_h0 \
  --protocol adapted_baseline \
  --baseline_protocol adapted_baseline \
  --baseline_method ours_14000 \
  --output_dir outputs/comparisons/adapted_baseline/all_sequences
```

By default, the two roots must contain identical sequence sets. Use repeated
`--sequence` options to select an explicit subset. `--allow_partial` permits an
intersection-only diagnostic and records every missing sequence; it should not
be used for a claimed full-dataset comparison.

Outputs are:

```text
<output_dir>/
├── comparison.csv
├── comparison.md
└── comparison_results.json
```

The table contains Endo-4DGS and RGB-D PSNR/SSIM, signed deltas defined as
`RGBD - Endo4DGS`, and Method-1 coverage. The ranking rule follows the inspected
challenge metric: arithmetic mean of per-frame PSNR is primary; arithmetic
mean of per-frame SSIM is used only as a secondary tie-breaker. JSON records
the resulting winner and deciding metric, PSNR and SSIM win/loss/tie counts,
both-metric wins, mean and median improvements, best and worst sequences,
missing sequences, protocol declarations, and source-file provenance.

The adapted baseline's first-run overlap-mask bug is not recorded in its old
`results.json`. Declare `--baseline_protocol adapted_baseline` only when you
know that result used the stable geometric adapted mask; the comparator cannot
infer this after the fact.

If you must regenerate the existing baseline outputs in its separate
environment, render only the test split from the already-trained checkpoint:

```bash
python render.py \
  --model_path /path/to/trained/session_004_scene_2_tool_1 \
  --iteration -1 \
  --skip_train \
  --skip_video
```

Then run the adapted metric. Because `render.py` first writes a method-dependent
support mask and the inspected `metrics.py` overwrites it only after loading
it, run `metrics.py` twice and retain the second `results.json`:

```bash
python metrics.py \
  --model_paths /path/to/trained/session_004_scene_2_tool_1

python metrics.py \
  --model_paths /path/to/trained/session_004_scene_2_tool_1
```

This does not retrain Endo-4DGS. If the existing `test/ours_*/renders` and
geometric `overlap_mask.png` are already present, skip rendering and one metric
run is sufficient. Do not use a newly trained checkpoint merely to obtain the
comparison.

## 8. Phase 10: evaluation-only failure visualizations

Phase 10 is implemented as an option of `evaluate.py`, keeping all access to
Endoscope 1 RGB and target tool masks inside the evaluation-only program. It
does not modify predictions, calibration, masks, or renderer parameters.

On restricted systems, create both evaluation and visualization outputs under
`/tmp`:

```bash
METHOD1_EVAL_TMP=$(mktemp -d /tmp/imed_nvs_phase10_eval.XXXXXX)
METHOD1_VIS_TMP=$(mktemp -d /tmp/imed_nvs_phase10_visuals.XXXXXX)

python evaluate.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --prediction_dir outputs/sequences/soft_depth_masked_h0/session_004_scene_2_tool_1 \
  --evaluation_dir "$METHOD1_EVAL_TMP" \
  --mask_protocol adapted_baseline \
  --save_visualizations \
  --visualization_protocol adapted_baseline \
  --visualization_dir "$METHOD1_VIS_TMP"
```

Representative frames are chosen deterministically: first, middle, last, best
and worst PSNR, and best and worst SSIM. Duplicate selections are merged. Add
specific zero-based output indices with `--visualization_extra_indices`.

The visualization directory contains:

```text
<visualization_dir>/
├── montages/             # nine-panel source/geometry/GT/error diagnostics
├── absolute_error/       # evaluation-mask-only RGB error maps
├── red_cyan_overlay/     # red=GT, cyan=prediction, white=aligned
└── phase10_summary.json  # metrics and non-causal diagnostic correlations
```

Each montage contains source RGB/depth, projected RGB/depth, confidence,
projection/evaluation masks, GT RGB, absolute error, and a red/cyan alignment
overlay. The JSON reports illumination bias, evaluation-region coverage,
projection-boundary error, high-gradient/thin-structure error, specular error,
and source-tool fraction. These are diagnostic correlations, not automatic
causal conclusions.

## 9. Phase 11: small staged parameter sweep

`phase11_sweep.py` performs a restartable, one-sequence ablation in three
stages. It avoids the full 72-configuration Cartesian product:

1. six H0/unfiltered runs: three renderers times source-tool masking on/off;
2. with the stage-1 renderer/mask fixed: H0, H1 morphological radius 1, and H2
   nearest propagation with radii 1, 2, and 3;
3. with stage-2 geometry/fill fixed: unfiltered, median 3x3, and bilateral
   depth filtering.

Already-completed configurations are reused. The unique total is 12 sequence
renders. Selection uses mean PSNR as primary and mean SSIM as secondary, exactly
matching the challenge ranking. Target RGB is opened only by `evaluate.py`
after each fixed prediction set exists; there is no per-frame tuning.

Run the preliminary sweep on the established sequence:

```bash
cd /mnt/cluster/workspaces/venkateda/Endo-4DGS/method1_rgbd_reprojection

bash scripts/run_phase11_sweep.sh \
  /mnt/cluster/datasets/iMED_NVS \
  session_004_scene_2_tool_1 \
  outputs/phase11/session_004_scene_2_tool_1 \
  --protocol adapted_baseline \
  --device cuda
```

If a run was interrupted after writing partial files, inspect it first, then
resume with `--overwrite_partial`. This replaces only matching outputs and
does not delete directories. Completed results are always reused.

Outputs are:

```text
outputs/phase11/session_004_scene_2_tool_1/
├── ablation_results.csv   # all unique configurations, sorted by PSNR/SSIM
├── ablation_results.json  # staged selections and full provenance
└── runs/
    └── <configuration>/
        ├── rgb/
        ├── valid_mask/
        ├── render_summary.json
        └── evaluation/
```

The bilateral treatment is fixed before evaluation: 5x5 support, spatial sigma
2 pixels, and range sigma 3 mm. The CSV records coverage, runtime, and peak GPU
memory alongside PSNR and SSIM. Because this sweep uses one public sequence,
its selected configuration is explicitly labelled preliminary and should be
confirmed on additional public sequences before final submission.

## Commands intentionally gated after Phase 11

Updating the Docker default to the selected configuration remains gated until
the ablation table has been reviewed.

## Scientific safeguards

- The transform is `T_cam1_cam2 = inverse(T_world_cam1) @ T_world_cam2`.
- `T_a_b` always means `X_a = T_a_b @ X_b`.
- Endoscope 1 RGB is not used for transform selection, tuning, rendering, or
  filling.
- Invalid output pixels remain black with an explicit validity mask.
- H0 never fills holes. H1-H3 save every synthesized pixel in `fill_mask`; large
  disocclusions remain invalid by design.
- Confirm with challenge organizers that `K1_L`, the Endoscope 1 static pose,
  and optional Endoscope 2 tool masks are legal held-out inference inputs.
- See `DATASET_NOTES.md` for an inconsistency in the adapted baseline overlap-
  mask implementation that must be resolved before calling evaluation official.
