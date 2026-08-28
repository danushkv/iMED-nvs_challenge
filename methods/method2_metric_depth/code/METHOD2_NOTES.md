# Method 2 inspection notes

Inspection date: 2026-08-24

Method 2 is a separate copy of the challenge-adapted Endo-4DGS baseline. The
original Endo-4DGS directory remains outside this workspace and must not be
modified.

## Current baseline

### RGB loss

The baseline trains only on Endoscope 2 cameras. At each iteration it renders
one source frame (`batch_size=1`) and loads the corresponding Endoscope 2 RGB,
depth, and source tool mask.

The primary RGB term is:

```text
L_rgb = mean(mask * abs(I_render - I_gt))
```

The inputs are first multiplied by the tissue mask, and `l1_loss` multiplies
the absolute error by the mask again. Because the final reduction is a mean
over the whole image rather than a division by the number of valid pixels,
the loss magnitude depends on tissue coverage.

Files:

- `train.py`, `scene_reconstruction`, lines around the tensor assembly and
  `Ll1` computation.
- `utils/loss_utils.py`, `l1_loss`.

### SSIM loss

`OptimizationParams.lambda_dssim` defaults to `0` and the iMED runtime config
does not override it. Therefore DSSIM/SSIM is not part of B0 training.

The dormant implementation is:

```text
L_dssim = lambda_dssim * (1 - SSIM(I_render_masked, I_gt_masked))
```

This is ordinary full-image SSIM applied after tool pixels have been zeroed;
it is not a strict masked SSIM reduction. Tool interiors therefore contribute
easy zero-versus-zero regions, and windows crossing tool boundaries mix valid
and invalid content.

### Depth loading and units

- Endoscope 2 depth is loaded from `endoscope2/depthL/*.npy` as `float32`.
- Resolution is 512x640; source RGB and masks are reduced from 1024x1280.
- Depth is optical-axis Z-depth in millimetres.
- The inspected public source collection contains finite positive depth; the
  training code nevertheless accepts only locations where both GT and rendered
  depth are positive.
- Depth is not converted to metres or normalized in the loader.
- The pretraining point cloud backprojects directly with depth in millimetres,
  and the static camera translations use the same coordinate system.
- Camera `trans=0` and `scale=1` leave geometry unscaled. `camera_extent=10`
  is passed as the Gaussian spatial learning-rate scale; it does not rescale GT
  depth or camera coordinates.
- Rasterized depth is consequently expressed in the same millimetre-valued
  scene/camera coordinate system as GT depth.

### Current depth loss

After masking locations where either depth is non-positive and applying the
source tissue mask, B0 independently normalizes predicted and GT depth by their
batch maxima:

```text
D_pred_norm = D_pred / (max(D_pred) + 1e-6)
D_gt_norm   = D_gt   / (max(D_gt)   + 1e-6)
L_depth     = depth_weight * mean(mask * abs(D_pred_norm - D_gt_norm))
```

`depth_weight = 0.01`. As with RGB L1, the reduction is over the whole image,
not strictly over the valid pixels. Independent normalization removes absolute
metric scale supervision.

Additional enabled geometry-related B0 terms are:

- gradient correlation/smoothness, weighted by `depth_weight=0.01`;
- pseudo-normal MAE, weighted by `normal_weight=0.001`;
- image and depth confidence losses, each with default weight `0.1`;
- fine-stage deformation/grid and image/depth TV regularization.

### Tool masks

Raw `toolL` uses 255 for tool and 0 for tissue. The loader converts this to:

```text
tissue_mask = 1 - raw_tool_mask / 255
```

Contrary to the initial Method 2 hypothesis, B0 is already source-tool-aware:

- the first-frame point-cloud initialization excludes source tool pixels;
- RGB, depth, normal, confidence, and dormant SSIM inputs are tissue-masked;
- training uses only Endoscope 2 masks.

Tool masks do not explicitly gate image-space Gaussian visibility statistics
or densification. Densification uses `radii > 0` and accumulated render-space
gradients. No mask dilation is implemented.

Therefore a B2 experiment must not duplicate the existing T1 behavior. A
scientifically distinct B2 could test strict valid-pixel loss normalization,
mask dilation, and only later (if justified) densification gating.

### Evaluation masks

The adapted `metrics.py` uses target Endoscope 1 tissue masks only for offline
evaluation. It combines them with a global cross-camera overlap mask and uses
strict masked PSNR and masked SSIM. These evaluation masks do not backpropagate
into training.

On public data, periodic `training_report` evaluations can read Endoscope 1 GT
to report metrics, but they do not contribute to the loss, optimizer, camera
selection, or checkpoint selection. Hidden challenge input can omit target RGB;
the challenge-adapted loader substitutes zero images only to preserve the target
camera list. Target RGB must never be used for Method 2 optimization or tuning.

### Appearance modelling

No exposure parameter, white-balance transform, affine RGB correction,
per-frame appearance code, illumination embedding, or appearance MLP was found.
The deformation network is time-conditioned for geometry/features, but there is
no separate lightweight appearance correction module.

### Training configuration

The challenge entrypoint defaults to:

```text
coarse_iterations = 300
fine_iterations   = 1000
batch_size        = 1
use_pretrain      = true
use_depth         = true
use_smooth        = true
use_normal        = true
use_confidence    = true
```

The runtime uses the official CUDA 11.8 baseline image with `simple-knn` and
`diff-gaussian-rasterization-depth` rebuilt for RTX A5000 compute capability
8.6. This changes CUDA compatibility only, not the numerical method.

## B0 reference

Sequence: `session_004_scene_2_tool_1`

```text
PSNR:            20.5385971 dB
SSIM:             0.6538426
LPIPS:            0.1131114
training process: 159 seconds from logged process timestamps
render kernel:     43.910 FPS, approximately 4.532 seconds for 199 frames
total wall time:  337.927 seconds (training + rendering + metrics + export)
GPU:              NVIDIA RTX A5000 24 GB
peak GPU memory:  not measured in this run
```

This is the fixed one-sequence development reference. The previously reported
challenge-wide baseline values (mean PSNR 21.011, mean SSIM 0.678) are retained
as external aggregate context, not substituted for this reproducible sequence
result.

## Proposed B1 implementation

### Files to modify

- `arguments/__init__.py`: add independently switchable depth-loss options
  and a primary-depth weight separate from B0's auxiliary depth weight.
- `train.py`: factor the depth calculation into explicit modes and add
  diagnostics without changing the B0 path.
- `imed_nvs_baseline.py`: expose/pass Method 2 experiment options through the
  challenge-compatible entrypoint or generated runtime config.
- A small dedicated loss utility or functions in `utils/loss_utils.py` only if
  this keeps the change clearer than an inline implementation.

### Required modes

```text
normalized     exact B0 equation
metric_l1      strict valid-pixel mean(abs(D_pred - D_gt))
metric_huber   strict valid-pixel Smooth-L1 in millimetres
```

The B0 path must remain bit-for-bit equivalent in formulation.

### Proposed metric equations

Let:

```text
M = tissue_mask AND finite(D_gt) AND finite(D_pred)
    AND D_gt > 0 AND D_pred > 0
```

Metric L1:

```text
L_depth_metric_l1 = depth_weight * sum(M * abs(D_pred - D_gt)) / max(sum(M), 1)
```

Metric Huber:

```text
e = D_pred - D_gt
L_depth_metric_huber = depth_weight * mean_valid(smooth_l1(e; beta_mm))
```

Both quantities are already in millimetres, so no geometric scale conversion
is required. The main numerical issue is loss magnitude: a raw millimetre loss
is much larger than B0's normalized loss. B1 must log unweighted and weighted
terms before selecting a small three-point depth-weight sweep. It must not reuse
`0.01` blindly. The new primary-depth weight is independent from the existing
`depth_weight=0.01` used by gradient and fine-stage TV terms, preventing the
weight sweep from changing multiple objectives simultaneously.

### Initial diagnostics

Log at a configurable interval:

```text
L_rgb
L_depth_unweighted
L_depth_weighted
L_smooth
L_normal
L_confidence_image
L_confidence_depth
L_total
mean_abs_depth_error_mm
median_abs_depth_error_mm
mean_relative_depth_error
valid_depth_fraction
```

Save GT depth, rendered depth, absolute depth error, source RGB GT, and source
RGB render. These diagnostics use Endoscope 2 only.

### Open issues before implementation

1. Measure B0 depth-loss and metric-error magnitudes during a short diagnostic
   run before selecting Huber beta and weights.
2. Preserve all other B0 terms for the B1 ablation.
3. Decide whether the gradient/confidence depth terms remain independently
   normalized in B1; initially they should remain unchanged so only the primary
   depth term varies.
4. The baseline already masks source tools, so B2 must be reframed as a mask
   formulation ablation rather than simply enabling tool masks.

No Method 2 training-code changes have been made at this checkpoint.
