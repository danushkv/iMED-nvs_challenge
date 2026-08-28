# Pinned upstream G-SHARP inspection

Inspection target: gsplat commit
`846c07932a77a901b474c40dd7fbfe42965ab354` (`gsplat==1.6.0`). The source
checkout is `method4_gsharp/third_party/gsplat` and has not been patched.

Primary integration surface:
`examples/dynamic_surgical_trainer.py`. Relevant dependencies are
`examples/datasets/endonerf.py`, `gsplat/init_utils.py`,
`gsplat/contrib/dynamic/*`, `gsplat/training/schedulers.py`,
`gsplat/losses.py`, `gsplat/regularizers.py`, `gsplat/strategy/default.py`,
and `gsplat/rendering.py`.

## Multi-frame initialization

`build_splats_from_parser()` materializes all training images, depths, masks,
camera-to-world poses, and intrinsics. It optionally clamps depth to 1.5 times
the maximum EndoNeRF `poses_bounds.npy` bound, then calls upstream
`multi_frame_depth_unprojection()`.

For every frame, `multi_frame_depth_unprojection()` keeps pixels satisfying
`mask != 0` and `depth > 0`, and applies the pinhole Z-depth equations:

```text
x = (u - cx) * depth / fx
y = (v - cy) * depth / fy
z = depth
X_world = camtoworld @ [x, y, z, 1]
```

All observations are concatenated and a `torch.randperm` subset is selected
when the count exceeds `init_max_points` (default 50,000). RGB comes directly
from the corresponding source pixels as float `[0,1]`.

`knn_scale_init(xyz, k=3)` computes a chunked pairwise KNN distance. For each
point it takes the RMS distance to its three nearest non-self neighbours,
clamps it to `1e-7`, and returns the natural logarithm. The trainer repeats
that scalar on XYZ, yielding isotropic log-scales, and adds
`log(init_scale_multiplier)` (default multiplier 1).

Quaternions are random normal four-vectors normalized to unit norm. Opacity is
stored as a logit initialized so `sigmoid(opacity) == 0.1`. Colors are direct
RGB parameters rather than spherical harmonics.

## Input conventions

- **Image:** `H x W x 3`, RGB float32 in `[0,1]`; upstream also accepts
  uint8 in `multi_frame_depth_unprojection()` and normalizes it.
- **Depth:** `H x W` float32 optical-axis Z-depth. Zero is the missing-depth
  sentinel. Upstream does not apply a universal unit conversion: depth,
  translations, means, scales, and AABB must use one consistent unit system.
- **Mask:** returned/training convention is tissue/include `1`, tool/exclude
  `0`. EndoNeRF on-disk masks are tool-white (`255`), tissue-black (`0`), so
  the loader computes `1 - raw/255`.
- **Pose:** homogeneous `4 x 4` camera-to-world. Unprojection multiplies by
  this pose; rendering explicitly computes `viewmat = inverse(camtoworld)`.
- **Intrinsics:** ordinary pinhole `3 x 3` matrix in the same pixel grid as
  image/depth. Upstream EndoNeRF uses one focal value with principal point at
  integer image centre. The iMED adapter must preserve its supplied `fx`,
  `fy`, `cx`, and `cy` and scale them when RGB is resized to depth resolution.
- **Time:** scalar float32 supplied to HexPlane. EndoNeRF uses `index/N`.
  The validated iMED pipeline uses ordered synchronized index divided by
  `N-1`; Method 4 uses that iMED convention, including endpoints 0 and 1.

## iMED mapping established from frozen geometry

- `endoscope2/L`: legal source RGB.
- `endoscope2/depthL`: legal source float32 NPY Z-depth in millimetres.
- `endoscope2/toolL`: source tool-white mask, inverted to tissue/include.
- `K2_L`: Endoscope2 source intrinsics.
- pose camera id 0: Endoscope2 camera-to-world.
- `K1_L`: legal target intrinsics, subject to challenge-input confirmation.
- pose camera id 1: Endoscope1 camera-to-world, subject to the same rule.
- Endoscope1 RGB is not discovered or loaded by the Method 4 adapter or
  renderer. It is accessible only to the separate post-render evaluator.

The full RGB grid is 1280x1024 while depth is 640x512. Source training uses
RGB and mask resized to 640x512 and `K2_L` scaled by 0.5 on both axes. Target
challenge rendering uses the full `K1_L` and 1280x1024 output grid.

Pose rows are `camera_id tx ty tz qx qy qz qw`; quaternion input order is
XYZW (SciPy `xyzw`). Both pose translations and source depth are treated as
millimetres. For the standard gsplat path, Gaussian quaternions are documented
as WXYZ, relevant to the deferred M4-B orientation initialization.

## HexPlane

`HexPlaneField` represents `(x,y,z,t)` using six bilinearly sampled 2D planes:
`xy`, `xz`, `xt`, `yz`, `yt`, and `zt`. Per-scale plane features are multiplied
elementwise and features from scales `(1,2)` are concatenated. Defaults are 32
features per plane and base resolution `[64,64,64,25]`; spatial resolutions
are multiplied per scale while temporal resolution remains 25.

Spatial coordinates are mapped through an AABB to `[-1,1]`; time is passed
unchanged. `grid_sample` uses bilinear interpolation, `align_corners=True`,
and `padding_mode="border"`. Temporal planes initialize to one and spatial
planes uniformly in `[0.1,0.5]`.

The trainer derives a symmetric spatial half-bound as
`max(cfg.hex_bounds=1.6, 2 * max(abs(initial_means)))`. This prevents initial
means from being clamped at the AABB boundary but means unit consistency is
essential.

## Deformation MLP

`DeformNetwork` consumes the concatenated HexPlane feature (64 dimensions at
the defaults), passes it through three `Linear(64)+ReLU` blocks, then uses
separate 3D position, 4D quaternion, and 1D opacity heads. Head weights and
biases initialize to zero, making initial deformation an identity. Outputs are
additive deltas on means, raw quaternion parameters, and opacity logits.
Time is encoded by HexPlane; the explicit `t` MLP argument is currently unused.

The trainer deforms means, quaternions, and opacity only. Scales and colors
remain canonical per-Gaussian parameters.

## Coarse/fine schedule

`TwoStageScheduler(500,3000)` gives 3,500 total trainer iterations:

- steps 0–499: coarse stage, always training-dataset item 0;
- steps 500–3499: fine stage, deterministic cycling with
  `(step - coarse_steps) % num_frames`.

Although the scheduler reports `shuffle=True` for fine, the provided trainer
does not create a DataLoader and uses the returned deterministic frame index.

## DynamicStrategy and DeformationTable

`DynamicStrategy` subclasses `DefaultStrategy`. The trainer initializes a
plain boolean `state["dynamic_mask"]` with every Gaussian `True`; all initial
Gaussians therefore pass through HexPlane and DeformNetwork. Strategy
duplicate/split/prune operations resize tensors in state, so this mask follows
Gaussian identity automatically.

`DeformationTable` remains importable but is **not used** by the current
trainer. It is a compatibility helper; the canonical runtime representation is
the plain `dynamic_mask` tensor.

## Densification, pruning, and opacity reset

Inherited `DefaultStrategy` defaults are unchanged:

```text
grow_grad2d=0.0002
grow_scale3d=0.01
prune_opa=0.005
prune_scale3d=0.1
refine_start_iter=500
refine_stop_iter=15000
refine_every=100
reset_every=3000
absgrad=False
```

Visible Gaussian screen-gradient statistics are accumulated. Starting after
step 500 and every 100 steps, high-gradient small Gaussians are duplicated,
high-gradient large Gaussians are split, and low-opacity Gaussians are pruned.
After step 3000, Gaussians above the large-scale threshold may also be pruned.
Opacity is reset to `2 * prune_opa` at step 3000. Packed rasterization metadata
and `packed=True` are passed consistently to the strategy.

The provided trainer passes `scene_scale=1.0`; this makes coordinate-unit
selection relevant to scale-based strategy thresholds and must be logged in
the first M4-A run rather than silently altered.

## Losses

The vanilla weighted objective is:

```text
0.8    * masked_l1
+ 0.2  * masked_ssim_loss
+ 1.0  * depth_loss
+ 1e-3 * image_total_variation
+ 1e-4 * spatial_plane_smoothness
+ 1e-4 * temporal_plane_smoothness
+ 1e-4 * temporal_plane_l1
```

- `masked_l1`: mean absolute RGB difference only where include mask is nonzero.
- `masked_ssim`: multiplies prediction and GT by the mask, then computes
  `1-SSIM` over the whole zeroed image. Its magnitude therefore depends on
  mask coverage by design.
- Binocular depth mode (default): L1 between reciprocal predicted and GT
  depth, only where both depths are valid and the tissue mask is nonzero.
- Monocular depth mode: `1 - Pearson correlation` on masked depth samples.
- Image TV: anisotropic absolute horizontal/vertical differences. The trainer
  calls it without a mask.
- Spatial/temporal plane smoothness: summed mean squared second differences.
- Temporal L1: summed mean absolute deviation of temporal planes from one.

## Rendering API

The trainer calls `gsplat.rasterization()` with deformed means/quaternions/
opacity, exponentiated log-scales, sigmoid opacity, direct RGB colors,
`viewmats=inverse(camtoworld)`, pinhole `Ks`, `sh_degree=None`, `packed=True`,
and `render_mode="RGB+ED"`.

The output is `B x H x W x 4`: RGB in channels 0–2 and expected projection
Z-depth in channel 3. `RGB+ED` normalizes accumulated depth by alpha. The M4
extension was compiled only for standard 3DGS channels 3 and 4, which exactly
covers RGB and RGB+depth used here; 3DGUT is not used.

## Checkpoint format

There is **no upstream checkpoint format** in
`dynamic_surgical_trainer.py`; it contains no `torch.save` or `torch.load`.
The file header explicitly describes checkpointing as absent from the initial
integration surface. Its optional post-training output is a GIF only.

Method 4A therefore needs a thin local checkpoint wrapper containing at least
the Gaussian `ParameterDict`, HexPlane state, DeformNetwork state,
`dynamic_mask`, config, frame/time metadata, unit convention, gsplat commit,
and initial/final counts. This wrapper must import and preserve upstream model,
loss, schedule, strategy, and renderer behavior rather than duplicating them.
