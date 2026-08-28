# Method 3 inspection notes

Inspection date: 2026-08-25.

This document records the Phase-0 inspection only. No Method-3 rendering code
has been implemented or executed.

## Leaderboard context

These are hidden-leaderboard values and must not be mixed with local/public
sequence metrics:

| Method | PSNR | SSIM |
| --- | ---: | ---: |
| Endo-4DGS baseline | 18.760 | 0.623 |
| Initial RGB-D reprojection | 19.247 | 0.581 |
| MV1A | 20.089 | 0.608 |

MV1A is the current strongest method by the primary PSNR metric.

## Exact MV1A identity

The following two Docker tags resolve to the same image ID:

```text
method1-rgbd-reprojection:phase11-candidate
docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest
image ID: aaa952c110d1
```

This confirms that MV1A is the inspected Phase-11 candidate. The initial
RGB-D submission is a different image:

```text
method1-rgbd-reprojection:dev
image ID: dfea0cf2696f
```

The MV1A source remains read-only at:

```text
/mnt/cluster/workspaces/venkateda/Endo-4DGS/method1_rgbd_reprojection
```

## MV1A inspection

### Input observations

MV1A reads one synchronized source observation at a time:

```text
endoscope2/L/frame_NNNNNN.png
endoscope2/depthL/frame_NNNNNN.npy
K.txt: K2_L and K1_L
pose.txt: camera ids 0 and 1
```

It does not read Endoscope1 RGB, depth, or masks. Source tool masking is
disabled. Source RGB is resized from the native 1280x1024 grid to the 640x512
metric-depth grid.

Depth is metric optical-axis Z in millimetres. Pose rows are camera-to-world
transforms with quaternions in `(qx,qy,qz,qw)` order:

```text
camera id 0 = Endoscope2 source
camera id 1 = Endoscope1 target
T_target_source = inverse(T_world_target) @ T_world_source
```

### Splatting

Each valid left RGB-D pixel is represented as an independent 3D point. After
target projection it contributes to its four neighbouring target pixels using
bilinear weights.

MV1A uses the `soft_depth` mode with:

```text
visibility_tolerance_mm = 1.0
visibility_relative = 0.01
depth_softness = 8.0
```

It does not estimate normals, tangents, local scale, or an anisotropic
footprint.

### Visibility

At each target pixel, MV1A finds the closest contributing target-camera depth
and computes:

```text
delta_z = candidate_z - nearest_z
tolerance = 1.0 mm + 0.01 * nearest_z
```

Candidates with `delta_z > tolerance` are rejected. Remaining contributions
receive:

```text
weight = bilinear_weight * exp(-8.0 * delta_z / tolerance)
```

This is a per-target-pixel nearest-Z gate. It prevents unrestricted blending
of substantially different foreground and background depths.

### Hole handling

After raw splatting, MV1A performs nearest-valid propagation with a Euclidean
radius of 3 pixels on the 640x512 internal grid. RGB and depth are copied from
the nearest raw-valid target pixel. Filled confidence is attenuated by
distance. Pixels beyond the radius remain black.

The filled RGB result is then bilinearly resized to the native 1280x1024
output resolution.

### Confidence

Raw confidence is bounded accumulated occupancy:

```text
C_raw = 1 - exp(-sum_splat_weights)
```

For a filled pixel:

```text
C_filled = C_nearest_valid * exp(-distance / 3)
```

Confidence does not currently include depth gradients, surface-normal
stability, viewing angle, image boundaries, tool boundaries, or temporal
consistency.

### Temporal, normals, stereo, and multiple observations

```text
Temporal usage: none
Surface normals: none
Endoscope2/R usage: none
Multiple source observations per target frame: none
```

Every requested timestamp is processed independently from its matching
Endoscope2/L RGB-D frame.

### Difference from the initial RGB-D adapter

The initial adapter used an automatic source-tool-mask policy and did not
apply the Phase-11 nearest-valid fill. Confirmed MV1A instead disables source
tool masking and applies radius-3 nearest-valid propagation. Its core
depth-aware bilinear renderer settings remain unchanged.

### Main reusable files and functions

The existing MV1A files must remain unmodified:

```text
docker_submission/phase11_candidate/nvs_method.py
  render_target_views

docker_submission/phase11_candidate/imed_io.py
  collect_source_frames
  load_source_frame
  load_calibration
  scale_intrinsics

camera.py
  backproject_depth
  transform_points
  project_points

reprojection.py
  RenderResult
  project_source_rgbd

soft_splatting.py
  render_soft_splat

hole_filling.py
  nearest_valid_fill

docker_submission/phase11_candidate/entrypoint.py
  discover_sequences
```

## Stereo feasibility

The inspected public sequence contains synchronized left and right RGB and
tool-mask streams. Endoscope2/L and Endoscope2/R each contain 199 RGB frames
with matching filenames.

```text
RGB available: yes, endoscope2/R
depth available: no, only endoscope2/depthL exists
intrinsics available: yes, K2_R
L/R extrinsic available: no
target transform computable: no for 2R
synchronized: yes for RGB filenames
legal to use: right RGB is a source observation, but calibrated right RGB-D
              reprojection is not defined by the inspected inputs
decision: DO NOT USE
```

`pose.txt` provides only two endoscope-level rows: Endoscope2 and Endoscope1.
It does not provide separate left/right optical-camera poses. No explicit
`T_2R_2L`, stereo baseline, right-camera extrinsic, or rectification transform
was found in the iMED metadata or loader.

Without right depth and the left-to-right rigid transform,
`T_target_2R` cannot be computed. Method 3 must not estimate the missing
calibration or infer it from Endoscope1 RGB. The stereo branch is therefore
dropped immediately.

## Smallest surface-aware extension

### Existing representation

```text
position: backprojected source XYZ
color: source RGB
target footprint: four bilinear neighbours
visibility: nearest target-Z tolerance
confidence: accumulated splat occupancy
```

### Proposed surfel representation

```text
position: X_source
color: source RGB
normal: n_source
tangents: du_source and dv_source
target center: projected X_target
target footprint: bounded anisotropic Gaussian
visibility: unchanged MV1A nearest-Z tolerance
confidence for M3-B: unchanged MV1A occupancy confidence
```

### Normal and tangent estimation

Use central differences on the organized RGB-D grid:

```python
du = 0.5 * (X[v, u + 1] - X[v, u - 1])
dv = 0.5 * (X[v + 1, u] - X[v - 1, u])
normal = normalize(cross(du, dv))
```

Reject a surface element if the center or required neighbours are invalid,
the tangent/normal is non-finite, the normal magnitude is unstable, or a
neighbour depth discontinuity is too large. The first fixed discontinuity gate
should be:

```text
max neighbour depth difference <= max(2.0 mm, 0.02 * center depth)
```

### Projected footprint

Transform and project the center and its horizontal/vertical neighbours. Form
target-image tangent vectors:

```text
e_u = 0.5 * (q(u + 1,v) - q(u - 1,v))
e_v = 0.5 * (q(u,v + 1) - q(u,v - 1))
```

Use the initial covariance:

```text
Sigma = footprint_scale^2 * (e_u e_u^T + e_v e_v^T) + sigma_min^2 I
```

Initial bounded settings:

```text
footprint_scale = 1.0
minimum radius = 0.5 target pixel
maximum radius = 2.0 target pixels
support window = at most 5x5
```

The existing MV1A depth-visibility rule must remain unchanged in M3-B. This
isolates the surface-footprint change and protects the primary PSNR signal.

Do not use the Endo-4DGS Gaussian rasterizer for the first experiment. A
bounded PyTorch target-space elliptical splat is the smaller and lower-risk
extension.

### Planned files

```text
surface_geometry.py  # organized XYZ, tangents, normals, validity
surface_splat.py     # bounded anisotropic target-space splatting
render_sequence.py   # exact MV1A mode plus surface mode
configs/m3_surface_default.json
```

The first implementation must not include stereo, temporal fusion, a learned
confidence model, or Endo-4DGS fallback.

Estimated implementation size is approximately 300-500 lines plus CLI and
diagnostic integration. Limiting support to at most 5x5 bounds both runtime
and memory.

## Recommended first experiment

```text
M3-A = exact confirmed MV1A

M3-B = exact MV1A with only the four-pixel bilinear footprint replaced by a
       bounded projected elliptical surfel footprint
```

The first sequence should be:

```text
session_004_scene_2_tool_1
```

Keep unchanged:

```text
source-tool masking disabled
depth filtering none
visibility tolerance 1.0 mm + 0.01 * nearest depth
depth softness 8.0
nearest-valid fill radius 3
640x512 internal grid
1280x1024 export
five-digit sequential filenames
```

Required M3-B diagnostics:

```text
RGB
target depth
raw and filled validity
confidence
normal visualization
surface-validity mask
projected footprint radius
rejected-normal fraction
raw and filled coverage
surface-recovered pixel count
runtime per frame
peak GPU memory
```

Compare official PSNR first and SSIM second. Keep the surface renderer only if
PSNR improves, or if PSNR is effectively tied while SSIM improves meaningfully.

## M3-B cross-sequence result

M3-B was evaluated against exact M3-A on four fixed 199-frame sequences.
Under corrected geometry, M3-B improved PSNR and SSIM on all 796 frames:

```text
mean delta PSNR: +0.07898 dB
mean delta SSIM: +0.01156
```

Under the baseline-adapted protocol, mean deltas were `+0.01108 dB` PSNR and
`+0.00965` SSIM. SSIM improved on all 796 frames, while PSNR improved on
633/796 frames. `session_005_scene_7_tool_2` regressed by `-0.09684 dB` under
that protocol. The organizer-official status of the two local mask protocols
remains unverified, so these results must not be presented as hidden scores.

Decision: retain M3-B as the current geometry candidate and test one
conservative confidence-aware extension.

## M3-C confidence-aware surface experiment

M3-C freezes every established M3-B setting. It adds only deterministic
source-geometry confidence:

```text
C_depth   = clamp(1 - max_neighbour_depth_delta / allowed_delta, 0, 1)
C_tangent = |du x dv| / (|du| |dv| + eps)
C_viewing = |normal . normalize(-X_source)|
C_geometry = C_depth * C_tangent * C_viewing
```

No target image, target mask, or target statistic enters this calculation.
If `C_geometry < 0.25`, the point uses MV1A's exact bilinear footprint. An
accepted surface contribution is weighted by `C_geometry` after its footprint
has been normalized. This makes confidence effective without allowing a large
ellipse to acquire arbitrary total weight.

M3-C deliberately leaves unchanged:

```text
surface covariance and radius bounds
target-depth visibility tolerance
depth softness
radius-3 nearest-valid fill
camera geometry
source tool-mask behavior
output resizing and naming
```

The initial gate is one fixed frame followed by all four validation sequences.
Do not sweep the confidence threshold using target RGB. If the fixed M3-C
configuration fails to retain M3-B's PSNR, drop it.
