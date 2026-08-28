# iMED NVS dataset and camera notes

Inspection date: 2026-08-22. Dataset root inspected read-only:
`/mnt/cluster/datasets/iMED_NVS`. Reference implementation inspected read-only:
the enclosing challenge-adapted Endo-4DGS repository.

## Verified layout

The root contains 20 `session_*` sequence directories. Each normally contains:

```text
<sequence>/
├── K.txt
├── pose.txt
├── endoscope1/
│   ├── L/frame_NNNNNN.png
│   ├── R/frame_NNNNNN.png
│   ├── depthL/frame_NNNNNN.npy
│   ├── toolL/frame_NNNNNN.png
│   └── toolR/frame_NNNNNN.png
└── endoscope2/
    ├── L/frame_NNNNNN.png       # Method input RGB
    ├── R/frame_NNNNNN.png       # Unused by Method 1
    ├── depthL/frame_NNNNNN.npy  # Method input depth
    ├── toolL/frame_NNNNNN.png   # Optional Method input mask
    └── toolR/frame_NNNNNN.png   # Unused by Method 1
```

`session_006_scene_7_tool_3` is an exception: both cameras' `toolL` and
`toolR` directories are empty. Its RGB and depth streams each contain 199
frames. The current Endo-4DGS `IMED_Dataset` would reject this sequence because
it asserts equal RGB/mask counts. Method 1 works on it only with source tool
masking disabled; it never fabricates a mask.

## Images, depth, and masks

- Left/right RGB: PNG, RGB `uint8`, `1024 x 1280` (`H x W`).
- Left depth: NPY, `float32`, `512 x 640`.
- Tool masks: grayscale PNG `uint8`, `1024 x 1280`, binary values `{0,255}`.
- Depth is optical-axis Z-depth in millimetres. This is stated by the adapted
  baseline README (`depthL (mm, used directly)`) and by its point-cloud code.
- The baseline downsamples RGB to depth resolution with bilinear interpolation
  and masks with nearest-neighbour interpolation. It divides both focal lengths
  and principal-point coordinates by 2.
- An exhaustive read of all 3,953 Endoscope 2 depth maps found one shape
  (`512 x 640`), one dtype (`float32`), minimum `12.9154806` mm, maximum
  `253.2931824` mm, and **zero** NaN, infinite, zero, or negative values.
- Thus there is no invalid-depth encoding present in the supplied source
  collection. The geometry code nevertheless rejects non-finite values and
  `depth <= 0`, matching the baseline overlap-mask rule `z > 0`.
- Visual inspection confirms white (`255`) pixels in `toolL` mark the tool.
  The baseline converts the raw mask as `1 - raw/255`, so `True/1` means valid
  non-tool tissue and `False/0` means excluded tool.

The expected Method-1 render resolution for compatibility with the adapted
baseline is `512 x 640` (`H x W`). Its target GT, masks, and renders are all
handled at that resolution.

## Intrinsics

`K.txt` contains four ordinary 3x3 row-formatted matrices: `K1_L`, `K1_R`,
`K2_L`, and `K2_R`. Method 1 uses `K2_L` for the source and `K1_L` for the
target. Intrinsics vary by recording session, so they must be loaded per
sequence rather than hard-coded.

Example from `session_004_scene_2_tool_1`, at full RGB resolution:

```text
K2_L = [[1039.5010572364,    0.0000000000, 565.5091746366],
        [   0.0000000000, 1040.0475664751, 532.1105243640],
        [   0.0000000000,    0.0000000000,   1.0000000000]]

K1_L = [[1033.3860978293,    0.0000000000, 615.3035648109],
        [   0.0000000000, 1032.5533163688, 505.2699727155],
        [   0.0000000000,    0.0000000000,   1.0000000000]]
```

At the `640 x 512` render/depth resolution these become:

```text
K2_L/2 = [[519.7505286182,   0.0000000000, 282.7545873183],
          [  0.0000000000, 520.0237832376, 266.0552621820],
          [  0.0000000000,   0.0000000000,   1.0000000000]]

K1_L/2 = [[516.6930489147,   0.0000000000, 307.6517824055],
          [  0.0000000000, 516.2766581844, 252.6349863578],
          [  0.0000000000,   0.0000000000,   1.0000000000]]
```

Only K matrices are supplied. No distortion coefficients, rectification maps,
or distortion model occur in the sequence directories. The adapted baseline
uses the images directly with an ideal pinhole model and performs no iMED
undistortion step. Therefore **the code assumes the supplied images and K are
already suitable for pinhole projection, but the inspected files do not
independently prove or explicitly state that the images are rectified**.

## Exact pose convention

Each `pose.txt` has exactly two static rows:

```text
camera_id tx ty tz qx qy qz qw
```

The adapted loader and evaluator both parse the quaternion with
`scipy.spatial.transform.Rotation.from_quat`, hence quaternion order is
unambiguously `(x,y,z,w)`. They construct each row as a **camera-to-world**
transform:

```text
X_world = T_world_camera @ X_camera
```

Camera-id mapping is:

- id `0`: Endoscope 2 (`T_world_cam2`), the source; it is identity in every
  inspected sequence.
- id `1`: Endoscope 1 (`T_world_cam1`), the target.

Method 1 uses column-vector mathematics and therefore computes:

```text
T_cam1_cam2 = inverse(T_world_cam1) @ T_world_cam2
X_cam1      = T_cam1_cam2 @ X_cam2
```

The text matrices are read row by row into NumPy's conventional row-major
storage, but storage order does not change the column-vector equation above.
Points held as `[N,3]` rows are transformed by the equivalent vectorized form
`X_cam1 = X_cam2 @ R_cam1_cam2.T + t_cam1_cam2`.

Example matrices from `session_004_scene_2_tool_1`:

```text
T_world_cam2 =
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]

T_world_cam1 =
[[ 0.92284772  0.15936175  0.35065070 -21.77597966]
 [-0.12705629  0.98538676 -0.11344437  13.28311256]
 [-0.36360525  0.06013950  0.92960984 -14.17173548]
 [ 0.          0.          0.           1.        ]]

T_cam1_cam2 = inverse(T_world_cam1) @ T_world_cam2 =
[[ 0.92284772 -0.12705629 -0.36360525 16.63069876]
 [ 0.15936175  0.98538676  0.06013950 -8.76646390]
 [ 0.35065070 -0.11344437  0.92960984 22.31684174]
 [ 0.          0.          0.          1.        ]]
```

`det(R_cam1_cam2)` is `1.0` to numerical precision. This convention is chosen
from two independent code paths (`scene/imed_loader.py` and `metrics.py`), not
from Endoscope 1 RGB performance.

## Frame synchronization and timestamps

- Synchronization is by exact basename (`frame_NNNNNN`) across Endoscope 2 RGB,
  depth and masks, and across the two camera RGB streams.
- Both cameras have exactly the same ordered RGB frame IDs in every sequence.
- Most sequences contain IDs `000002, 000005, ..., 000596` (step 3).
- `session_004_scene_2_tool_2` omits the block `000518` through `000548` and
  `000554` through `000596` (26 frames total); both cameras omit the same IDs.
- `session_005_scene_7_tool_3` omits `000290`; both cameras omit it.
- No physical timestamps are supplied. The baseline assigns normalized time by
  sorted list index, `index / (N-1)`, not by parsing temporal units from the ID.
- The camera transforms are static per sequence, not per frame.

## Adapted evaluation mask and output convention

The inspected `metrics.py` does the following for iMED:

1. Loads the target `endoscope1/toolL` mask and converts it to tissue-valid via
   `1 - raw/255`.
2. Builds a single global overlap mask from the **first** Endoscope 2 depth map:
   backproject with K2, transform cam2 -> world -> cam1, project with K1, round,
   then apply 2 iterations of 3x3 dilation, 2 iterations of 11x11 closing, and
   binary hole filling.
3. Multiplies the per-frame target tissue-valid mask by the global overlap mask.
4. Computes strictly masked PSNR and SSIM. SSIM uses an 11x11 Gaussian window
   with sigma 1.5, then averages the SSIM map over the final mask.

There are two implementation details to resolve before calling this the
organizer's official metric:

- `metrics.py` backprojects a `640 x 512` depth map using the **unscaled**
  full-resolution K matrices, unlike `scene/imed_loader.py`, which divides K by
  2. It then derives a provisional target size from `2*cx, 2*cy` and resizes the
  mask. This appears inconsistent and must be confirmed rather than silently
  copied into the geometry renderer.
- `render.py` first writes `overlap_mask.png` from nonzero pixels in the first
  model render. `metrics.py` loads and applies that existing mask before it
  computes and overwrites it with the geometric global mask. Consequently the
  first metric run can depend on render support, while later runs may not.

The adapted baseline output filenames are sequential, zero-based, five-digit
names (`00000.png`, `00001.png`, ...), not original `frame_NNNNNN` names. Its
metric layout is:

```text
<model>/test/ours_<iteration>/
├── renders/00000.png
├── gt/00000.png
├── masks/00000.png
└── overlap_mask.png
```

Method 1 uses the same sequential filename stems inside its own `rgb/`,
`valid_mask/`, and debug folders, and records the original frame name in JSON.

## Challenge-rule questions that remain open

The local dataset and adapted baseline demonstrate that `K1_L`, `K2_L`, and
both static poses are technically available in these sequence folders. They do
**not** establish the organizer's held-out inference contract. Before challenge
submission, confirm from the organizer rules that target Endoscope 1 intrinsics
and `T_world_cam1` are provided inference inputs. Method 1 must not run on held-
out data if those fields are evaluation-only.

Likewise, confirm whether Endoscope 2 `toolL` masks are inference inputs. The
implementation makes their use opt-in (`--mask_source_tools`) and does not infer
or replace a missing source mask.

Endoscope 1 RGB, depth, and tool masks are never opened by the inference or
identity scripts. `evaluate.py` opens Endoscope 1 RGB and `toolL` only after
predictions already exist; it cannot modify calibration, renderer settings, or
predictions.

## Phase-8 evaluation protocols

Because the adapted overlap-mask implementation above is internally
inconsistent, Method 1 records two fixed, explicitly named protocols rather
than selecting one from GT performance:

- `corrected_geometry` scales K2 from full RGB to source depth resolution and
  K1 from full target RGB to prediction resolution before building the global
  overlap mask.
- `adapted_baseline` reproduces the inspected `metrics.py` float32 projection,
  unscaled K matrices, provisional target dimensions, nearest resize, and
  SciPy morphology.

Both protocols use the adapted baseline's exact masked PSNR and SSIM formulas.
The per-frame mask is target tissue-valid AND global geometric overlap.
Prediction invalidity is not used to shrink this mask; invalid black pixels
inside it are penalized, while raw and filled prediction coverage are reported
separately.

The adapted baseline's method-dependent first-render support behavior is
available only through the explicit diagnostic flag
`--include_first_frame_support`; it is disabled by default. Neither protocol is
labelled organizer-official until the overlap-mask ambiguity is clarified.
