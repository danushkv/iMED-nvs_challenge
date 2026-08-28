# Method 3 Surface Fusion — Ablation and Submission-Candidate Report

**Project:** MICCAI 2026 iMED Novel View Synthesis challenge  
**Report date:** 2026-08-25  
**Workspace:** `/mnt/cluster/workspaces/venkateda/method3_surface_fusion`  
**Status:** M3-B and M3-C selected for separate submission Docker builds

## 1. Purpose

Method 3 investigates whether a small geometry-derived surface extension can
improve the confirmed MV1A RGB-D reprojection method without redesigning it.
The primary objective is average PSNR; SSIM is secondary.

The original Endo-4DGS repository and confirmed MV1A implementation remain
read-only references. Method 3 does not train or optimize a neural network.

## 2. Hidden-leaderboard context

These are historical hidden-leaderboard results. They are not directly
comparable to the local protocol values later in this report.

| Method | Hidden PSNR | Hidden SSIM |
|---|---:|---:|
| Endo-4DGS challenge baseline | 18.760 | 0.623 |
| Initial RGB-D reprojection | 19.247 | 0.581 |
| MV1A | **20.089** | **0.608** |

M3-B and M3-C have not yet been evaluated on the hidden leaderboard.

## 3. Confirmed MV1A reference

The following two Docker tags resolve to the same confirmed MV1A image:

```text
method1-rgbd-reprojection:phase11-candidate
docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest
image ID: aaa952c110d1
```

The initial Method-1 image is distinct:

```text
method1-rgbd-reprojection:dev
image ID: dfea0cf2696f
```

Confirmed MV1A configuration:

```text
input RGB: Endoscope2/L
input depth: Endoscope2/depthL
source tool masking: disabled
depth filtering: none
renderer: depth-aware bilinear soft splatting
visibility tolerance: 1.0 mm + 0.01 * nearest target depth
depth softness: 8.0
hole fill: nearest-valid propagation, radius 3 on 640x512 grid
internal resolution: 640x512
export resolution: 1280x1024
temporal fusion: none
stereo: none
```

## 4. Stereo feasibility decision

Endoscope2/R RGB and `K2_R` exist, but inspected challenge data does not
provide right-camera depth or an explicit left-to-right rigid transform.
Therefore `T_target<-2R` cannot be computed legally from the supplied inputs.

**Decision:** stereo was not implemented. No missing calibration was invented
or estimated.

## 5. Compared methods

### M3-A — exact MV1A reference

M3-A reuses the confirmed MV1A implementation without changing its geometry,
splatting, visibility, filling, resizing, or filename behavior.

### M3-B — surface-aware splatting

M3-B replaces a valid source point's four-neighbour bilinear footprint with a
bounded projected elliptical Gaussian derived from organized RGB-D tangents.

Fixed configuration:

```text
normal: normalize(cross(du, dv)) using central 3D differences
depth discontinuity gate: max(2.0 mm, 0.02 * source depth)
footprint scale: 1.0
target radius range: 0.5 to 2.0 pixels
support window: 5x5
Mahalanobis cutoff: 9.0
invalid/unreliable surface fallback: exact MV1A bilinear contribution
visibility, fill, and export: unchanged from MV1A
```

Configuration record:
`configs/m3_surface_default.json`

### M3-C — confidence-gated surface splatting

M3-C freezes M3-B's surface settings and adds deterministic source-geometry
confidence:

```text
C_depth   = clamp(1 - max_neighbour_depth_delta / allowed_delta, 0, 1)
C_tangent = |du x dv| / (|du| |dv| + eps)
C_viewing = |normal . normalize(-X_source)|
C_geometry = C_depth * C_tangent * C_viewing
```

Fixed confidence behavior:

```text
threshold: 0.25
accepted surface weight: normalized footprint weight * C_geometry
below threshold: exact MV1A bilinear fallback
target nearest-depth visibility: unchanged
```

Configuration record:
`configs/m3_confidence_default.json`

No confidence parameter was estimated from Endoscope1 RGB.

## 6. Local evaluation subset

All three methods were evaluated on the same fixed four-sequence public-GT
subset. Every sequence contains 199 synchronized frames, for 796 paired frames
in total.

```text
session_004_scene_2_tool_1
session_004_scene_6_tool_2
session_005_scene_7_tool_2
session_007_scene_11_tool_3
```

This is a controlled local ablation subset, not the hidden challenge set.

## 7. Metric protocol caveat

The reused evaluator reports two protocols:

1. `corrected_geometry`: scales intrinsics to the actual image grids.
2. `adapted_baseline`: faithfully reproduces the inspected challenge-adapted
   Endo-4DGS `metrics.py` behavior, including its unscaled-K/provisional-size
   overlap-mask construction.

Both use the challenge-adapted masked PSNR/SSIM implementation. Invalid
prediction pixels inside the evaluation mask are penalized. Endoscope1 RGB and
tool masks are opened only by offline evaluation after predictions are frozen.

The evaluator records that organizer-official status of these two local mask
protocols remains unverified. Consequently, both are retained in this report.

## 8. Corrected-geometry results

### Per-sequence PSNR

| Sequence | M3-A | M3-B | M3-C |
|---|---:|---:|---:|
| session_004_scene_2_tool_1 | 20.1849 | **20.2641** | 20.2354 |
| session_004_scene_6_tool_2 | 19.8311 | **19.8871** | 19.8652 |
| session_005_scene_7_tool_2 | 17.6205 | 17.7486 | **17.7506** |
| session_007_scene_11_tool_3 | 18.5017 | **18.5544** | 18.5474 |
| **Mean** | 19.0346 | **19.1135** | 19.0997 |

### Per-sequence SSIM

| Sequence | M3-A | M3-B | M3-C |
|---|---:|---:|---:|
| session_004_scene_2_tool_1 | 0.60437 | **0.61142** | 0.60998 |
| session_004_scene_6_tool_2 | 0.60378 | **0.61232** | 0.60986 |
| session_005_scene_7_tool_2 | 0.39676 | **0.42016** | 0.41755 |
| session_007_scene_11_tool_3 | 0.36368 | **0.37093** | 0.36915 |
| **Mean** | 0.49215 | **0.50371** | 0.50163 |

Aggregate deltas versus M3-A:

| Method | Delta PSNR | Delta SSIM |
|---|---:|---:|
| M3-B | **+0.07898** | **+0.01156** |
| M3-C | +0.06510 | +0.00949 |

Frame-level consistency versus M3-A:

```text
M3-B PSNR wins: 796/796
M3-B SSIM wins: 796/796
M3-C PSNR wins: 796/796
M3-C SSIM wins: 796/796
```

**Corrected-geometry winner: M3-B.**

## 9. Baseline-adapted results

### Per-sequence PSNR

| Sequence | M3-A | M3-B | M3-C |
|---|---:|---:|---:|
| session_004_scene_2_tool_1 | 21.7617 | **21.7998** | 21.7914 |
| session_004_scene_6_tool_2 | 20.3699 | **20.4413** | 20.4083 |
| session_005_scene_7_tool_2 | 20.9750 | 20.8782 | **20.9860** |
| session_007_scene_11_tool_3 | 17.8254 | **17.8570** | 17.8563 |
| **Mean** | 20.2330 | 20.2441 | **20.2605** |

### Per-sequence SSIM

| Sequence | M3-A | M3-B | M3-C |
|---|---:|---:|---:|
| session_004_scene_2_tool_1 | 0.68156 | **0.68777** | 0.68701 |
| session_004_scene_6_tool_2 | 0.62403 | **0.63205** | 0.62888 |
| session_005_scene_7_tool_2 | 0.58787 | **0.60430** | 0.60214 |
| session_007_scene_11_tool_3 | 0.36723 | **0.37517** | 0.37086 |
| **Mean** | 0.56517 | **0.57482** | 0.57222 |

Aggregate deltas versus M3-A:

| Method | Delta PSNR | Delta SSIM |
|---|---:|---:|
| M3-B | +0.01108 | **+0.00965** |
| M3-C | **+0.02751** | +0.00705 |

M3-B loses `0.09684 dB` on `session_005_scene_7_tool_2`. M3-C
recovers this failure and exceeds M3-A on that sequence by `0.01102 dB`.

Frame-level consistency versus M3-A:

```text
M3-B PSNR wins: 633/796
M3-B SSIM wins: 796/796
M3-C PSNR wins: 715/796
M3-C SSIM wins: 796/796
```

**Baseline-adapted primary-PSNR winner: M3-C.**  
**Baseline-adapted SSIM winner: M3-B.**

## 10. Direct M3-B versus M3-C interpretation

Under corrected geometry, M3-C is below M3-B by:

```text
PSNR: -0.01388 dB
SSIM: -0.00207
```

Under the baseline-adapted protocol, M3-C is relative to M3-B by:

```text
PSNR: +0.01643 dB
SSIM: -0.00260
```

M3-B improves structure and coverage more consistently. M3-C is more
conservative and has the strongest baseline-adapted average PSNR because it
avoids M3-B's sequence-specific PSNR failure.

## 11. Coverage

Mean coverage over the four evaluation sequences:

| Protocol | Measure | M3-A | M3-B | M3-C |
|---|---|---:|---:|---:|
| Corrected | Raw | 93.6887% | **93.9563%** | 93.7728% |
| Corrected | Filled | 97.0638% | **97.0754%** | 97.0655% |
| Adapted | Raw | 93.3541% | **93.4982%** | 93.3628% |
| Adapted | Filled | 96.0432% | **96.0535%** | 96.0440% |

M3-B provides the highest raw coverage. Filled coverage differs little because
all methods retain MV1A's radius-3 nearest-valid propagation.

Internal-grid mean raw/filled coverage before evaluation masking:

| Method | Raw | Filled |
|---|---:|---:|
| M3-A | 70.7798% | 74.1041% |
| M3-B | **71.2525%** | **74.1182%** |
| M3-C | 70.9960% | 74.1059% |

## 12. Runtime and memory

Stored mean renderer time excludes most diagnostic file I/O:

| Method | Mean seconds/frame | Relative to M3-A | Maximum GPU memory |
|---|---:|---:|---:|
| M3-A | 0.01134 | 1.00x | 173.6 MB |
| M3-B | 0.03419 | 3.01x | 199.3 MB |
| M3-C | 0.03703 | 3.26x | 201.1 MB |

Mean development sequence wall time, including extensive diagnostic saving:

```text
M3-A: 176.3 seconds
M3-B: 256.3 seconds
M3-C: 404.8 seconds
```

M3-C's development wall time is inflated by writing combined confidence NPYs
and multiple confidence-component PNGs. Submission containers should write
only required challenge renders and should not export development diagnostics.

## 13. Scientific and leakage audit

Rendering inputs for M3-A, M3-B, and M3-C:

```text
Endoscope2/L RGB
Endoscope2/depthL
K2_L
K1_L
challenge-provided camera poses/extrinsics
```

Rendering does not read:

```text
Endoscope1 RGB
Endoscope1 depth
Endoscope1 tool masks
target RGB statistics or histograms
Endoscope2/R
```

Endoscope1 RGB and target tool masks are used only by offline evaluation after
predictions are frozen. They are not used to estimate confidence, select
transforms, adjust cameras, correct color, or optimize per-frame parameters.

## 14. Submission decision

Build two independent, reproducible submission images:

1. **M3-C primary-PSNR candidate**
   - strongest mean PSNR under the baseline-adapted protocol;
   - no sequence-mean PSNR regression versus M3-A in the four-sequence subset;
   - slightly lower SSIM and coverage than M3-B.

2. **M3-B structural/coverage candidate**
   - strongest corrected-geometry PSNR;
   - strongest SSIM under both protocols;
   - strongest raw coverage;
   - one baseline-adapted sequence-specific PSNR regression.

If submission quota permits, submit both tags. Hidden evaluation is required
to resolve the local metric-protocol ambiguity. Do not claim a Method-3 hidden
improvement until those submissions complete.

## 15. Docker handoff requirements

Each image must:

```text
read challenge sequences under /input
write predictions under /output/<sequence>/renders/
write 00000.png, 00001.png, ... as standard RGB PNG
preserve the expected 1280x1024 output size
never write into /input
never read Endoscope1 RGB during inference
avoid development-only confidence/debug exports
```

Candidate-specific renderer selection:

```text
M3-B: --renderer surface
M3-C: --renderer surface_confidence
```

The production adapter should reuse the existing inference-only iMED loader
and geometry functions. It should not depend on Endo-4DGS training or its CUDA
Gaussian rasterizer.

## 16. Reproducibility paths

Source and configuration:

```text
surface_geometry.py
surface_splat.py
confidence.py
render_sequence.py
configs/m3_surface_default.json
configs/m3_confidence_default.json
scripts/run_surface.sh
scripts/run_confidence.sh
```

Stored predictions and summaries:

```text
outputs/M3_A_MV1A/<sequence>/
outputs/M3_B_surface/<sequence>/
outputs/M3_C_confidence/<sequence>/
```

Stored evaluation results:

```text
outputs/M3_A_MV1A/<sequence>/evaluation_full/results.json
outputs/M3_B_surface/<sequence>/evaluation_full/results.json
outputs/M3_C_confidence/<sequence>/evaluation_full/results.json
```

## 17. Components not implemented

The following remain intentionally absent:

```text
stereo/right-camera fusion
temporal multi-frame fusion
learned confidence
Endo-4DGS fallback
target-driven appearance correction
target-driven parameter selection
```

The next development step should be chosen only after preserving Docker-ready
M3-B and M3-C candidates.
