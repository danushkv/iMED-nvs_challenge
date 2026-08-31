# Method 3: Surface Fusion

Method 3 is a training-free extension of MV1A. It tests whether organized
RGB-D surface geometry can replace each point's four-pixel bilinear footprint
with a bounded projected elliptical footprint.

## Variants

| Variant | Change from MV1A | Release status |
| --- | --- | --- |
| M3-A | Exact MV1A wrapper | Reference |
| M3-B | Surface normal and bounded elliptical footprint | Structural/SSIM candidate |
| M3-C | M3-B plus source-only geometry confidence | Local primary-PSNR candidate |

M3-B fixed settings:

```text
normal:                       normalize(cross(du, dv))
depth discontinuity:          max(2.0 mm, 0.02 × depth)
footprint scale:              1.0
target radius:                0.5 to 2.0 pixels
support:                      5×5
Mahalanobis cutoff:           9.0
unreliable-surface fallback:  exact MV1A bilinear contribution
```

M3-C adds:

```text
C = depth_margin × tangent_stability × abs(normal · view)
threshold = 0.25
below threshold = exact MV1A fallback
```

## Four-sequence result

| Protocol | M3-A | M3-B | M3-C |
| --- | ---: | ---: | ---: |
| Corrected PSNR | 19.0346 | **19.1135** | 19.0997 |
| Corrected SSIM | 0.49215 | **0.50371** | 0.50163 |
| Adapted PSNR | 20.2330 | 20.2441 | **20.2605** |
| Adapted SSIM | 0.56517 | **0.57482** | 0.57222 |

Under corrected geometry, both variants beat M3-A on all 796 frames. Under the
adapted protocol, both beat M3-A in SSIM on all 796 frames.

Hidden validation narrowly favored M3-B:

| Variant | Hidden PSNR | Hidden SSIM |
| --- | ---: | ---: |
| M3-B | **20.136** | **0.615** |
| M3-C | 20.130 | 0.614 |

M3-B improved on hidden MV1A by `+0.047 dB` PSNR and `+0.007` SSIM and is the
strongest hidden-PSNR method in the release.

## Environment with uv

Use the MV1A-compatible stack. The confirmed environment was PyTorch 2.1.2 and
CUDA 11.8:

```bash
cd methods/method3_surface_fusion/code
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.1.2 torchvision==0.16.2
uv pip install numpy==1.24.4 Pillow==10.2.0 scipy
```

Method 3 reuses the assembled MV1A/Method 1 geometry. Pass its release path
explicitly rather than relying on the original absolute development default:

```bash
python render_sequence.py \
  --mv1a_root ../../method1_rgbd_reprojection/code \
  --data_root /path/to/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --output_dir outputs/M3_B_surface/session_004_scene_2_tool_1 \
  --renderer surface \
  --device cuda
```

Use `surface_confidence` for M3-C. The assembled original scripts and complete
ablation report provide the frozen multi-sequence commands.

## Docker

The production Docker adapter derives from the confirmed MV1A image and selects
`surface` or `surface_confidence` explicitly at build time. It must not run both
and choose from target-view quality.

## Scientific boundary

No stereo, temporal fusion, learned confidence, Endo-4DGS fallback, or
target-driven correction is present. Endoscope1 RGB is evaluation-only.
