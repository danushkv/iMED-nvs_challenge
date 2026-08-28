# Method 4B result

Sequence: `session_004_scene_2_tool_1` (199 frames)

M4-B was a controlled experiment against the frozen corrected 50k M4-A. It
changed Gaussian initialization only: M3-derived source normals, anisotropic
surface scales, and confidence/view/normal-aware 0.5 mm voxel fusion. Opacity,
HexPlane, deformation MLP, losses, learning rates, DynamicStrategy, 500 +
3,000 step schedule, renderer, and evaluation were unchanged.

## Initialization and source reconstruction

| Metric | M4-A | M4-B | Delta |
| --- | ---: | ---: | ---: |
| Initial Gaussians | 50,000 | 50,000 | 0 |
| Final Gaussians | 83,852 | 99,278 | +15,426 |
| Source PSNR | 24.594303 | 24.159316 | -0.434987 dB |
| Source SSIM | 0.841456 | 0.841608 | +0.000152 |
| Source depth MAE | 5.240815 mm | 6.565997 mm | +1.325182 mm |
| Source depth coverage | 100% | 100% | 0 |
| Training runtime | 298.20 s | 295.05 s | -3.15 s |
| Peak allocated VRAM | 7,416,909,824 B | 578,816,512 B | -6,838,093,312 B |

The M4-B audit counted 65,208,320 raw observations, 59,909,994 after tissue
masking, 58,535,506 after surface/confidence filtering, 200,194 after the
deterministic per-frame candidate budget, 88,921 after voxel deduplication,
and 50,000 final initial Gaussians. All 199 source frames contributed. No
target camera or Endoscope1 RGB entered initialization or training.

## Target NVS

| Protocol | Metric | M4-A | M4-B | Delta | M4-B wins/losses/ties |
| --- | --- | ---: | ---: | ---: | ---: |
| corrected_geometry | PSNR | 20.665110 | 20.884137 | +0.219027 dB | 127 / 72 / 0 |
| corrected_geometry | SSIM | 0.700477 | 0.699523 | -0.000954 | 96 / 103 / 0 |
| adapted_baseline | PSNR | 22.210973 | 22.041414 | -0.169560 dB | 62 / 137 / 0 |
| adapted_baseline | SSIM | 0.728242 | 0.709179 | -0.019063 | 6 / 193 / 0 |

Both protocols retain 100% evaluation coverage. The adapted-baseline protocol
is the same local headline comparison used for the M3-A/M3-B and representative
M4-A results. Organizer-official status remains unverified, so the corrected
geometry result is retained in the record rather than hidden.

## Visual finding

M4-B sometimes retains slightly sharper upper-background tissue and its
corrected-geometry PSNR improves on 127 frames. However, it creates stronger
elongated/radial splat streaks and more fragmented foreground structure. It
does not recover the missing tool, disoccluded anatomy, or previously unseen
tissue. Its worst adapted frame loses 0.993111 dB PSNR and 0.046263 SSIM; its
best adapted PSNR gain is only 0.320259 dB.

## Decision

**DROP M4-B.** It fails the one-sequence controlled kill rule under the local
headline protocol and loses SSIM on 193/199 frames. Do not run the additional
three sequences, thickness-ratio tests, confidence-weighted opacity (M4-B2),
or static/dynamic prior. Preserve M4-A as the independent G-SHARP method.
