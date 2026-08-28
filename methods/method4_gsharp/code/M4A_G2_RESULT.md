# M4-A G2 result — 100k initialization

The final M4-A ablation changed only `init_max_points` from 50,000 to
100,000 on the preselected representative sequence
`session_004_scene_2_tool_1`. The corrected scene scale, seed, architecture,
losses, learning rates, DynamicStrategy, and 500 + 3,000 step schedule were
unchanged.

## Source reconstruction

| Metric | M4-A 50k | G2 100k | Delta |
| --- | ---: | ---: | ---: |
| PSNR | 24.594303 | 25.469932 | +0.875629 dB |
| SSIM | 0.841456 | 0.846280 | +0.004824 |
| Depth MAE | 5.240815 mm | 5.210746 mm | -0.030069 mm |
| Coverage | 100% | 100% | 0 |
| Final Gaussians | 83,852 | 128,990 | +45,138 |

## Target NVS

| Protocol | Metric | M4-A 50k | G2 100k | Delta | G2 wins/losses/ties |
| --- | --- | ---: | ---: | ---: | ---: |
| corrected_geometry | PSNR | 20.665110 | 20.251483 | -0.413627 dB | 34 / 165 / 0 |
| corrected_geometry | SSIM | 0.700477 | 0.688781 | -0.011696 | 30 / 169 / 0 |
| adapted_baseline | PSNR | 22.210973 | 21.922119 | -0.288854 dB | 46 / 153 / 0 |
| adapted_baseline | SSIM | 0.728242 | 0.713689 | -0.014554 | 29 / 170 / 0 |

Training took 296.24 seconds and peaked at 7.42 GB allocated VRAM. The 640 x
512 target render took 44.55 seconds. All 199 target frames were rendered from
legal target calibration only, with `target_rgb_access=false`.

## Decision

Increasing initialization density improves source fitting but degrades target
generalization consistently. Freeze the original corrected 50k configuration
as M4-A, stop M4-A ablations, and proceed to M4-B. Do not use G2 as the M4-A
submission configuration.
