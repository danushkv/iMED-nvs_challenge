# Results

## Hidden leaderboard

Historical challenge submissions:

| Method | PSNR | SSIM | Notes |
| --- | ---: | ---: | --- |
| Endo-4DGS baseline | 18.760 | 0.623 | Context only; baseline code excluded |
| M1 initial RGB-D reprojection | 19.247 | 0.581 | Training-free geometry |
| MV1A | **20.089** | **0.608** | Depth-aware soft splatting and bounded fill |

No hidden result is claimed for M2, M3, M4-A, or M4-B in this release record.

## Method 2 paired ablation

Method 2 uses its Endo-4DGS-native local evaluator. Four-sequence development
means:

| Variant | PSNR | SSIM | ΔPSNR vs B0 | ΔSSIM vs B0 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| B0 paired baseline | 19.23250 | 0.56610 | — | — | Reference |
| B1 metric L1, `5e-5` | **19.26171** | 0.56593 | **+0.02922** | -0.00017 | Keep |
| B3 appearance | 19.23207 | **0.56758** | -0.00043 | +0.00148 | Reject for primary metric |
| B4 source DSSIM | 19.20909 | 0.56572 | -0.02341 | -0.00038 | Reject |
| B5 no confidence | 19.05258 | 0.56173 | -0.17992 | -0.00437 | Reject |
| B6 combined | 19.19967 | 0.56496 | -0.03283 | -0.00114 | Reject |

On the preselected holdout, B1 improved B0 from 19.09972/0.50229 to
**19.18463/0.50529**: `+0.08490 dB` PSNR and `+0.00300` SSIM. Across all five
sequences it improved mean PSNR by `+0.04035 dB`, with 4/5 wins.

## Method 3 four-sequence local comparison

### Corrected-geometry protocol

| Method | Mean PSNR | Mean SSIM | ΔPSNR vs M3-A | ΔSSIM vs M3-A |
| --- | ---: | ---: | ---: | ---: |
| M3-A exact MV1A | 19.0346 | 0.49215 | — | — |
| M3-B surface-aware | **19.1135** | **0.50371** | **+0.07898** | **+0.01156** |
| M3-C confidence-gated | 19.0997 | 0.50163 | +0.06510 | +0.00949 |

M3-B and M3-C improved PSNR and SSIM on all 796 frames under this protocol.

### Adapted-baseline protocol

| Method | Mean PSNR | Mean SSIM | PSNR wins vs M3-A | SSIM wins vs M3-A |
| --- | ---: | ---: | ---: | ---: |
| M3-A exact MV1A | 20.2330 | 0.56517 | — | — |
| M3-B surface-aware | 20.2441 | **0.57482** | 633/796 | **796/796** |
| M3-C confidence-gated | **20.2605** | 0.57222 | **715/796** | **796/796** |

M3-B is the structural/coverage candidate; M3-C is the baseline-adapted
primary-PSNR candidate.

## Method 4 four-sequence local comparison

M4-A uses 50k initialization points, 500 coarse steps, 3000 fine steps, and a
millimetre-aware DynamicStrategy scene scale. Its four-sequence mean was:

| Protocol | PSNR | SSIM |
| --- | ---: | ---: |
| Corrected geometry | 18.17030 | 0.57430 |
| Adapted baseline | 18.28060 | **0.62341** |

Under the adapted-baseline protocol, the paired four-sequence comparison was:

| Method | PSNR | SSIM |
| --- | ---: | ---: |
| M3-B | **20.24409** | 0.57482 |
| M4-A | 18.28060 | **0.62341** |

M4-A was retained as an independent G-SHARP submission candidate: it was much
weaker in PSNR but stronger in average SSIM. This is not evidence that the
methods can be blended or selected using target imagery.

## Method 4 representative-sequence ablations

Sequence `session_004_scene_2_tool_1`, 199 frames:

| Variant | Initial/final Gaussians | Source PSNR/SSIM | Adapted target PSNR/SSIM | Decision |
| --- | --- | --- | --- | --- |
| M4-A 50k | 50,000 / 83,852 | 24.5943 / 0.84146 | **22.21097 / 0.72824** | Keep |
| G2 100k | 100,000 / 128,990 | **25.4699 / 0.84628** | 21.92212 / 0.71369 | Drop |
| M4-B surface init | 50,000 / 99,278 | 24.1593 / **0.84161** | 22.04141 / 0.70918 | Drop |

The 100k experiment improved source fitting but reduced target generalization.
M4-B improved corrected-geometry PSNR by 0.219 dB, but under the shared
headline adapted protocol it lost 0.170 dB PSNR and 0.0191 SSIM, losing SSIM
on 193/199 frames.

Machine-readable values and per-sequence breakdowns are in `experiments/`.
