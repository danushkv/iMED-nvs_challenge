# Results

## Hidden leaderboard

Historical challenge submissions:

| Method | PSNR | SSIM | Notes |
| --- | ---: | ---: | --- |
| Endo-4DGS baseline | 18.760 | 0.623 | Context only; baseline code excluded |
| M1 initial RGB-D reprojection | 19.247 | 0.581 | Training-free geometry |
| MV1A (Method 1a) | 20.089 | 0.608 | Depth-aware soft splatting and bounded fill |
| M2 metric depth | 18.722 | **0.617** | Best SSIM among our methods; below baseline PSNR/SSIM |
| M3-B surface-aware | **20.136** | 0.615 | Best hidden PSNR |
| M3-C confidence-gated | 20.130 | 0.614 | Narrowly below M3-B |
| M4-A G-SHARP | 19.007 | 0.614 | Frozen 50k configuration |
| M4-B surface initialization | 19.040 | 0.614 | Slight hidden PSNR gain over M4-A |

Hidden-validation deltas worth retaining:

- M3-B versus MV1A: `+0.047 dB` PSNR and `+0.007` SSIM.
- M3-C versus MV1A: `+0.041 dB` PSNR and `+0.006` SSIM.
- M2 versus the hidden Endo-4DGS baseline: `-0.038 dB` PSNR and `-0.006` SSIM.
- M4-B versus M4-A: `+0.033 dB` PSNR and no difference at the reported
  three-decimal SSIM precision.

M3-B is therefore the strongest hidden-PSNR method in this release. M2 has the
strongest hidden SSIM among our variants, although the external Endo-4DGS
baseline remains higher at 0.623.

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

The hidden M2 result was `18.722/0.617`, below the hidden Endo-4DGS reference
by `0.038 dB` PSNR and `0.006` SSIM. The locally accepted metric-depth change
therefore did not transfer into a hidden improvement over the challenge
baseline.

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

Hidden validation resolved the near tie in favor of M3-B:

| Method | Hidden PSNR | Hidden SSIM |
| --- | ---: | ---: |
| M3-B | **20.136** | **0.615** |
| M3-C | 20.130 | 0.614 |

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

Hidden validation produced `19.007/0.614` for M4-A and `19.040/0.614` for
M4-B. Thus M4-B's hidden PSNR was 0.033 dB higher despite failing the local
adapted-protocol kill rule. The local DROP decision remains part of the
experimental record; the hidden result shows that the one-sequence local
ranking did not predict the hidden ordering within Method 4.

## GeoSCOPE (Method 8) strict hole-only fallback

GeoSCOPE leaves every radius-3-filled M3-B-valid pixel byte-identical and uses
the frozen Method-2 render only at the remaining unsupported pixels. It uses
the geometric validity mask, never an RGB-black heuristic. Across the fixed
four-sequence corrected-geometry development set:

| Variant | Mean PSNR | Delta PSNR vs M3-B | Mean SSIM | Delta SSIM vs M3-B | Sequence PSNR wins/losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 strict holes | **19.617677** | **+0.504134** | **0.526226** | **+0.022518** | **4 / 0** |
| F2 exclude 1 px boundary | 19.508873 | +0.395330 | 0.513824 | +0.010117 | 4 / 0 |
| F2 exclude 2 px boundary | 19.437339 | +0.323795 | 0.512169 | +0.008459 | 3 / 1 |

F1 yielded 720/796 per-frame PSNR wins and 796/796 SSIM wins, with zero
protected M3-B pixels changed. Boundary exclusion removed useful fallback
pixels, so F2 was rejected. F1 is the final submitted method; no hidden
GeoSCOPE result is recorded in this snapshot.

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
