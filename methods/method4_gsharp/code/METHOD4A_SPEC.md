# Method 4A — G-SHARP-iMED

## Objective

Adapt the pinned upstream G-SHARP implementation in gsplat to the MICCAI iMED
Novel View Synthesis challenge and evaluate one representative sequence before
making a continue/stop decision.

## Frozen methods and legal-data boundary

Never modify Endo4DGS, Method 1, MV1A, Method 2, or Method 3. Method 4 must
remain isolated under `method4_gsharp/`.

Endoscope1 RGB is evaluation-only. It must never be used for training,
initialization, tuning, appearance correction, pose estimation, model
selection, or transform selection. Legal target intrinsics/extrinsics may be
used for rendering. The dataset itself is read-only.

## Required M4-A experiment

1. Inspect and document upstream `examples/dynamic_surgical_trainer.py` and
   its dependencies in `GSHARP_NOTES.md`; import working upstream components
   rather than reimplementing them.
2. Use the isolated uv environment and pinned gsplat commit documented in
   `README.md`.
3. Adapt legal Endoscope2/L RGB, depth, tissue/tool mask, calibration, and
   timestamp into `image`, `depth`, `mask`, `camtoworld`, `K`, and `time`.
4. Verify the complete camera convention against the frozen, validated RGB-D
   geometry implementation before training. Report numerical backprojection
   error and stop on disagreement.
5. Keep depth, camera translation, Gaussian means, and HexPlane bounds in
   consistent units. Do not normalize depth independently per frame. Log XYZ
   ranges, median neighbour distance, and scene extent.
6. Preserve upstream multi-frame depth unprojection, KNN scale initialization,
   tissue masking, RGB initialization, and opacity initialization. Start near
   50,000 initial points.
7. Run vanilla upstream architecture/losses with 500 coarse and 3,000 fine
   steps. Do not begin with architecture or loss-weight changes.
8. Evaluate source-view RGB/depth reconstruction and masking before target
   rendering.
9. Render Endoscope1 using legal camera parameters without loading Endoscope1
   RGB. Write challenge-format numbered PNG files.
10. Only after rendering, run the same evaluation protocol used for M3-A/M3-B.

## Current baseline results

| Method | PSNR | SSIM |
| --- | ---: | ---: |
| Hidden Endo4DGS baseline | 18.760 | 0.623 |
| Hidden RGB-D reprojection | 19.247 | 0.581 |
| Hidden MV1A | 20.089 | 0.608 |
| M3-A adapted baseline | 21.76169 | 0.68156 |
| M3-B surface-aware | 21.79984 | 0.68777 |

M3-B improves 199/199 evaluated frames over M3-A in both PSNR and SSIM.

## First-sequence stop gate

After one complete sequence, report source metrics, target metrics, runtime,
peak VRAM, and initial/final Gaussian counts. Continue only if geometry and
training are stable and target quality is competitive, SSIM is substantially
stronger, or an obvious small fix is indicated. Stop on unresolved geometry,
unstable deformation, dramatically weak target quality, impractical runtime,
or need for architectural redesign.

If promising, run the identical configuration on three more sequences. Tiny
follow-ups are limited to 100k initialization points if coverage is limiting,
a small depth-loss adjustment if depth error is clearly responsible, or a
longer fine stage if curves are still improving. No broad sweep.

## M4-B gate

Do not implement Method 4B until one M4-A sequence has completed training,
source reconstruction, target rendering, and evaluation. See
`METHOD4B_PLAN.md` for the deferred controlled experiment.
