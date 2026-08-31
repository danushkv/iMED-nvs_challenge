# Method 8 final report

## Decision

Freeze **F1 strict hole-only Method-2 fallback**.

```text
fallback_mask = NOT M3B.filled_valid_mask
output = M3B.copy()
output[fallback_mask] = Method2[fallback_mask]
```

F1 never changes a raw-valid or radius-3-filled M3B pixel. The protected-pixel
invariant was checked at both the 640x512 working resolution and the 1280x1024
challenge resolution. Across every completed experiment:

```text
M3B-valid pixels changed = 0
```

No RGB-black heuristic, target RGB, target tool mask, alpha blending,
confidence replacement, retraining, or learned inpainting is used at
inference.

## Frozen inputs

- Geometry renderer: M3B `surface`, including its existing radius-3 fill.
- Learned fallback: Method 2 `B1_metric_l1_w5e-5` (300 coarse and 1000 fine
  Endo-4DGS iterations, metric Endoscope2 depth L1 weight `5e-5`).
- Evaluation: the same corrected-geometry official-adapted PSNR/SSIM protocol
  used for M3B development comparisons.

## Four-sequence development result

| Variant | Mean PSNR | Mean delta PSNR | Mean SSIM | Mean delta SSIM | Sequence PSNR wins/losses |
|---|---:|---:|---:|---:|---:|
| F1 strict holes | 19.617677 | **+0.504134** | 0.526226 | **+0.022518** | **4 / 0** |
| F2 exclude 1 px boundary | 19.508873 | +0.395330 | 0.513824 | +0.010117 | 4 / 0 |
| F2 exclude 2 px boundary | 19.437339 | +0.323795 | 0.512169 | +0.008459 | 3 / 1 |

F1 produced 720 PSNR frame wins and 76 losses, and 796 SSIM frame wins with
zero SSIM losses. The one heterogeneous sequence,
`session_005_scene_7_tool_2`, gained only `+0.018646` dB and had 76 PSNR frame
losses, but its sequence mean remained positive and SSIM improved by
`+0.020131`.

## Hole-specific evidence

On the representative sequence, evaluated unsupported pixels occupied
1.610338% of the evaluated image. Replacing black/unsupported M3B values with
Method 2 changed hole-only quality as follows:

```text
hole PSNR: 11.285906 -> 15.982444 dB
hole MAE:   0.254896 -> 0.132878
```

The global representative-sequence result was `+0.361495` dB PSNR and
`+0.012472` SSIM, with all 199 frames improving in both metrics.

## F2 conclusion

Boundary exclusion is rejected. Removing fallback pixels within one or two
pixels of valid M3B support reduced both aggregate PSNR and SSIM. Strict F1
therefore remains the smallest and strongest rule.

## Submission configuration

The production submission must:

1. run frozen M3B and retain its internal `filled_valid_mask`;
2. run frozen submitted Method 2 from legal Endoscope2 inputs only;
3. verify exact sequence/frame alignment;
4. resize the boolean unsupported mask to 1280x1024 with nearest-neighbor
   sampling;
5. copy Method-2 RGB only at those unsupported pixels;
6. assert that all M3B-valid output pixels are byte-identical;
7. write only contiguous RGB files at
   `/output/<sequence>/renders/00000.png`, `00001.png`, and so on.

The local evaluation gate is not part of inference and must not be included in
the submission runtime.

Canonical machine-readable results are in
`results/summary/method8_summary.json` and
`results/summary/method8_ablation.csv`.
