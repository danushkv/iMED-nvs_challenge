# Method 8: Strict hole-only learned fallback

Method 8 is a source-only hybrid of two frozen methods. It preserves every
pixel supported by M3-B, then uses the accepted Method-2 Endo-4DGS render only
where M3-B remains unsupported after its existing radius-3 fill.

```text
fallback_mask = NOT M3B.filled_valid_mask
output = M3B.copy()
output[fallback_mask] = Method2[fallback_mask]
```

The validity mask is geometric; black RGB values are never treated as holes.
The mask is resized to challenge resolution with nearest-neighbor sampling.
The implementation asserts byte equality with M3-B everywhere its filled
validity mask is true.

## Frozen constituents

- M3-B: surface-aware splatting with its existing radius-3 fill.
- Method 2: `B1_metric_l1_w5e-5`, 300 coarse and 1000 fine iterations.

There is no retraining beyond the frozen Method-2 procedure, alpha blending,
confidence replacement, target mask requirement, or evaluation-time method
selection. Endoscope1 RGB is evaluation-only.

## Four-sequence result

Under the corrected-geometry development protocol:

| Variant | Mean PSNR | Delta PSNR | Mean SSIM | Delta SSIM | Sequence PSNR wins/losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 strict holes | **19.617677** | **+0.504134** | **0.526226** | **+0.022518** | **4 / 0** |
| F2 exclude 1 px | 19.508873 | +0.395330 | 0.513824 | +0.010117 | 4 / 0 |
| F2 exclude 2 px | 19.437339 | +0.323795 | 0.512169 | +0.008459 | 3 / 1 |

F1 changed zero protected M3-B pixels in every experiment. F2 was rejected
because excluding pixels near geometric support reduced both PSNR and SSIM.
F1 is the frozen submission candidate.

## Release contents

- `code/hole_fallback.py`: offline aligned fallback construction and metric
  diagnostic.
- `code/aggregate_fallbacks.py`: fixed four-sequence F1/F2 aggregation.
- `code/METHOD8_FINAL_REPORT.md`: complete decision record.
- `code/DOCKERIZATION_HANDOFF.md`: combined-container production contract.
- `../../experiments/method8_ablation.csv`: per-sequence ablation values.
- `../../experiments/method8_summary.json`: aggregate selection record.

Development scripts retain their original workspace defaults as provenance.
Adjust their roots when running from an assembled release. The Docker handoff
is a specification; the final combined Docker implementation is not claimed
to be present in this snapshot.
