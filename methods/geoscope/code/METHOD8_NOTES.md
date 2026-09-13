# Method 8 notes

## Frozen fallback selection

The Method-2 submission notes identify `B1_metric_l1_w5e-5` as the submitted
configuration:

- metric Endoscope2 depth L1
- depth weight `5e-5`
- 300 coarse and 1000 fine iterations
- source-only training and source tool masking
- target RGB unavailable to training/rendering

This is used rather than an arbitrary Method-2 ablation. Original B0 may be
tested separately only if F1 is useful; it is not averaged with Method 2.

## Alignment checks

The program requires:

- identical contiguous five-digit M3B and Method-2 filenames;
- identical frame counts;
- M3B `render_summary.json` source ids equal sorted target evaluation ids;
- Method-2 `run_manifest.txt` names the requested sequence and the frozen
  `B1_metric_l1_w5e-5` experiment;
- exact 640x512 internal dimensions and aligned 1280x1024 native streams.

## Final decision

F1 improved all four development-sequence means. The four-sequence mean gain
was `+0.504134` dB PSNR and `+0.022518` SSIM, with zero protected M3B pixels
changed.

F2 was subsequently tested with one- and two-pixel exclusion distances. Both
were weaker than strict F1: their mean PSNR gains were `+0.395330` and
`+0.323795` dB respectively. F2 is rejected, and F3 is unnecessary.

The frozen recommendation is **F1 strict holes**. See
`results/METHOD8_FINAL_REPORT.md`.

## F1 representative-sequence result

On `session_004_scene_2_tool_1`, Method 2 improved evaluated M3B-hole PSNR
from `11.285906` to `15.982444` dB and reduced hole MAE from `0.254896` to
`0.132878`. Strict F1 improved global corrected-geometry PSNR by `+0.361495`
dB and SSIM by `+0.012472`, won all 199 frames in both metrics, and changed
zero frozen M3B-valid pixels. Decision: **STRONGLY KEEP**, subsequently
confirmed on the fixed four-sequence development set.
