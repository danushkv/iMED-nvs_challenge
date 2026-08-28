# Method 3 submission notes

## Candidate decision

The completed ablation report calls for two independent images because the
local evaluation protocols disagree:

- M3-C is the primary baseline-adapted PSNR candidate.
- M3-B is the corrected-geometry/SSIM/coverage candidate.

The shared Dockerfile bakes the renderer into `METHOD3_RENDERER`; the build
script maps `m3c` to `surface_confidence` and `m3b` to `surface`. Runtime users
cannot switch candidates through command-line flags.

## Leakage audit

The production adapter and parent inference loader discover and open only:

```text
endoscope2/L/frame_*.png
endoscope2/depthL/frame_*.npy
K.txt (K2_L and K1_L)
pose.txt (camera ids 0 and 1)
```

They do not discover or open Endoscope1 RGB/depth/masks, Endoscope2/R, metrics,
evaluation output, prior renders, or checkpoints. Surface confidence is
deterministic and computed only from Endoscope2/L metric depth geometry.

The Method-3 image layer does not include `render_sequence.py`, so development
diagnostics and summary JSON exports are unavailable in the production path.
Only required challenge renders are written.

## Image contents and base ambiguity

The confirmed MV1A base tag was recorded with image ID prefix `aaa952c110d1`,
but its immutable registry digest was not saved. The current Dockerfile uses
the recorded Synapse `latest` tag and requires manual identity inspection.
For complete reproducibility, replace `MV1A_BASE` with the confirmed
`docker.synapse.org/...@sha256:<digest>` reference once obtained.

No custom CUDA extension is introduced by Method 3. It uses only PyTorch
tensor operations already available in the MV1A CUDA 11.8/PyTorch 2.1.2 image.

## PyTorch 2.1 compatibility

The production source uses sequential one-axis reductions for the 2x2
covariance finiteness check. This is equivalent to a tuple-axis reduction but
is required because the confirmed PyTorch 2.1.2 parent does not accept
`Tensor.all(dim=(-1, -2))`.
