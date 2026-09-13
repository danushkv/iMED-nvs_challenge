# Method 8 Dockerization handoff

## Objective

Package the frozen **F1 strict hole-only fallback** for the iMED NVS challenge.
Do not rerun ablations or alter either constituent method.

The final container receives read-only data at `/input`, writes final RGB PNGs
under `/output`, has one A100 80 GB GPU, 120 GB system RAM, no network, and a
three-hour total time limit.

## Frozen components

### M3B

Production packaging reference:

```text
/mnt/cluster/workspaces/venkateda/method3_surface_fusion/
  docker_submission/method3_candidates/
```

Use renderer `surface`, not `surface_confidence`. Its current production
adapter renders M3B correctly but saves only RGB. The combined entrypoint must
retain the in-memory radius-3 `filled.filled_valid_mask` before native export.
Do not infer holes from black RGB.

Frozen M3B settings are documented in
`docker_submission/method3_candidates/README.md`.

### Method 2

Production packaging reference:

```text
/mnt/cluster/workspaces/venkateda/method2_endo4dgs_plus/
  docker_submission/method2_metric_l1/
```

Use exactly:

```text
300 coarse iterations
1000 fine iterations
depth loss = metric_l1
depth weight = 5e-5
metrics disabled
source-only staging
```

The Method-2 Dockerfile rebuilds the CUDA rasterizers for A100 `sm_80` and
other architectures. Preserve that build strategy.

## Required production algorithm

For each sequence:

1. Discover the sequence using only `K.txt`, `pose.txt`,
   `endoscope2/L`, `endoscope2/depthL`, and the Method-2-required legal source
   `endoscope2/toolL`.
2. Run frozen M3B `surface` at its 640x512 depth resolution.
3. Retain both M3B filled RGB and `filled_valid_mask` in a private temporary
   work directory or memory. Do not publish intermediate files as challenge
   predictions.
4. Run the frozen Method-2 source-only entrypoint into a separate private
   temporary output root.
5. Require identical sequence names, frame counts, and contiguous five-digit
   output indices.
6. Export M3B RGB to 1280x1024 using its existing bilinear path.
7. Resize `filled_valid_mask` to 1280x1024 with **nearest-neighbor** sampling.
8. Define `fallback_mask = ~filled_valid_mask_native`.
9. Copy native Method-2 RGB into M3B only at `fallback_mask`.
10. Assert byte equality between final and M3B RGB wherever
    `filled_valid_mask_native` is true.
11. Save standard RGB PNGs only at
    `/output/<sequence>/renders/<five-digit-index>.png`.

Never use target RGB, target depth, target masks, target image statistics, or
evaluation code at inference. There is no alpha blending and no boundary
exclusion. F2 is rejected.

## Recommended image construction

Use the Method-2 submission image/Dockerfile as the heavy base because it
already contains the Endo-4DGS runtime and compatible CUDA extensions. Add
only the lightweight M3B production dependencies and a new combined
entrypoint. Do not depend on two containers at evaluation time.

The M3B production layer currently inherits several utilities from the MV1A
image (`camera.py`, `reprojection.py`, `soft_splatting.py`, `hole_filling.py`,
and `imed_io.py`). Vendor the exact frozen versions into the combined build
context or use a deliberate multi-stage build to copy them. Do not silently
reimplement their calibration logic.

Private intermediate storage may be under a uniquely created `/tmp` directory
inside the container. Ensure it is automatically cleaned, and do not write to
`/input`. Because Method 2 creates checkpoints, monitor temporary disk usage
and remove each sequence's private training artifacts after its final fused
renders are safely written.

## Mandatory tests before tagging

1. Image inspection: entrypoint, command, CUDA architectures, and no network
   dependency.
2. CUDA import/smoke test on available GPU.
3. One public sequence with `/input` mounted read-only.
4. Exact output contract: correct frame count, `00000.png...`, RGB,
   1280x1024.
5. Compare Docker F1 output byte-for-byte with the saved local F1 output for
   the same public sequence.
6. Independently assert zero changed M3B-valid pixels.
7. Confirm the runtime never opens Endoscope1 RGB/mask/depth paths.
8. Confirm total projected runtime for all requested sequences remains within
   three hours on the A100 evaluator.

## Authoritative local evidence

```text
method8_hole_fallback/results/METHOD8_FINAL_REPORT.md
method8_hole_fallback/results/summary/method8_summary.json
method8_hole_fallback/results/summary/method8_ablation.csv
```

## Explicit non-goals

Do not add F2 boundary exclusion, confidence gating, blending, target-tool
handling, SGS, G-SHARP, temporal geometry, triangle/plane completion,
retraining beyond the frozen Method-2 schedule, or any evaluation-only gate.
