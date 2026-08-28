# Evaluation protocol

## Keep score families separate

This project contains three distinct score families:

1. **Hidden leaderboard:** organizer evaluation of submitted Docker images.
2. **Local corrected geometry:** local masked PSNR/SSIM with intrinsics scaled
   to the actual evaluation grids.
3. **Local adapted baseline:** faithful reproduction of the inspected
   challenge-adapted Endo-4DGS metric behavior, including its provisional-size
   overlap-mask construction.

The organizer-official status of the two local mask variants was not confirmed.
Both are retained so that a favorable protocol is not selectively reported.
Never place a hidden number and a local number in a single ranked column.

Method 2's stored `metrics.py` results form a fourth, Endo-4DGS-native local
family. M2 is compared to its paired B0 runs rather than numerically ranked
against M3/M4 values from a different evaluator.

## Fixed local subset

The shared M3/M4 local comparison uses four sequences, each with 199 frames:

```text
session_004_scene_2_tool_1
session_004_scene_6_tool_2
session_005_scene_7_tool_2
session_007_scene_11_tool_3
```

Method 2 uses the same four development sequences and additionally reserves
`session_006_scene_7_tool_1` as a holdout confirmation sequence.

## Target access

Rendering and training complete before evaluation begins. The offline evaluator
may then read Endoscope1 RGB and target masks to compute metrics. All reports
should record:

```text
evaluation_only_target_access: true
target_rgb_used_for_rendering_or_tuning: false
```

## Reproducing evaluation

After running `scripts/assemble_sources.sh`, the shared M1 evaluator is under:

```text
methods/method1_rgbd_reprojection/code/evaluate.py
```

Method-specific READMEs give the exact prediction directory and evaluator
arguments. Store evaluation results outside the code tree or in a Git-ignored
`outputs/` directory. Commit only compact aggregate CSV/JSON records needed to
support reported tables.

## Reporting checklist

- Name the score family and sequence subset.
- Report sequence count and frame count.
- Preserve per-frame CSV when making win/loss claims.
- Report mean PSNR and SSIM together.
- For geometric methods, report raw/filled coverage.
- For learned methods, report training time, rendering time, initial/final
  Gaussian counts, and peak VRAM.
- Mark rejected ablations explicitly.
