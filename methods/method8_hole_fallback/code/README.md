# Method 8: M3B + strict hole-only learned fallback

Method 8 is the final low-cost experiment. It does not retrain or rerender a
model. It combines two frozen artifacts:

- M3B from `../outputs/M3_B_surface/<sequence>`
- submitted Method 2, experiment `B1_metric_l1_w5e-5`, from
  `/mnt/cluster/workspaces/venkateda/method2_endo4dgs_plus/results/`

The 640x512 Method-2 inputs are the native Endo-4DGS files under
`test/ours_1000/renders`. The 1280x1024 challenge files are under `renders`.
Both streams use the same contiguous five-digit indices as M3B.

## F1 invariant

```text
fallback_mask = NOT M3B.filled_valid_mask
F1 = M3B.copy()
F1[fallback_mask] = Method2[fallback_mask]
```

No RGB-black heuristic is used. M3B raw and radius-3-filled pixels are both
protected. The program asserts zero changes on all protected pixels at both
640x512 evaluator resolution and 1280x1024 challenge resolution.

There is no alpha blending, boundary erosion, confidence gating, target tool
mask dependency at inference, SGS, temporal geometry, or new training.

## Evaluation-gated first experiment

The local experiment deliberately follows the requested order:

1. verify M3B/Method-2/target timestamp and filename alignment;
2. use evaluation-only Endoscope1 RGB/tool masks to measure both predictors
   specifically on M3B-unsupported pixels inside the corrected evaluation
   region;
3. stop without creating F1 when Method 2 has worse hole PSNR or MAE;
4. otherwise construct F1, assert the invariant, and compute official-adapted
   global PSNR/SSIM using the same metric functions as M3B.

Target RGB affects only the explicit KEEP/STOP diagnostic. It is never used to
alter pixels, tune a threshold, train, calibrate, or blend.

Run from this directory:

```bash
bash scripts/run_f1_one.sh session_004_scene_2_tool_1 cpu
```

If a visible GPU is preferred:

```bash
bash scripts/run_f1_one.sh session_004_scene_2_tool_1 cuda:0
```

Read:

```text
results/F1_M2_holes/session_004_scene_2_tool_1/hole_diagnostic.json
results/F1_M2_holes/session_004_scene_2_tool_1/fallback_report.json
```

The completed four-sequence ablation selects F1 strict holes. F2 boundary
exclusion at one and two pixels was weaker; F3 was not pursued. See:

```text
results/METHOD8_FINAL_REPORT.md
results/summary/method8_summary.json
results/summary/method8_ablation.csv
DOCKERIZATION_HANDOFF.md
```
