# Method 2 change log

Every Method 2 training change must remain independently switchable and easy
to compare against B0.

## B1: metric depth supervision

| File | Function/area | Change | Motivation | CLI/config |
| --- | --- | --- | --- | --- |
| `arguments/__init__.py` | `ModelHiddenParams` | Added depth mode, independent primary-depth weight, Huber beta, and diagnostic interval defaults. | Expose B0 and B1 without replacing the original depth formulation or changing auxiliary weights. | `depth_loss`, `primary_depth_weight`, `depth_huber_beta`, `depth_diagnostics_interval` |
| `train.py` | `scene_reconstruction` | Preserved the B0 normalized branch and added strict-valid metric L1 and Huber branches. | Test whether absolute millimetre geometry improves NVS. | Values supplied through `ModelHiddenParams` |
| `train.py` | Method 2 diagnostic helpers | Added source-only depth statistics, CSV logging, and debug images. | Measure term magnitude and geometry errors without target RGB leakage. | `depth_diagnostics_interval` |
| `imed_nvs_baseline.py` | runtime config and both subcommands | Added challenge-compatible depth arguments and propagated them to training. | Run isolated ablations through the existing Docker interface. | `--depth-loss`, `--depth-weight`, `--depth-huber-beta`, `--depth-diagnostics-interval` |
| `imed_nvs_baseline.py` | `run_sequence` no-render return | Return the model directory instead of exporting nonexistent renders when both rendering and metrics are disabled. | Permit short source-only loss-magnitude diagnostics. | `--no-render --no-metrics` |
| `docker/Dockerfile.method2-sm86` and `.dockerignore` | development image | Copy Method 2 source over the validated local `sm86` baseline while excluding results and caches. | The Docker daemon cannot bind-mount the cluster workspace; a derived image avoids repeated `/tmp` source staging. | Image `method2-endo4dgs-plus:dev` |
| `scripts/run_ablation.sh` | experiment orchestration | Added overwrite-safe, per-sequence staging, GPU selection, persistent preservation, and root-owned temporary cleanup. | Run paired sequence ablations concurrently on two GPUs without accumulating `/tmp` data. | Positional `EXPERIMENT SEQUENCE GPU_ID`; optional `METHOD2_*` environment variables |
| `configs/method2_sequence_split.txt` | experiment protocol | Fixed four development sequences, one held-out confirmation sequence, and sequence-to-GPU assignments before inspecting additional metrics. | Reduce single-sequence selection bias and keep B0/B1 comparisons paired. | Read by the experimenter; runner takes one explicit sequence |

### Modes

- `normalized`: exact B0 primary-depth equation and default behavior.
- `metric_l1`: strict valid-tissue mean absolute Z-depth error in millimetres.
- `metric_huber`: strict valid-tissue Smooth-L1 Z-depth error in millimetres.

The CLI's `--depth-weight` controls only `primary_depth_weight`. The original
`depth_weight=0.01` remains unchanged for gradient and TV regularization, so a
B1 weight sweep varies only the intended primary supervision term.

## B2: source tool-mask formulation

| File | Function/area | Change | Motivation | CLI/config |
| --- | --- | --- | --- | --- |
| `arguments/__init__.py` | `ModelHiddenParams` | Added opt-in strict valid-tissue reduction and a zero-default tool dilation radius. | Keep B0 exact while making B2 independently switchable. | `tool_aware_loss`, `tool_mask_dilation` |
| `train.py` | Method 2 tool-mask helpers and `scene_reconstruction` | Added GPU mask dilation, strict valid-element L1, source-only mask diagnostics, and propagation of the effective tissue mask through existing masked losses. | Test boundary uncertainty and eliminate frame-dependent dilution of primary losses by excluded pixels. | Values supplied through `ModelHiddenParams` |
| `imed_nvs_baseline.py` | runtime config and both subcommands | Exposed tool-loss normalization and dilation through the challenge-compatible entrypoint. | Preserve the standard training/render/evaluation interface. | `--tool-aware-loss`, `--tool-mask-dilation 0|3|5` |
| `scripts/run_ablation.sh` | B2 experiment mapping | Added independent valid-mean, dilation-3, and dilation-5 experiments, each based on B0 depth. | Prevent B1 from confounding the B2 component decision. | `B2_tool_validmean_d0`, `B2_tool_dilate3`, `B2_tool_dilate5` |

The baseline already uses source masks for point-cloud initialization and all
major reconstruction losses. B2 does not claim to introduce tool masking from
scratch. Dilation is applied at the internal 640x512 training resolution and
only to source loss masks. It does not alter challenge evaluation masks or use
Endoscope 1 RGB. Gaussian densification masking remains unchanged for this
first B2 experiment.

### B2 result and decision

| B2 formulation | Mean PSNR change | Mean SSIM change | PSNR wins | Worst PSNR change |
| --- | ---: | ---: | ---: | ---: |
| strict valid mean, dilation 0 | -0.02139 dB | -0.00145 | 1/4 | -0.08035 dB |
| B0 mean, dilation radius 3 | -0.18711 dB | -0.00253 | 1/4 | -0.62622 dB |
| B0 mean, dilation radius 5 | -0.17201 dB | -0.00138 | 1/4 | -0.42638 dB |

The dilation failure was not caused by an empty mask. On
`session_005_scene_7_tool_2`, radius 3 reduced sampled tissue support from
approximately 88.6% to 86.9% (coarse) and 87.6% to 85.7% (fine), yet PSNR
dropped by 0.62622 dB. The pipeline is sensitive to even this narrow boundary
removal. All tested B2 changes are rejected, no B2 variant is evaluated on the
holdout, and M2 retains the baseline tool-mask behavior.

### Leakage controls

The B1 loss and diagnostics consume only the currently sampled Endoscope 2
training camera's RGB, depth, and tissue mask. They never inspect Endoscope 1
RGB, target histograms, or target error.

### Status

Implementation staged on 2026-08-24. The source-only 20-coarse/50-fine Huber
diagnostic passed. A complete 300-coarse/1000-fine Huber run at weight `1e-4`
on `session_004_scene_2_tool_1` produced PSNR `20.4857426` and SSIM
`0.6567062`, versus paired B0 PSNR `20.5385971` and SSIM `0.6538426`. This is
a `-0.0528545` dB PSNR change and `+0.0028636` SSIM change, so no B1 depth
formulation has yet been accepted. The next decision uses a fixed multi-sequence
paired ablation rather than this single sequence.

The four-sequence development comparison subsequently rejected both Huber
weights: `1e-4` changed mean PSNR by `-0.04582` dB and mean SSIM by
`-0.00182`, while `1e-3` changed mean PSNR by `-0.00593` dB and mean SSIM by
`-0.00471`. Metric L1 at weight `5e-5` is provisionally promising: mean PSNR
changed by `+0.02922` dB with 3/4 sequence wins, while mean SSIM changed by
`-0.00017`. The fixed development set will therefore test only the two
remaining L1 weights (`5e-6`, `5e-4`) before selecting one configuration for
the untouched holdout sequence.

The completed L1 sweep selected `5e-5`: across four development sequences it
changed mean PSNR by `+0.02922` dB with 3/4 wins and mean SSIM by `-0.00017`.
On the preselected holdout `session_006_scene_7_tool_1`, it improved PSNR by
`+0.08490` dB and SSIM by `+0.00300`. Across all five sequences, mean PSNR
improved by `+0.04035` dB and mean SSIM by `+0.00047`, with 4/5 PSNR wins and
a worst PSNR change of only `-0.00968` dB. Metric L1 at weight `5e-5` is the
accepted B1 component; both Huber variants and the other L1 weights are
rejected.

## B3: training-only appearance correction

| File | Function/area | Change | Motivation | CLI/config |
| --- | --- | --- | --- | --- |
| `arguments/__init__.py` | `ModelHiddenParams` | Added zero-default appearance enable, learning rate, identity regularization, and diagnostics interval. | Keep B0 exact and make A1 independently switchable. | `appearance_correction`, `appearance_lr`, `appearance_reg_weight`, `appearance_diagnostics_interval` |
| `train.py` | `_Method2PerFrameAppearance`, `training`, `scene_reconstruction` | Added per-source-frame RGB scale/bias, separate optimizer, identity regularization, diagnostics, and final source-parameter export. | Let source exposure fluctuations be absorbed without requiring target appearance at inference. | Values supplied through `ModelHiddenParams` |
| `imed_nvs_baseline.py` | runtime config and both subcommands | Exposed the A1 controls through the existing challenge interface. | Preserve independent B0/B3 execution. | `--appearance-correction`, `--appearance-lr`, `--appearance-reg-weight`, `--appearance-diagnostics-interval` |
| `scripts/run_ablation.sh` | B3 experiment mapping | Added the initial `lr=1e-3`, regularization `1e-2` experiment on B0. | Obtain a fast A1 signal before any additional appearance variant. | `B3_appearance_lr1e-3_reg1e-2` |

The appearance transform is applied only to source Endoscope 2 renderings used
inside the training objective. It is not part of the Gaussian checkpoint,
`render.py` never loads it, and Endoscope 1 rendering remains the raw Gaussian
output. Saved appearance tensors are explicitly labeled training-only. No
target image, target histogram, or target-specific optimization is used.

## B4-B6: overnight targeted ablations

| Experiment | Isolated change |
| --- | --- |
| `B4_source_dssim_w1e-2` | Adds `0.01 * (1 - SSIM)` to source reconstruction using the baseline's dormant zero-masked SSIM path. |
| `B5_no_confidence` | Disables both baseline confidence losses while retaining RGB, depth, smoothness, normal, and TV terms. |
| `B6_combined_b1_b3_b4` | Combines accepted B1 metric L1 with B3 appearance and B4 source DSSIM; rejected B2 is excluded. |

`imed_nvs_baseline.py` exposes `--ssim-weight` and
`--disable-confidence`, with B0 defaults of zero and false respectively.
`scripts/run_overnight_b3_b6.sbatch` runs B3-B6 on the fixed four-sequence
development set using two concurrent one-GPU workers. It does not touch the
holdout. `scripts/summarize_ablation.py` produces paired per-sequence and
aggregate CSV/JSON rankings using mean PSNR first and mean SSIM second.
