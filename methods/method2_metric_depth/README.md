# Method 2: Endo-4DGS with metric depth

Method 2 asks whether absolute metric Endoscope2 depth supervision improves
the challenge-adapted Endo-4DGS model. It is released as a compact overlay on
the official challenge image rather than as a second full copy of Endo-4DGS.

## Accepted configuration

```text
coarse iterations:       300
fine/total iterations:   1000
primary depth loss:      strict valid-tissue metric L1
primary depth weight:    5e-5
source units:             millimetres
tool behavior:           baseline source masking
appearance correction:  disabled
source DSSIM:            0
confidence losses:       enabled
```

Only B1 metric L1 at `5e-5` was accepted. B2 mask formulations, B3 appearance,
B4 source DSSIM, B5 confidence removal, and B6 combined changes were rejected.

Four-sequence development result versus paired B0:

| Variant | PSNR | SSIM | ΔPSNR | ΔSSIM |
| --- | ---: | ---: | ---: | ---: |
| B0 | 19.23250 | 0.56610 | — | — |
| B1 metric L1 | **19.26171** | 0.56593 | **+0.02922** | -0.00017 |

On the preselected holdout, B1 improved by `+0.08490 dB` PSNR and `+0.00300`
SSIM. See [`method2_ablation.csv`](../../experiments/method2_ablation.csv) for
the rejected variants.

Hidden validation gave `18.722 PSNR / 0.617 SSIM`. Relative to the hidden
Endo-4DGS challenge reference (`18.760/0.623`), this is `-0.038 dB` PSNR and
`-0.006` SSIM. The local/holdout metric-depth gain therefore did not transfer
into a hidden improvement over the challenge baseline.

## Source overlay

The scientific/runtime delta consists of:

```text
train.py
arguments/__init__.py
scene/imed_loader.py
imed_nvs_baseline.py
```

`scripts/assemble_sources.sh` copies these files, the experiment configs and
scripts, the reports, and the Docker submission adapter into `code/`.

## Recommended reproduction: Docker

Method 2 depends on Endo-4DGS CUDA extensions. Use its Docker path instead of
attempting to share the M1 or M4 environment:

```bash
cd methods/method2_metric_depth/code
bash docker_submission/method2_metric_l1/scripts/build.sh \
  imed-nvs:method2-metric-l1
```

The image derives from the official organizer baseline and rebuilds
`simple-knn` and `diff-gaussian-rasterization-depth` for its configured CUDA
architectures. Inspect the Docker README before changing the architecture list.

Run the fixed entrypoint with read-only input and writable output mounts:

```bash
docker run --rm --gpus all \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  imed-nvs:method2-metric-l1
```

## Leakage boundary

Training uses Endoscope2 RGB, metric depth, and source tool masks. The wrapper
constructs target cameras from legal calibration without exposing Endoscope1
RGB, depth, masks, or target statistics. Metrics are disabled in submission
inference.
