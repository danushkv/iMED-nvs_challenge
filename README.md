# iMED Novel View Synthesis Experiments

<p align="center">
  <img src="assets/method_lineage.svg" alt="Lineage of the released iMED NVS methods" width="100%">
</p>

This private research release records the progression of our MICCAI 2026 iMED
Novel View Synthesis challenge experiments: from training-free RGB-D
reprojection, through surface-aware splatting, to dynamic Gaussian training.
The original Endo-4DGS challenge baseline is retained as numerical context but
is not duplicated here.

## Included methods

| Folder | Short name | Representation | Training | Outcome |
| --- | --- | --- | --- | --- |
| [`method1_rgbd_reprojection`](methods/method1_rgbd_reprojection/README.md) | M1 | RGB-D point reprojection | None | First submitted geometry method |
| [`method1_5_mv1a`](methods/method1_5_mv1a/README.md) | MV1A | Depth-aware soft splats + bounded hole fill | None | Best hidden result among the reprojection stages |
| [`method2_metric_depth`](methods/method2_metric_depth/README.md) | M2 | Endo-4DGS + metric source-depth supervision | Per sequence | Small, holdout-confirmed gain over its paired baseline |
| [`method3_surface_fusion`](methods/method3_surface_fusion/README.md) | M3-B/C | Source-derived elliptical surface splats | None | Strong local SSIM/coverage improvements |
| [`method4_gsharp`](methods/method4_gsharp/README.md) | M4-A/B | Dynamic G-SHARP Gaussians | Per sequence | M4-A retained; M4-B dropped |

“Method 1.5” was an informal development name. In this release it is called
**MV1A**, matching the submitted image and experiment records.

## Headline challenge context

These are historical hidden-leaderboard results. They must not be compared as
if they were produced by the local public-sequence protocol.

| Method | Hidden PSNR | Hidden SSIM |
| --- | ---: | ---: |
| Endo-4DGS challenge baseline | 18.760 | 0.623 |
| M1 initial RGB-D reprojection | 19.247 | 0.581 |
| MV1A | **20.089** | **0.608** |

See [Results and protocol notes](docs/RESULTS.md) for the complete local
tables, per-sequence values, ablations, and the metric-protocol caveat.
The optional [qualitative comparison](docs/QUALITATIVE.md) displays one
unaltered prediction from each assembled method after its assets are approved
for inclusion.

## Repository layout

```text
imed_nvs_release/
├── README.md
├── CONTRIBUTING.md
├── THIRD_PARTY.md
├── assets/
├── docs/
│   ├── DATA_AND_LEGALITY.md
│   ├── EVALUATION.md
│   ├── QUALITATIVE.md
│   ├── REPRODUCIBILITY.md
│   ├── REPOSITORY_MAP.md
│   └── RESULTS.md
├── experiments/
│   ├── development_sequences.txt
│   ├── method2_ablation.csv
│   ├── method3_ablation.csv
│   ├── method4_ablation.csv
│   └── results.csv
├── methods/
│   ├── method1_rgbd_reprojection/
│   ├── method1_5_mv1a/
│   ├── method2_metric_depth/
│   ├── method3_surface_fusion/
│   └── method4_gsharp/
└── scripts/
    ├── assemble_sources.sh
    ├── collect_qualitative_assets.sh
    └── validate_release.sh
```

The documentation and result manifests are committed directly. To avoid
accidentally copying virtual environments, CUDA toolkits, datasets,
checkpoints, or multi-gigabyte result folders, the curated code snapshot is
assembled explicitly from the existing experiment roots.

From the `Endo-4DGS` workspace root:

```bash
bash imed_nvs_release/scripts/assemble_sources.sh
bash imed_nvs_release/scripts/validate_release.sh
```

Neither script trains a model or reads the challenge dataset. The assembly
script is non-destructive: it accepts an identical existing file but refuses
to overwrite a differing file.

Optionally collect one frame of already-generated predictions for the private
qualitative gallery:

```bash
bash imed_nvs_release/scripts/collect_qualitative_assets.sh
```

Review dataset redistribution rights before committing any qualitative asset.
Ground-truth Endoscope1 images are intentionally never copied by that script.

## Reproducibility philosophy

- Each method has its own environment and README; there is no misleading
  “one environment fits all” installation.
- Fixed configurations are stored beside code and named by the experiment that
  produced the reported result.
- Hidden scores and local scores remain in separate tables.
- Training/rendering paths never use Endoscope1 RGB. Target RGB is accessed
  only by offline evaluation after predictions are frozen.
- Rejected experiments are documented rather than silently removed.
- Large generated artifacts and environments are excluded; compact CSV/JSON
  summaries and exact commands are retained.

## Citation and release status

This is a challenge research artifact, not yet a polished public software
release. Add author names, institutional acknowledgements, and a project
citation before changing repository visibility. Review [Third-party software
and licensing](THIRD_PARTY.md), especially the research-only Endo-4DGS and
Gaussian-Splatting license, before redistribution.
