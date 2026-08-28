# Repository and experiment map

The release is assembled from four frozen experiment roots.

| Release component | Source root | Source policy |
| --- | --- | --- |
| M1 and MV1A | `Endo-4DGS/method1_rgbd_reprojection` | Copy curated standalone geometry and submission files |
| M2 | sibling `method2_endo4dgs_plus` | Copy only the Endo-4DGS overlay, configs, reports, and Docker adapter |
| M3 | sibling `method3_surface_fusion` | Read-only source; copy curated inference files, configs, reports, and Docker adapter |
| M4 | `Endo-4DGS/method4_gsharp` | Copy adapter/trainer/configuration files; reference pinned gsplat separately |

## Method lineage

```text
Endo-4DGS challenge baseline (context only)
│
├── M2: Endo-4DGS + strict metric source-depth L1
│
└── M1: direct RGB-D reprojection
    └── MV1A: depth-aware soft splatting + radius-3 fill
        └── M3-A: exact MV1A reference
            ├── M3-B: bounded surface-aware footprint
            └── M3-C: confidence-gated M3-B

Independent G-SHARP branch
└── M4-A: upstream depth initialization + iMED scene-scale correction
    ├── G2: 100k initialization (rejected)
    └── M4-B: M3-style surface initialization (rejected)
```

M4-B borrows the source-geometry idea from M3, but not M3's target prediction.
It remains a G-SHARP model trained with the same dynamic representation as
M4-A.

## Why code is curated rather than copied wholesale

The development roots contain environments, CUDA toolkits, downloaded
installers, compiled extensions, checkpoints, thousands of renders, logs, and
evaluation-only target artifacts. Those are poor Git history and can obscure
the actual experiment. `scripts/assemble_sources.sh` uses an explicit
allowlist, keeping the release small and auditable.

Method 2 is intentionally represented as an overlay because its Docker image
starts from the official Endo-4DGS challenge image. Copying another full
Endo-4DGS tree would duplicate upstream code while making the actual four-file
scientific delta harder to see.
