# Reproducibility guide

## 1. Assemble the curated snapshot

From the original `Endo-4DGS` workspace root:

```bash
bash imed_nvs_release/scripts/assemble_sources.sh
bash imed_nvs_release/scripts/validate_release.sh
```

Optional source-root overrides are useful if the sibling repositories move:

```bash
METHOD2_SOURCE=/path/to/method2_endo4dgs_plus \
METHOD3_SOURCE=/path/to/method3_surface_fusion \
bash imed_nvs_release/scripts/assemble_sources.sh
```

Review the resulting Git diff before committing. The script never copies
datasets, environments, compiled extensions, downloads, checkpoints, or bulk
results.

## 2. Choose one method environment

There is intentionally no root environment:

| Family | Recommended environment |
| --- | --- |
| M1 / MV1A / M3 | PyTorch 2.1.2 + CUDA 11.8, matching the confirmed MV1A image |
| M2 | Official challenge Endo-4DGS CUDA 11.8 Docker base plus the released overlay |
| M4 | CPython 3.11, PyTorch 2.9.1+cu126, CUDA toolkit/runtime 12.6, pinned gsplat |

Follow the method README rather than installing all dependencies into one
environment.

## 3. Verify geometry before expensive work

M1 provides a source-to-source identity gate. M4 provides an independent
cross-implementation backprojection test. Run the relevant gate before full
training or sequence rendering. A camera convention failure invalidates every
downstream metric.

## 4. Reproduce one representative sequence

Start with `session_004_scene_2_tool_1` and preserve a new output directory.
Do not overwrite stored development outputs. For M4, follow this order:

```text
geometry verification
→ training
→ Endoscope2 source reconstruction
→ legal target-camera rendering
→ offline target evaluation
```

## 5. Scale to the fixed sequence set

Only after one sequence passes its method-specific gates should the unchanged
configuration run on the remaining fixed sequences. Do not tune per sequence.

## 6. Record provenance

For each run save:

```text
method and variant
configuration path and hash
source revision / upstream commit
Python, PyTorch, CUDA, GPU
dataset sequence and synchronized frame count
random seed
command line
start/end time
peak VRAM
metric protocol
target_rgb_access audit flag
```

The result CSVs in `experiments/` are compact reference records, not a
replacement for raw logs and per-frame metrics in archival storage.
