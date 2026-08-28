# Method 2 submission notes

## Packaged source delta

The official Endo-4DGS reference was treated as read-only. The accepted
configuration requires these completed Method-2 files from the repository:

- `train.py`: metric L1/Huber depth implementation and independently weighted
  primary depth term;
- `arguments/__init__.py`: Method-2 depth configuration fields (with all other
  experimental features disabled by default);
- `scene/imed_loader.py`: target-camera construction without target imagery;
- `imed_nvs_baseline.py`: challenge dataset runner, runtime configuration,
  rendering, and output export.

The container copies no experiment scripts, diagnostics, local results,
metrics module, result checkpoints, logs, or caches. B2-B6 code present in the
completed source is not active in the fixed submission command.

## Scientific leakage audit

- **Endoscope1 RGB:** not read. The wrapper never probes or links the mounted
  `endoscope1` tree. The loader receives only an empty synthetic target folder
  and frame-name metadata.
- **Target image statistics:** not read or computed. No target-driven color
  correction, selection, tuning, or optimization is active.
- **Target tool masks:** not read or exposed. Baseline source tool handling uses
  only `endoscope2/toolL`.
- **`metrics.py`:** not copied from the Method-2 build context and never run;
  `--no-metrics` is fixed in the wrapper's command. A copy may exist in the
  official parent filesystem, but it is outside the executed inference path.
- **Local result files:** excluded by the Dockerfile-specific strict allowlist
  and never read.
- **Previously generated checkpoints:** excluded and never read. Rendering uses
  only checkpoints newly generated for the current sequence in `/output`.
- **Training data:** Endoscope2 RGB, metric depth, and baseline source tool mask
  only.
- **Depth supervision:** metric L1 on Endoscope2 metric optical-axis depth only,
  weighted by `5e-5`.
- **Target rendering:** camera parameters come only from challenge-provided
  `K1_L` and camera-id-1 pose; target pixels are unavailable to the loader.

## Default execution

Container entrypoint:

```text
python /workspace/Endo-4DGS/docker_submission_entrypoint.py
```

Container command:

```text
--input /input --output /output
```

For each discovered sequence, the wrapper executes the equivalent of:

```text
python /workspace/Endo-4DGS/imed_nvs_baseline.py \
  --repo /workspace/Endo-4DGS \
  run-sequence \
  --sequence <internal-source-only-sequence> \
  --output /output/<sequence> \
  --iterations 1000 \
  --coarse-iterations 300 \
  --depth-loss metric_l1 \
  --depth-weight 5e-5 \
  --no-metrics
```

## Open ambiguities

1. The organizer should confirm the evaluator GPU family. The multi-architecture
   rebuild covers common Volta through Hopper targets and carries Hopper PTX,
   but the exact hidden hardware is not documented locally.
2. The organizer should confirm that `endoscope2/depthL` and
   `endoscope2/toolL` are available for all hidden sequences; the accepted
   baseline training path requires both.
3. The completed Method-2 loader requires synchronized one-to-one Endoscope-2
   source and Endoscope-1 target frame ids. The wrapper checks optional
   `target_frames.txt` metadata against the source ids and otherwise derives the
   target list from Endoscope-2 RGB. A hidden target-only subset would require
   organizer clarification and is not silently accepted.
4. The official parent must retain the extension source trees and CUDA compiler
   used by its published Dockerfile so the portable rebuild can complete.
