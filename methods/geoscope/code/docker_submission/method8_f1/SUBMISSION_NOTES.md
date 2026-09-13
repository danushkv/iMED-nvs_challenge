# Method 8 F1 submission notes

## Frozen scientific configuration

- M3B renderer: `surface` at metric-depth resolution (normally 640x512).
- M3B fill: exact radius-3 nearest-valid fill.
- M3B native export: existing PyTorch bilinear resize with
  `align_corners=False`, clamp, multiply by 255, round, uint8.
- M3B mask native export: nearest-neighbor resize.
- Method 2: 300 coarse iterations, 1000 fine iterations, metric L1 depth,
  depth weight `5e-5`, source-only staging, metrics off.
- Fusion: strict F1, `fallback = ~filled_valid_mask_native`.

The production invariant is checked twice in the fusion path: selected valid
bytes must equal M3B exactly, and no changed RGB pixel may overlap M3B validity.

## Input boundary

Sequence discovery prunes every directory named `endoscope1`. The wrapper
reads only `K.txt`, `pose.txt`, `endoscope2/L`, `endoscope2/depthL`, and
`endoscope2/toolL`, plus optional frame-name-only `target_frames.txt` metadata.
As in the accepted Method-2 adapter, an empty private `endoscope1` directory is
created solely so the unchanged loader can construct target camera records;
no target pixels are exposed to it.

## Deliberately excluded

- `method8_hole_fallback/hole_fallback.py` and all evaluation logic;
- target Endoscope-1 RGB, depth, masks, or statistics;
- F2 boundary distance transforms and boundary exclusions;
- alpha blending, confidence gates, tool handling, SGS/G-SHARP, temporal or
  learned completion;
- local result trees, cached predictions, metrics, checkpoints, logs, and
  development reports.

## Parent and portability caveats

The heavy parent is the official Method-2 baseline. Both CUDA extensions are
rebuilt for `7.0;7.5;8.0;8.6;8.9;9.0+PTX`, preserving native A100 `sm_80`
support. The exact M3B helper modules are copied from the existing MV1A image
in a deliberate build stage. Its `latest` tag is mutable, so inspect and pin a
registry digest before final archival if one is available.

No claim of Docker-vs-reference pixel parity or A100 runtime is made until the
manual commands in `README.md` are run and their results recorded.
