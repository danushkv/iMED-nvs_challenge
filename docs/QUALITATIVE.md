# Qualitative results

The public landing page prioritizes two compact GeoSCOPE reconstruction GIFs:

```text
assets/qualitative/geoscope_scene2.gif
assets/qualitative/geoscope_scene6.gif
```

Both are generated from frozen F1 predictions. They contain no target ground
truth and receive no method-specific image enhancement. See the asset README
and adjacent JSON manifests for sequence, frame, resolution, and FPS details.
Redistribution of these prediction-only examples has been confirmed as
permitted for this public release.

## Why prediction-only examples

Challenge data and Endoscope1 ground truth are governed by organizer terms and
are not redistributed here. Prediction-only animation still reveals temporal
stability, deformation, holes, boundary behavior, and splat artifacts without
publishing target frames.

## Reproduce the gallery

From the repository root, pass the existing frozen F1 result root:

```bash
bash scripts/create_qualitative_gifs.sh /path/to/method8_hole_fallback/results/F1_M2_holes
```

The generator verifies a contiguous five-digit stream, uses the same frame
range for both examples, preserves aspect ratio, and enforces a 9.5 MB limit.

## Extended method comparison

The experiment lineage graphic retains M1, MV1A, M2, M3, and M4. An optional
future comparison gallery should use identical sequences, indices, resolution,
and FPS for every method. Do not add GT unless redistribution permission is
explicitly documented, and never present a rejected ablation as the final
method.
