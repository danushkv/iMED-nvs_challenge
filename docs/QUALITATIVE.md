# Planned qualitative GIF comparison

Qualitative assets are intentionally deferred. The earlier static PNG gallery
was removed; synchronized GIFs will communicate temporal stability,
deformation, flicker, disocclusion behavior, and splat streaking more clearly.

## Proposed gallery

Use the same sequence, output-index range, crop, output resolution, and frame
rate for every method:

| GIF | Role |
| --- | --- |
| `mv1a.gif` | Training-free reprojection reference |
| `method2_metric_l1.gif` | Endo-4DGS metric-depth result |
| `m3b_surface.gif` | Hidden-PSNR winner |
| `m3c_confidence.gif` | Confidence-gated M3 comparison |
| `m4a_gsharp.gif` | Frozen vanilla G-SHARP adaptation |
| `m4b_surface_init.gif` | Controlled initialization ablation and failure analysis |

Recommended first sequence:

```text
session_004_scene_2_tool_1
```

## Fair-comparison rules

- Use predictions that were already frozen before hidden validation.
- Use identical temporal indices and FPS.
- Normalize display resolution without changing aspect ratio or content.
- Do not apply denoising, sharpening, interpolation, color correction, or
  method-specific crops.
- Label M4-B as a controlled ablation, not a blended method.
- Keep Endoscope1 GT out of the repository unless redistribution is explicitly
  permitted. If GT is later included, identify it as evaluation-only.
- Record the source render directory, frame range, resize operation, and GIF
  command in a small manifest beside the assets.

The release `.gitignore` excludes qualitative GIFs by default, making their
eventual inclusion an explicit decision.
