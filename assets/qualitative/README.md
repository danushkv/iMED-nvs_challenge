# GeoSCOPE qualitative examples

The public README displays two small prediction-only GIFs:

| Asset | Sequence | Output indices | Display |
| --- | --- | --- | --- |
| `geoscope_scene2.gif` | `session_004_scene_2_tool_1` | 20–176, stride 6 | 360 px wide, 10 FPS |
| `geoscope_scene6.gif` | `session_004_scene_6_tool_2` | 20–176, stride 6 | 360 px wide, 10 FPS |

They contain frozen GeoSCOPE target-view predictions only. They do not contain
Endoscope1 ground truth, error maps, target masks, source frames, interpolation,
denoising, sharpening, or color correction. A JSON manifest beside each GIF
records the exact selected indices and encoding settings.

## Maintainer generation command

From the release repository root:

```bash
bash scripts/create_qualitative_gifs.sh /path/to/method8_hole_fallback/results/F1_M2_holes
```

The script uses `uv run`, refuses non-contiguous input, and removes any GIF
larger than 9.5 MB. Review both animations before staging them. Redistribution
of these prediction-only examples has been confirmed as permitted for this
public release.
