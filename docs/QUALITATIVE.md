# Qualitative comparison

Representative output index `00000` from
`session_004_scene_2_tool_1`. These are unmodified stored predictions; no
Endoscope1 ground truth, target mask, synthetic imagery, resizing, color
correction, or manual enhancement was added to the release assets.

<table>
  <tr>
    <th>MV1A</th>
    <th>M2 · Metric depth</th>
  </tr>
  <tr>
    <td><img src="../assets/qualitative/session_004_scene_2_tool_1__00000__mv1a.png" alt="MV1A prediction" width="480"></td>
    <td><img src="../assets/qualitative/session_004_scene_2_tool_1__00000__method2_metric_l1.png" alt="Method 2 metric-depth prediction" width="480"></td>
  </tr>
  <tr>
    <th>M3-B · Surface footprint</th>
    <th>M3-C · Confidence-gated surface</th>
  </tr>
  <tr>
    <td><img src="../assets/qualitative/session_004_scene_2_tool_1__00000__m3b_surface.png" alt="Method 3B surface-aware prediction" width="480"></td>
    <td><img src="../assets/qualitative/session_004_scene_2_tool_1__00000__m3c_confidence.png" alt="Method 3C confidence-gated prediction" width="480"></td>
  </tr>
  <tr>
    <th>M4-A · G-SHARP</th>
    <th>M4-B · Surface initialization (dropped)</th>
  </tr>
  <tr>
    <td><img src="../assets/qualitative/session_004_scene_2_tool_1__00000__m4a_gsharp.png" alt="Method 4A G-SHARP prediction" width="480"></td>
    <td><img src="../assets/qualitative/session_004_scene_2_tool_1__00000__m4b_surface_init.png" alt="Dropped Method 4B surface-initialized prediction" width="480"></td>
  </tr>
</table>

## Reading the comparison

- M1/MV1A and M3 are direct source-geometry methods. Their characteristic
  limitation is missing or stretched content where the target view has no
  source support.
- M2 and M4 optimize a per-sequence Gaussian representation. They can fill the
  complete raster, but source reconstruction quality does not guarantee target
  generalization.
- M3-B emphasizes bounded surface structure and produced the strongest local
  corrected-geometry SSIM/coverage behavior.
- M4-A was retained as the independent G-SHARP experiment.
- M4-B is shown for failure analysis: its surface initialization produced
  stronger streaking/fragmentation and lost adapted-protocol SSIM on 193 of
  199 frames.

The stored files are not all the same resolution: M1/M4 local evaluation
renders are 640×512, while M2/M3 stored challenge renders are 1280×1024. The
browser scales them to a common display width only. Quantitative results come
from the frozen evaluators, not from these displayed copies.

Before committing the PNG files, confirm that derived challenge predictions
may be redistributed. The files are Git-ignored by default and require an
explicit decision to include.
