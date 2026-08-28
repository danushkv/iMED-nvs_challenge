# Method 4B — Geometry-Bootstrapped G-SHARP

## Status and hard gate

**Status: COMPLETE — DROP after the one-sequence controlled gate.**

Method 4B may begin only after Method 4A has run end-to-end on one sequence,
including source reconstruction, target rendering, and evaluation with the
same local protocol used for Method 3.

That gate is satisfied. The frozen 50k M4-A representative result is
22.210973 PSNR / 0.728242 SSIM under the adapted-baseline protocol. The final
100k G2 ablation improved source reconstruction but regressed target NVS, as
recorded in `M4A_G2_RESULT.md`; therefore M4-A is frozen at 50k.

M4-B subsequently achieved 22.041414 PSNR / 0.709179 SSIM under that same
adapted-baseline protocol, a regression of 0.169560 dB / 0.019063 SSIM. It
lost SSIM on 193/199 frames. The kill rule therefore stops M4-B; full metrics
and visual findings are recorded in `METHOD4B_RESULT.md`.

The frozen Method 3 implementation must never be modified. M3-B and M4-A
must remain unchanged baselines.

## Research question

Can the validated M3 surface-aware, source-derived RGB-D geometry provide a
better Gaussian initialization for G-SHARP than its generic multi-frame depth
initialization?

This changes the G-SHARP 3D representation. It is not target-image blending
and must not consume M3's final target prediction.

## Controlled first comparison

The initial M4-B experiment differs from M4-A in exactly one component:

```text
Gaussian initialization
```

Keep identical to M4-A:

- losses and loss weights;
- HexPlane;
- deformation MLP;
- coarse/fine steps;
- learning rates;
- DynamicStrategy and densification/pruning;
- renderer and evaluation protocol.

## Legal inputs

Initialization may use only M3's validated source-derived geometry and legal
Endoscope2 observations. Endoscope1 RGB remains evaluation-only and must not
enter initialization, training, tuning, pose/transform selection, or model
selection.

For each accepted M3 source sample, recover where available:

```text
position XYZ
RGB
surface normal
local surface footprint
depth
geometry confidence
timestamp/source frame
```

Do not consume M3's final target prediction.

## Surface-aware Gaussian initialization

Replace only M4-A's initialization chain:

```text
multi_frame_depth_unprojection
→ isotropic KNN scale
→ random quaternion
```

with:

- **Mean:** validated 3D source surface point.
- **Color:** corresponding Endoscope2 RGB.
- **Rotation:** construct an orthonormal `tangent_1, tangent_2, normal`
  basis and convert it to gsplat's quaternion convention. Add ordering and
  round-trip tests.
- **Scale:** anisotropic surface element. Use the local tangent footprint for
  `sx`/`sy`; set `sz = min(sx, sy) * thickness_ratio` conservatively.
- **Opacity:** use exactly M4-A's default opacity in the first comparison.

If a tiny thickness check is necessary, the only candidates are `0.1`, `0.2`,
and `0.3`; do not conduct a broad sweep.

## Multi-frame deduplication

Use simple world-coordinate voxel/grid deduplication. Combine compatible
observations using geometry confidence, viewing quality, and surface-normal
agreement without over-engineering the fusion.

Report:

```text
raw observations
after masking
after confidence filtering
after deduplication
final Gaussian count
```

## Dynamic training

After initialization, preserve M4-A's complete dynamic learning stack:

```text
HexPlane
DeformNetwork
DynamicStrategy
coarse/fine schedule
densification/pruning
losses
```

Initialization supplies source-observed tissue geometry; G-SHARP still learns
residual temporal deformation.

## Required first ablation

Compare only standard M4-A initialization against surface-aware M4-B
initialization on one sequence. Report:

| Metric | M4-A | M4-B | Delta |
| --- | ---: | ---: | ---: |
| PSNR | | | |
| SSIM | | | |
| Source PSNR | | | |
| Source SSIM | | | |
| Initial Gaussian count | | | |
| Final Gaussian count | | | |
| Runtime | | | |

Also report per-frame PSNR and SSIM win/loss counts using the Method 3
comparison convention.

## Visual analysis

Compare M3-B, M4-A, M4-B, and evaluation-only GT on identical frames. Focus
on missing source coverage, stretched splats, hidden tissue, strong tissue
deformation, texture blur, and SSIM-sensitive edges.

Determine whether M4-B provides faster convergence, sharper structure,
better target coverage, higher target PSNR, or higher target SSIM.

## Kill rule

Run one sequence first. If a geometrically correct M4-B initialization does
not improve M4-A, stop without extensive tuning. If it improves M4-A, run the
unchanged configuration on the same three additional M4-A sequences.

## Strictly optional follow-ups

Only if the first M4-B initialization helps:

- **M4-B2:** confidence-weighted, safely clamped opacity using M3 geometry
  confidence; retain moderately uncertain geometry.
- A weak static/dynamic prior from cross-frame geometric consistency, only if
  straightforward and M4-B is already competitive.

Neither belongs in the initial M4-B experiment.

## Final report template

```text
METHOD 4B
---------

Standard M4-A init:
  initial points:
  PSNR:
  SSIM:

Surface M4-B init:
  initial points:
  PSNR:
  SSIM:

Delta PSNR:
Delta SSIM:

Per-frame PSNR wins/losses:
Per-frame SSIM wins/losses:

Initialization differences:
Runtime difference:
Memory difference:

Visual improvement:
Visual degradation:

DECISION:
KEEP / DROP
```
