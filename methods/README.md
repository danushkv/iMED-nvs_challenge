# Method development map

Start with the smallest method that answers your research question. The
methods deliberately use separate environments where their CUDA stacks differ.

| Method | What to change | Entry point | Environment |
| --- | --- | --- | --- |
| [GeoSCOPE](geoscope/README.md) | Support masks, fallback policies, final challenge runtime | `docker_submission/method8_f1/combined_entrypoint.py` | Docker |
| [M1](method1_rgbd_reprojection/README.md) | Camera conventions, projection, z-buffering, soft splats | `render_sequence.py` | Root uv |
| [MV1A](method1_5_mv1a/README.md) | Fast depth-aware splatting and bounded fill | submission `nvs_method.py` | Root uv / Docker |
| [M2](method2_metric_depth/README.md) | Endo-4DGS losses, masks, appearance, deformation | `imed_nvs_baseline.py` | CUDA 11.8 Docker |
| [M3](method3_surface_fusion/README.md) | Normals, anisotropic footprints, surface confidence | `render_sequence.py` | Root uv |
| [M4](method4_gsharp/README.md) | Dynamic Gaussian surfels and initialization | `train_imed.py` | Separate uv + CUDA 12.6 |

## Recommended progression

1. Run M1's identity test to verify camera geometry.
2. Render one frame with MV1A or M3-B.
3. Render one complete sequence without changing parameters.
4. Evaluate offline, after prediction files are frozen.
5. Move to M2/M4 only when the experiment genuinely requires optimization.
6. Use GeoSCOPE when reproducing the final submitted system.

Do not install M4's PyTorch/CUDA stack into the root geometry environment, and
do not build Endo-4DGS extensions against a CUDA toolkit version different
from the PyTorch CUDA version.
