# Method 4: G-SHARP-iMED

Method 4 adapts the upstream dynamic surgical G-SHARP trainer in gsplat to the
iMED camera, depth, masking, and scale conventions.

## Variants and decisions

| Variant | Initialization | Status |
| --- | --- | --- |
| M4-A | Upstream multi-frame depth unprojection + KNN scale | Keep as independent G-SHARP candidate |
| G2 | M4-A with 100k rather than 50k initial points | Drop |
| M4-B | M3-style normals, anisotropic surface scales, voxel fusion | Drop |

M4-B changes initialization only; it is not an image-space blend and never
consumes M3's target prediction.

## Pinned environment

```text
Python:        3.11
PyTorch:       2.9.1+cu126
CUDA:          12.6 runtime and toolkit
gsplat commit: 846c07932a77a901b474c40dd7fbfe42965ab354
verified GPU:  NVIDIA A100 80GB PCIe, sm_80
MAX_JOBS:      2
```

### uv environment

Create a new environment; do not install this stack into an existing
Endo-4DGS environment:

```bash
cd methods/method4_gsharp/code
uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.9.1 torchvision==0.24.1
uv pip install -r docker_submission/requirements.txt
```

Add gsplat as a pinned submodule or clone, then build only the kernels used by
the trainer:

```bash
git clone https://github.com/nerfstudio-project/gsplat.git third_party/gsplat
git -C third_party/gsplat checkout 846c07932a77a901b474c40dd7fbfe42965ab354

export CUDA_HOME=/path/to/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=2

BUILD_3DGS=1 NUM_CHANNELS='3,4' \
uv pip install --no-build-isolation --no-deps -e third_party/gsplat -v
```

Change `TORCH_CUDA_ARCH_LIST` for the actual deployment GPU. A build compiled
only for A100 is not automatically portable to Blackwell.

Run the forward/backward smoke test before touching the dataset:

```bash
python scripts/smoke_test_gsplat.py
```

## Frozen M4-A

```text
config:            configs/m4a_scale_fixed.json
initial points:    50,000
coarse/fine steps: 500 / 3,000
world scale:       1.0 (millimetres)
seed:              42
initial opacity:   0.1
```

The critical iMED integration fix derives `DynamicStrategy.scene_scale` from
the maximum-axis extent of the initial XYZ point cloud. Do not restore the
upstream fixture value `1.0`, which caused catastrophic late pruning in a
millimetre-scale scene.

Representative sequence:

```text
initial/final Gaussians: 50,000 / 83,852
training time:           298.2 s
peak allocated VRAM:     7.42 GB
source PSNR/SSIM:        24.5943 / 0.84146
adapted target:          22.21097 / 0.72824
```

## Reproduction order

Use the exact commands in the assembled original `code/README.md`:

```text
verify_geometry_convention.py
→ train_imed.py
→ render_imed.py source gate
→ render_imed.py --view target
→ evaluate.py outside inference
```

M4-B uses `train_imed_m4b.py` and `configs/m4b_surface_init.json`, but its
documented decision is DROP. The default Docker image must remain M4-A.

## Docker

The assembled `docker_submission/` contains separate M4-A and M4-B Dockerfiles
and entrypoints. Read `DOCKERIZATION_HANDOFF.md` before building. Do not package
the local environment, CUDA installer, checkpoints, or development outputs.
