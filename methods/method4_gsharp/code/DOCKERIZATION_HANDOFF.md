# Method 4 Dockerization Handoff

This document is the implementation contract for the agent that packages G-SHARP-iMED for challenge inference. It covers both Method 4 variants, but the default submission candidate is **M4-A**. **M4-B is a preserved, unsuccessful ablation and must not silently replace M4-A.**

## 1. Decision and scope

| Variant | Purpose | Status | Package by default? |
| --- | --- | --- | --- |
| M4-A | Vanilla G-SHARP adapted to iMED, with the validated scene-scale correction | Submission candidate | Yes |
| M4-B | M3-style surface-aware Gaussian initialization with otherwise identical training | Experimental ablation; DROP | No; optional explicit profile only |

The Docker work must not modify:

- Endo4DGS baseline or Methods 1–3.
- The frozen M4-A or M4-B implementation, configuration, losses, schedules, or renderer math.
- Any dataset files under `/input`.

Keep new packaging code under:

```text
method4_gsharp/docker_submission/
```

Do not implement M4-B2, blend M4-A and M4-B, or run both variants and select a result from target-view quality.

## 2. Legal-data boundary

Training and initialization may use only Endoscope2 source data:

```text
/input/<sequence>/K.txt
/input/<sequence>/pose.txt
/input/<sequence>/endoscope2/L/frame_*.png
/input/<sequence>/endoscope2/depthL/frame_*.npy
/input/<sequence>/endoscope2/toolL/frame_*.png
```

Endoscope1 RGB is evaluation-only. The submitted container must never discover, enumerate, open, copy, hash, or inspect it. In particular:

- Discover sequences from direct children of `/input`, then validate only the whitelisted source/calibration paths above.
- Do not recursively walk an entire sequence directory.
- Do not include or invoke `evaluate.py`, `compare_m3.py`, official evaluation code, GT metrics, or Method 3 outputs in inference.
- Endoscope1 intrinsics and extrinsics from `K.txt` and `pose.txt` are legal target-camera inputs.
- M4-B must consume its own source-derived geometry; it has no runtime dependency on the Method 3 repository.

## 3. Validated conventions that must not regress

The iMED adapter is [datasets/imed_nvs.py](datasets/imed_nvs.py). Preserve these conventions exactly:

- Source camera: Endoscope2/L, `K2_L`, pose camera ID `0`.
- Target camera: Endoscope1/L, `K1_L`, pose camera ID `1`.
- Pose rows: `camera_id tx ty tz qx qy qz qw`.
- Poses are camera-to-world matrices.
- Depth is optical-axis Z depth in iMED millimetres.
- `world_scale = 1.0`; do not normalize depth independently by image.
- RGB native size is 1280×1024. Source training depth/image grids are 640×512, with intrinsics scaled consistently.
- On-disk tool mask value 255 means tool/excluded; value 0 means tissue/included.
- Time is the sorted synchronized frame index divided by `N - 1`.

The geometry check passed against the validated Method 1 implementation:

```text
world max absolute error: 7.105427357601002e-15
world mean absolute error: 1.834180955266144e-15
world max L2 error:        1.0048591735576161e-14
round-trip pixel error:    0.0
target RGB accessed:       false
```

Do not reinterpret poses, invert them again, convert millimetres to metres, or change mask polarity.

## 4. Reproducible software environment

The working installation is:

```text
Python:        3.11
PyTorch:       2.9.1+cu126
CUDA runtime:  12.6
CUDA toolkit:  12.6
Test GPU:      NVIDIA A100 80GB PCIe (SM 8.0)
gsplat commit: 846c07932a77a901b474c40dd7fbfe42965ab354
MAX_JOBS:      2
```

The local upstream checkout is:

```text
method4_gsharp/third_party/gsplat
```

Important build requirements:

1. Pin the exact gsplat commit above. Record it as image metadata and print it at startup.
2. Use a CUDA **development** build stage containing `nvcc` and a C++ compiler. A runtime-only image cannot compile the gsplat extension.
3. The CUDA toolkit used to compile the extension must match PyTorch's CUDA build. The earlier failure came from CUDA toolkit 12.2 compiling against a different PyTorch CUDA build. NVIDIA driver compatibility does not fix a toolkit-versus-PyTorch extension-build mismatch.
4. Install PyTorch 2.9.1 CUDA 12.6 in the isolated image; do not upgrade or depend on any host Python environment.
5. Build the pinned local gsplat checkout with `MAX_JOBS=2`. Preserve the successful no-build-isolation/no-dependency approach if required by its setup at the pinned commit.
6. Do not blindly install [requirements-lock.txt](requirements-lock.txt): it contains an absolute editable path from the development machine. Replace that line with the copied pinned checkout or a wheel built from it inside the image.
7. Pin the base-image digest and the resulting Python dependencies in the Docker deliverable.

The successful development activation script is [scripts/activate.sh](scripts/activate.sh). It is useful as a reference, but do not copy the host virtual environment or local CUDA toolkit into the final image.

### GPU architecture warning

The development build used:

```text
TORCH_CUDA_ARCH_LIST=8.0
```

That is correct for A100 but is not portable to an unknown challenge GPU. Before finalizing the image, obtain the organizer's deployment-GPU contract and explicitly compile for its compute capability. If multiple architectures are possible, build the required fat binary/PTX configuration and test it. Do not rely on automatic architecture detection during `docker build`, because a GPU may not be visible to the build stage.

In particular, an A100-only extension must not be presented as Blackwell-compatible without rebuilding and testing for the Blackwell architecture.

## 5. M4-A: default submission variant

Use:

```text
config:   method4_gsharp/configs/m4a_scale_fixed.json
trainer:  method4_gsharp/train_imed.py
renderer: method4_gsharp/render_imed.py
adapter:  method4_gsharp/datasets/imed_nvs.py
```

The controlled configuration is:

```text
initial points: 50,000
coarse steps:   500
fine steps:     3,000
seed:           42
world scale:    1.0 millimetre units
init opacity:   0.1
```

Critical correction: `DynamicStrategy.scene_scale` is derived from the initial XYZ maximum-axis extent. Do not restore the upstream hardcoded `1.0`; that correction is part of the frozen M4-A result.

Representative-sequence result (`session_004_scene_2_tool_1`):

```text
initial Gaussians: 50,000
final Gaussians:   83,852
training time:     298.2 s
peak allocated:    7.42 GB
source PSNR/SSIM:  24.5943 / 0.84146
source depth MAE:  5.2408 mm
adapted target:    22.210973 PSNR / 0.728242 SSIM
```

Across the four development sequences, M4-A averaged 18.280596 PSNR / 0.623406 SSIM. It remained an independent G-SHARP submission candidate even though it trailed M3-B in PSNR. The 100k-point G2 experiment improved source reconstruction but reduced representative target performance; it is documented in [M4A_G2_RESULT.md](M4A_G2_RESULT.md) and must not be the default.

## 6. M4-B: optional reproducibility profile only

Use:

```text
config:         method4_gsharp/configs/m4b_surface_init.json
trainer:        method4_gsharp/train_imed_m4b.py
initializer:    method4_gsharp/surface_initialization.py
renderer:       method4_gsharp/render_imed.py
result record:  method4_gsharp/METHOD4B_RESULT.md
```

M4-B changes only Gaussian initialization: it uses source RGB-D surface normals, confidence/view-normal weighting, 0.5 mm voxel deduplication, anisotropic surface scales with thickness ratio 0.2, 50k points, and opacity 0.1. Its checkpoint remains compatible with the common renderer.

Representative result:

```text
initial Gaussians: 50,000
final Gaussians:   99,278
training time:     295.05 s
peak allocated:    0.58 GB
source PSNR/SSIM:  24.1593 / 0.841608
source depth MAE:  6.566 mm
adapted target:    22.041414 PSNR / 0.709179 SSIM
delta vs M4-A:    -0.169559 PSNR / -0.019063 SSIM
SSIM frames:       6 wins / 193 losses
decision:          DROP
```

If M4-B is packaged at all, expose it only through an explicit image tag or immutable startup/build setting such as `METHOD_VARIANT=m4b`. The default must be `m4a`. Never auto-select a variant using Endoscope1 RGB, target metrics, or target appearance, and never blend their renders.

## 7. Required container behavior

The Docker agent should first inspect the official organizer template and preserve its required entrypoint/argument contract. Under the currently established challenge convention, the container should:

1. Treat `/input` as read-only and `/output` as the only persistent output location.
2. Discover valid sequences using only the whitelisted legal paths.
3. For each sequence, create a unique temporary working directory such as `/tmp/method4/<sequence>`.
4. Train the selected variant from that sequence's Endoscope2 data.
5. Render Endoscope1 using only target calibration and the trained checkpoint.
6. Process sequences sequentially so checkpoints, optimizer state, CUDA memory, and temporary renders do not accumulate.
7. Remove only the explicitly resolved per-sequence temporary directory after successful rendering if disk cleanup is needed.
8. Return a nonzero exit status on a failed sequence unless the official interface specifies another policy. Never silently skip a sequence.

Development checkpoints are sequence-specific artifacts, not universal pretrained models. Do not bake one development checkpoint into the image and apply it to arbitrary sequences unless challenge rules explicitly permit that different inference model.

Source-view diagnostics are useful during validation but should be disabled by default in challenge execution to save runtime and disk. Log the variant, input frame count, initial/final Gaussian counts, runtime, and peak VRAM to stdout rather than adding non-render files to a strict output tree.

## 8. Required challenge output

Write exactly:

```text
/output/<sequence>/renders/00000.png
/output/<sequence>/renders/00001.png
...
```

Requirements:

- Five-digit, zero-based filenames in synchronized temporal order.
- RGB PNG files.
- Native target resolution. For the inspected data this is 1280×1024.
- Do not pass the development comparison option `--target-size 640 512`; omitting `--target-size` lets the renderer use the target camera's native dimensions.
- Do not place checkpoints, metrics, source reconstructions, error maps, or logs in the renders directory.

## 9. Build context and image hygiene

Add a `.dockerignore` that excludes at least:

```text
envs/
toolkits/
downloads/
outputs/
checkpoints/
logs/
build logs
datasets or mounted challenge data
development evaluation outputs
```

Copy only the Method 4 runtime source, selected configs, and the pinned gsplat source/build artifact. Do not copy Method 3, Endoscope1 GT, cached evaluation data, the host uv environment, or the multi-gigabyte local CUDA installation.

A multi-stage image is preferred: compile gsplat in the CUDA development stage, then copy the compatible Python environment/extensions into a smaller runtime image. The runtime must retain every shared library required by PyTorch and gsplat.

## 10. Validation checklist

Do not consider the image ready until all checks below pass on a GPU matching the deployment architecture:

- [ ] Print Python, PyTorch, CUDA runtime, GPU, gsplat location/commit, architecture list, and selected Method 4 variant.
- [ ] Run [scripts/smoke_test_gsplat.py](scripts/smoke_test_gsplat.py) inside the image. Both gsplat RGB+ED forward and backward must pass; an import-only test is insufficient.
- [ ] Stage a legal-input test containing source data plus `K.txt`/`pose.txt`, without an Endoscope1 RGB directory.
- [ ] Run one sequence end to end with M4-A.
- [ ] Verify the expected frame count and contiguous names (`00000.png` through the final frame).
- [ ] Verify every render is RGB and at native target resolution.
- [ ] Verify the input mount is unchanged.
- [ ] Verify logs report `target_rgb_access: false` or an equivalent audited assertion.
- [ ] Inspect the final image filesystem to ensure it contains no GT target RGB, evaluation outputs, Method 3 results, or development checkpoints.
- [ ] Measure runtime, peak VRAM, temporary disk use, and final image size against organizer limits.
- [ ] If M4-B is offered, run it only as a separately selected reproducibility profile and verify that the default remains M4-A.

Exact output hashes may vary because GPU training kernels can be nondeterministic. Validate conventions, counts, output structure, and reasonable numerical behavior; any local evaluation must happen outside the submission container after rendering.

Observed M4-A peak allocated memory was approximately 7.2–7.5 GB on the tested sequences, so 24 GB is comfortably above observed model allocation. This is not a substitute for testing the final image, because CUDA workspaces, compilation, native-resolution rendering, and allocator reservation add overhead. Do not hardcode an A100 or 80 GB requirement.

## 11. Docker-agent deliverables

The Docker agent should return:

```text
image tag and image ID:
base image and digest:
Python version:
PyTorch version:
CUDA runtime/toolkit:
compiled CUDA architecture list:
gsplat commit:
default variant:
entrypoint contract:
tested sequence:
render count/resolution:
runtime:
peak VRAM:
temporary disk peak:
final image size:
source-only leakage audit:
```

The Docker agent may create packaging files and validate the container, but must not push an image, submit it, delete experiment results, change a frozen model/configuration, or choose M4-B over M4-A without explicit approval.
