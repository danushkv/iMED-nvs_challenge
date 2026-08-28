# Method 4A — G-SHARP-iMED

Method 4A is isolated from Endo4DGS and Methods 1–3. Its Python environment,
CUDA toolkit, upstream gsplat checkout, and compiled extension all live under
`method4_gsharp/`.

Method 4B is explicitly deferred until M4-A completes one end-to-end evaluated
sequence. Its gated specification is recorded in `METHOD4B_PLAN.md`; no M4-B
implementation should be created before that gate is satisfied.

The authoritative M4-A requirements and current baselines are recorded in
`METHOD4A_SPEC.md`.

## Verified environment

Recorded on 2026-08-26:

| Component | Pinned/verified value |
| --- | --- |
| Environment manager | uv 0.11.26 |
| Environment | `method4_gsharp/envs/gsplat` |
| Python | CPython 3.11 (3.11.15 build interpreter) |
| PyTorch | 2.9.1+cu126 |
| PyTorch CUDA runtime | 12.6 |
| Local CUDA toolkit | `method4_gsharp/toolkits/cuda-12.6` |
| GPU | NVIDIA A100 80GB PCIe |
| GPU architecture | Ampere `sm_80` |
| gsplat | 1.6.0, editable local checkout |
| gsplat commit | `846c07932a77a901b474c40dd7fbfe42965ab354` |
| gsplat source | `method4_gsharp/third_party/gsplat` |

The exact commit is also stored in `GSHARP_GSPLAT_COMMIT.txt`, and the Python
package snapshot is stored in `requirements-lock.txt`.

## Shell activation

### Normal start of a new session

First connect to or request an A100 node. Then run from the Endo-4DGS
repository root:

```bash
cd /mnt/cluster/workspaces/venkateda/Endo-4DGS
source method4_gsharp/scripts/activate.sh
```

The script must be **sourced**, not run as `bash activate.sh`, because its
environment variables must remain in the current shell. It activates only
`method4_gsharp/envs/gsplat` and configures the Method 4 local CUDA toolkit.

Confirm that the expected environment and GPU are active:

```bash
python -c 'import torch, gsplat; print("torch:", torch.__version__); print("CUDA:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0)); print("gsplat:", gsplat.__file__)'
```

Expected identifying values:

```text
torch: 2.9.1+cu126
CUDA: 12.6
GPU: NVIDIA A100 80GB PCIe
gsplat: .../method4_gsharp/third_party/gsplat/gsplat/__init__.py
```

After activation, run Method 4 commands with the normal `python` and
`uv pip` commands. For example:

```bash
python method4_gsharp/third_party/gsplat/examples/dynamic_surgical_trainer.py --help
uv pip freeze > method4_gsharp/requirements-lock.txt
```

The trainer help is a wide Tyro table. To constrain it to a readable width
and page through it, use:

```bash
COLUMNS=100 python method4_gsharp/third_party/gsplat/examples/dynamic_surgical_trainer.py --help | less -R
```

Use the arrow/Page Up/Page Down keys to navigate and `q` to exit. To save a
plain-text copy instead:

```bash
NO_COLOR=1 COLUMNS=100 python method4_gsharp/third_party/gsplat/examples/dynamic_surgical_trainer.py --help > method4_gsharp/dynamic-surgical-help.txt
less method4_gsharp/dynamic-surgical-help.txt
```

When finished, leave the environment with:

```bash
deactivate
```

### Manual equivalent

If the activation helper is unavailable, the equivalent commands are:

```bash
source method4_gsharp/envs/gsplat/bin/activate

export CUDA_HOME="$PWD/method4_gsharp/toolkits/cuda-12.6"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=2
```

`BUILD_3DGS=1` and `NUM_CHANNELS='3,4'` are build-time variables. They are
needed only when rebuilding gsplat, not for normal training or rendering.

## Reproducible gsplat build

Only the standard 3DGS kernels required by the upstream dynamic surgical
trainer are compiled. The trainer uses pinhole rasterization with
`render_mode="RGB+ED"`, so compiled channel counts 3 and 4 cover RGB and
RGB-plus-depth rendering.

```bash
BUILD_3DGS=1 \
NUM_CHANNELS='3,4' \
MAX_JOBS=2 \
uv pip install \
  --no-build-isolation \
  --no-deps \
  -e method4_gsharp/third_party/gsplat \
  -v 2>&1 | tee method4_gsharp/gsplat-build-a100-3dgs-only.log
```

`--no-deps` is required here. Without it, uv may resolve a newer PyTorch/CUDA
stack and replace the pinned `torch==2.9.1+cu126` environment.

The successful build took 9m34s. Full default builds stalled for long periods
while compiling the unused 3DGUT parallel-batch rasterizer with all default
channel template combinations. `BUILD_3DGS=1` is an upstream-supported build
selection; no gsplat source was patched.

## Import verification

The verified import reported:

```text
torch: 2.9.1+cu126
CUDA: 12.6
GPU: NVIDIA A100 80GB PCIe
gsplat: /mnt/cluster/workspaces/venkateda/Endo-4DGS/method4_gsharp/third_party/gsplat/gsplat/__init__.py
```

The successful build log is `gsplat-build-a100-3dgs-only.log`.

## CUDA smoke test

After activating the Method 4 environment, run the synthetic forward/backward
test:

```bash
python method4_gsharp/scripts/smoke_test_gsplat.py
```

The test renders one synthetic Gaussian through the same four-channel
`RGB+ED` rasterization path used by the dynamic surgical trainer, then runs
backpropagation and checks that the outputs and parameter gradients are
finite. It does not read the challenge dataset or start training.

Verified on the A100 on 2026-08-26:

```text
gsplat RGB+ED forward: OK
gsplat backward:       OK
render shape:          (1, 64, 64, 4)
alpha maximum:         0.884470
torch:                 2.9.1+cu126
CUDA runtime:          12.6
GPU:                   NVIDIA A100 80GB PCIe
```

## Required geometry gate

Before creating a checkpoint or starting training, compare the independent M4
adapter against the frozen validated RGB-D geometry implementation:

```bash
python method4_gsharp/scripts/verify_geometry_convention.py \
  --data-root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --frame-index 0 \
  --samples 16 \
  --world-scale 1.0 \
  --report method4_gsharp/outputs/session_004_scene_2_tool_1/geometry_report.json
```

This command reads one legal Endoscope2 source RGB-D-mask frame plus both
camera calibration matrices. It never discovers or loads Endoscope1 imagery.
It compares selected depth-pixel world points, intrinsics, and camera-to-world
poses with Method 1's frozen geometry and checks a source pixel round trip.

The final line must be:

```text
GEOMETRY CONVENTION: PASS
```

Do not start M4-A training if it reports `FAIL`.

The gate passed on 2026-08-26 for `session_004_scene_2_tool_1`, frame ID 2,
with 16 samples: maximum world-coordinate absolute error
`7.105427357601002e-15`, zero pixel round-trip error, and maximum calibration
error `2.0698206299130106e-05`, below the `1e-4` tolerance. The report also
confirmed `target_rgb_access=false`.

## Source reconstruction gate after training

After M4-A training completes, evaluate all legal Endoscope2 source views
before attempting target-camera rendering:

```bash
python method4_gsharp/render_imed.py \
  --data-root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --checkpoint method4_gsharp/outputs/session_004_scene_2_tool_1/checkpoints/m4a_final.pt \
  --output-dir method4_gsharp/outputs \
  --device cuda \
  2>&1 | tee method4_gsharp/outputs/session_004_scene_2_tool_1/source_render.log
```

This reads Endoscope2 observations only. It saves GT/rendered source RGB,
GT/rendered depth, RGB/depth absolute-error images, exact depth arrays, and
per-frame metrics under:

```text
method4_gsharp/outputs/session_004_scene_2_tool_1/source_reconstruction/
```

The aggregate report is `source_reconstruction/source_metrics.json` and must
contain `target_rgb_access: false`. Inspect this source gate before proceeding
to legal Endoscope1-camera rendering. In particular, the first vanilla run
ended with only 2,136 Gaussians after a large late-stage prune; source metrics
and artifacts determine whether that checkpoint remains usable.

### Millimetre scene-scale correction

The first checkpoint failed the source gate with 10.4355 dB PSNR, 0.26399
SSIM, 47.22 mm depth MAE, and 52.35% depth coverage. Its Gaussian count fell
from 90,537 at step 3000 to 2,136 at completion. The cause is the upstream
surgical trainer's fixture `DynamicStrategy` scene scale of `1.0`: with iMED
coordinates preserved in millimetres, the default `prune_scale3d=0.1`
therefore pruned Gaussians larger than 0.1 mm after step 3000.

The corrected M4-A wiring derives `strategy_scene_scale` from the maximum XYZ
axis extent of the initial point cloud. For the representative sequence this
is approximately 139.11 mm. This is an input-unit integration correction
only. Initialization, architecture, HexPlane, deformation MLP, losses,
learning rates, schedule, strategy thresholds, and renderer remain unchanged.
The controlled configuration record is `configs/m4a_scale_fixed.json`.

Preserve the failed run under `outputs/`. Train the corrected run into a new
directory:

```bash
mkdir -p method4_gsharp/outputs_m4a_scale_fixed/session_004_scene_2_tool_1

python method4_gsharp/train_imed.py \
  --data-root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --output-dir method4_gsharp/outputs_m4a_scale_fixed \
  --coarse-steps 500 \
  --fine-steps 3000 \
  --init-max-points 50000 \
  --world-scale 1.0 \
  --device cuda \
  2>&1 | tee method4_gsharp/outputs_m4a_scale_fixed/session_004_scene_2_tool_1/train.log
```

The log should report approximately:

```text
[strategy_scene_scale] source=initial_xyz_max_axis_extent, value=139.111435
```

Do not use `--overwrite` on the failed run. After corrected training, inspect
its Gaussian-count trajectory before running the source gate again.

The corrected checkpoint passed its source gate across all 199 frames with
24.5943 dB PSNR, 0.84146 SSIM, 5.24 mm depth MAE, and 100% rendered-depth
coverage. Its final representation contains 83,852 Gaussians.

## Legal target-camera rendering

Only after the source gate passes, render the Endoscope1 target camera into a
fresh challenge-format directory:

```bash
mkdir -p method4_gsharp/target_output_m4a_scale_fixed

python method4_gsharp/render_imed.py \
  --view target \
  --data-root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --checkpoint method4_gsharp/outputs_m4a_scale_fixed/session_004_scene_2_tool_1/checkpoints/m4a_final.pt \
  --output-dir method4_gsharp/target_output_m4a_scale_fixed \
  --device cuda \
  2>&1 | tee method4_gsharp/outputs_m4a_scale_fixed/session_004_scene_2_tool_1/target_render.log
```

This produces exactly:

```text
method4_gsharp/target_output_m4a_scale_fixed/
└── session_004_scene_2_tool_1/
    └── renders/
        ├── 00000.png
        └── ...
```

The accompanying `target_render_summary.json` records frame count, resolution,
runtime, peak VRAM, checkpoint identity, and `target_rgb_access: false`. Target
RGB is not discovered or loaded by this rendering path. Do not evaluate until
the render count and summary have been inspected.

### Like-for-like M3 evaluation render

M3-A's recorded 21.76168497 PSNR / 0.68155512 SSIM result uses 640x512
predictions and the frozen `adapted_baseline` evaluation protocol. Keep the
native 1280x1024 challenge render above, but create a separate direct 640x512
render for the controlled local comparison:

```bash
mkdir -p method4_gsharp/target_eval_m4a_640x512

python method4_gsharp/render_imed.py \
  --view target \
  --target-size 640 512 \
  --data-root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --checkpoint method4_gsharp/outputs_m4a_scale_fixed/session_004_scene_2_tool_1/checkpoints/m4a_final.pt \
  --output-dir method4_gsharp/target_eval_m4a_640x512 \
  --device cuda \
  2>&1 | tee method4_gsharp/outputs_m4a_scale_fixed/session_004_scene_2_tool_1/target_render_640x512.log
```

In addition to challenge-format `renders/`, this local evaluation render writes
hard-linked `rgb/` entries, alpha images, alpha-derived diagnostic validity
masks, and the synchronization manifest required by the frozen Method 3
evaluator. Prediction validity never changes the PSNR/SSIM evaluation mask.

After inspecting that render, invoke Method 1's frozen evaluator directly:

```bash
python method1_rgbd_reprojection/evaluate.py \
  --data_root /mnt/cluster/datasets/iMED_NVS \
  --sequence session_004_scene_2_tool_1 \
  --prediction_dir method4_gsharp/target_eval_m4a_640x512/session_004_scene_2_tool_1 \
  --evaluation_dir method4_gsharp/outputs_m4a_scale_fixed/session_004_scene_2_tool_1/target_evaluation \
  --mask_protocol adapted_baseline \
  --device cuda
```

This is the only stage that accesses Endoscope1 RGB and target tool masks, and
it does so for evaluation only after the checkpoint and target renders are
frozen.

## Three-sequence unchanged continuation

The corrected representative run achieved 22.21097 PSNR / 0.72824 SSIM under
the same 640x512 adapted-baseline protocol as M3-A/M3-B, exceeding M3-B by
0.41113 dB and 0.04047 SSIM. The continue rule therefore selects three fixed,
additional sequences from the exact read-only Method 3 ablation subset:

```text
session_004_scene_6_tool_2
session_005_scene_7_tool_2
session_007_scene_11_tool_3
```

All three contain 199 matching legal Endoscope2 RGB, depth, and tool-mask
frames. Together with the completed `session_004_scene_2_tool_1`, these exactly
match the four sequences reported by the read-only Method 3 ablation at
`/mnt/cluster/workspaces/venkateda/method3_surface_fusion`. The selection was
made from Method 3's documented fixed subset, without inspecting Endoscope1
RGB, and is recorded in `configs/m4a_dev_sequences.json`.

Run the geometry gate and unchanged corrected M4-A training sequentially:

```bash
bash method4_gsharp/scripts/run_dev.sh
```

Optional explicit roots, if needed:

```bash
bash method4_gsharp/scripts/run_dev.sh \
  /mnt/cluster/datasets/iMED_NVS \
  /mnt/cluster/workspaces/venkateda/Endo-4DGS/method4_gsharp/outputs_m4a_scale_fixed
```

The script activates the isolated uv environment itself, refuses to replace
an existing checkpoint, and stops after all three training runs. It does not
render target views or access Endoscope1 RGB. Inspect all three geometry and
training logs before source sanity rendering.

After all three geometry and training logs pass inspection, run their
Endoscope2-only source reconstruction gates:

```bash
bash method4_gsharp/scripts/render_source_dev.sh
```

This script also activates the isolated environment itself and refuses to
replace existing source metrics. It stops after source rendering and never
discovers or loads Endoscope1 data. Inspect all three `source_metrics.json`
files before any target-camera rendering.

After the source gates pass, produce frozen 640x512 target-camera renders for
the same Method 3 comparison protocol:

```bash
bash method4_gsharp/scripts/render_target_dev.sh
```

The script writes under `method4_gsharp/target_eval_m4a_640x512/`, refuses to
replace an existing target render, and stops before offline GT evaluation. It
uses Endoscope1 intrinsics/extrinsics only and never discovers or loads
Endoscope1 RGB.

After all four frozen target render sets have been inspected, run the exact
same evaluator and both protocols used by the read-only Method 3 ablation:

```bash
bash method4_gsharp/scripts/evaluate_dev.sh
```

This evaluates the completed reference sequence plus the three additional
sequences. It calls the frozen `method1_rgbd_reprojection/evaluate.py` used by
Method 3 with `--mask_protocol both`. This offline stage accesses Endoscope1
RGB and tool masks only for metrics after training and rendering are frozen.
It does not modify Method 3 or any prediction/checkpoint.

After evaluation, reproduce the aggregate and 796-frame comparisons from saved
CSV metrics only:

```bash
python method4_gsharp/compare_m3.py
```

The comparison treats `/mnt/cluster/workspaces/venkateda/method3_surface_fusion`
as read-only and never opens challenge images. It writes JSON and Markdown
reports beneath `method4_gsharp/outputs_m4a_scale_fixed/`.

## Final M4-A ablation: G2 100k initialization

Before proceeding to M4-B, run one final controlled G-SHARP-only experiment on
the original representative sequence. G2 changes only the maximum number of
initial points from 50,000 to 100,000. The corrected millimetre scene-scale
wiring, seed, 500-step coarse stage, 3,000-step fine stage, losses, learning
rates, architecture, and dynamic strategy remain identical to M4-A. The exact
record is `configs/m4a_g2_100k.json`.

The stage runner activates the isolated uv environment and uses separate train
and render roots, so it cannot overwrite the frozen 50k M4-A result. Run each
stage only after inspecting the preceding one:

```bash
bash method4_gsharp/scripts/run_m4a_g2_one.sh train
bash method4_gsharp/scripts/run_m4a_g2_one.sh source
bash method4_gsharp/scripts/run_m4a_g2_one.sh target
bash method4_gsharp/scripts/run_m4a_g2_one.sh evaluate
```

Outputs are isolated under:

```text
method4_gsharp/outputs_m4a_g2_100k/
method4_gsharp/target_eval_m4a_g2_100k_640x512/
```

The `evaluate` stage is the only one that accesses Endoscope1 RGB, and only
after the G2 checkpoint and target renders are frozen. Regardless of the G2
result, stop M4-A ablations after this single sequence and proceed to the
controlled M4-B initialization experiment.

G2 completed and regressed target NVS despite improving source reconstruction.
The exact metrics and decision are recorded in `M4A_G2_RESULT.md`; the frozen
50k corrected configuration remains M4-A.

## Method 4B: geometry-bootstrapped G-SHARP

M4-B changes only Gaussian initialization. It derives source surfels using the
frozen M3 central-difference normal and depth-discontinuity definitions, keeps
M4-A's tissue mask, fuses source observations in 0.5 mm world-space voxels,
and initializes conservative anisotropic Gaussians with thickness ratio 0.2.
Opacity remains the unchanged M4-A value of 0.1. Confidence is used only to
combine compatible observations; confidence-weighted opacity is not part of
M4-B1.

The implementation is self-contained under `method4_gsharp/`. It inspected
the frozen Method 3 repository read-only and does not import it at runtime,
consume its target predictions, or write to it. Initialization and training
accept no Endoscope1 observation or target camera.

First run the mandatory source-only geometry and quaternion gate:

```bash
bash method4_gsharp/scripts/run_m4b_one.sh validate
```

Do not train unless the report ends with `M4-B INITIALIZATION: PASS`. After a
pass, run the controlled 50k, 500 + 3,000 step training:

```bash
bash method4_gsharp/scripts/run_m4b_one.sh train
```

After inspecting training, proceed through the same explicit gates as M4-A:

```bash
bash method4_gsharp/scripts/run_m4b_one.sh source
bash method4_gsharp/scripts/run_m4b_one.sh target
bash method4_gsharp/scripts/run_m4b_one.sh evaluate
```

Only `evaluate` accesses Endoscope1 RGB, strictly after the checkpoint and
target renders are frozen. Finally produce the saved-metrics-only controlled
comparison and per-frame win/loss counts:

```bash
python method4_gsharp/compare_m4ab.py
```

M4-B uses separate output roots and cannot overwrite M4-A:

```text
method4_gsharp/outputs_m4b_surface/
method4_gsharp/target_eval_m4b_surface_640x512/
```

The exact initialization record is `configs/m4b_surface_init.json`. Apply the
one-sequence kill rule: if geometrically valid M4-B does not improve M4-A,
stop without thickness or confidence tuning.

M4-B completed and triggered the kill rule. Under the adapted-baseline
headline protocol it regressed M4-A by 0.169560 dB PSNR and 0.019063 SSIM,
losing SSIM on 193/199 frames. The corrected-geometry protocol showed a
0.219027 dB PSNR gain but a small SSIM regression; visual inspection found
stronger elongated/radial splat artifacts and no new disocclusion recovery.
The decision is **DROP M4-B** and preserve M4-A as the independent G-SHARP
method. Do not run M4-B2 or the additional three sequences. See
`METHOD4B_RESULT.md`.

## Isolation constraints

- Do not upgrade PyTorch or CUDA in any existing Method 1–3 environment.
- Do not modify Endo4DGS or Methods 1–3.
- Do not write into the challenge dataset.
- Endoscope1 RGB is evaluation-only and must never enter training,
  initialization, tuning, pose/transform selection, or model selection.
