# Endo-4DGS for iMED Static Two-Camera Evaluation

This repository supports the iMED 2026 challenge subtask on deformable novel view synthesis, part of EndoVis 2026 at MICCAI 2026 (Strasbourg, France).

[[Challenge Website](https://imed-challenge.github.io/)] [[Participate](https://www.synapse.org/Synapse:syn74277461/wiki/639538)] [[Parent Challenge Hub](https://opencas.dkfz.de/endovis/challenges/2026/)]

<p align="center">
  <img src="assets/comparison.gif" width="900" alt="Ground truth (left), trained endoscope render (middle), and evaluated endoscope render (right) on session_004_scene_2_tool_1"/>
</p>

## Huge Thanks to the Original Repository

This project is a fork of the original Endo-4DGS repository by Huang et al.

- Original codebase: [lastbasket/Endo-4DGS](https://github.com/lastbasket/Endo-4DGS)
- This fork: [smbonilla/Endo-4DGS](https://github.com/smbonilla/Endo-4DGS)

Please cite the original Endo-4DGS work:

```bibtex
@inproceedings{huang2024endo,
  title={Endo-4dgs: Endoscopic monocular scene reconstruction with 4d gaussian splatting},
  author={Huang, Yiming and Cui, Beilei and Bai, Long and Guo, Ziqi and Xu, Mengya and Islam, Mobarakol and Ren, Hongliang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={197--207},
  year={2024},
  organization={Springer}
}
```

## What This Fork Adds

This fork adds iMED dataset support for a static two-camera protocol:

- Training view: `endoscope2/L`
- Test view: `endoscope1/L`
- Depths: `depthL` (mm, used directly)
- Masks: `toolL`, loaded and converted as `mask = 1 - raw_mask/255`
- Poses: `pose.txt` with exactly 2 rows (`k tx ty tz qx qy qz qw`)
- Intrinsics: `K.txt`

Expected dataset layout:

```text
./data/imed/session_004_scene_2_tool_1
├── K.txt
├── pose.txt
├── endoscope1
│   ├── L
│   ├── depthL
│   └── toolL
└── endoscope2
    ├── L
    ├── depthL
    └── toolL
```

## Method: Train on One Camera, Test on the Other

For each static iMED session:

1. Train Gaussian splats only on `endoscope2` images.
2. Keep both cameras static and use the provided two-camera pose relation.
3. Render novel views from the held-out `endoscope1` camera.
4. Evaluate rendered test images against `endoscope1` ground truth.

This setup measures cross-camera generalization rather than interpolation within one camera stream.

## Metrics and Timing

- **PSNR / SSIM:** computed in `metrics.py`.
- **LPIPS:** also computed in `metrics.py`.
- **Inference speed:** printed as `FPS` in `render.py` during rendering.

For iMED two-camera evaluation, PSNR/SSIM are computed on the valid reprojection region from `endoscope2` into `endoscope1` (single global mask per sequence).

## Setup

```bash
git clone https://github.com/smbonilla/Endo-4DGS.git
cd Endo-4DGS
git submodule update --init --recursive
conda create -n ED4DGS python=3.8
conda activate ED4DGS
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization-depth
pip install -e submodules/simple-knn
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchmetrics
```

## Run iMED Training

```bash
sh train_imed.sh
```

## Run Rendering + Evaluation

```bash
python render.py --model_path <OUTPUT_PATH> --pc --skip_video --skip_train --configs arguments/imed.py
python metrics.py --model_path <OUTPUT_PATH>
```

## Docker-Compatible Branch

This branch packages the iMED NVS baseline and a participant-facing
Docker-compatible submission scaffold.

- The repository root builds the Endo-4DGS baseline image.
- `imednvs_submission/` is a CUDA/3DGS-ready Docker submission scaffold that
  participants can copy and replace with their own per-sequence optimization
  and rendering method.
- Challenge data are not included in either Docker image. Mount iMED NVS data
  at runtime as `/input:ro` and write predictions or model outputs to `/output`.

The published baseline image is:

```bash
docker pull docker.synapse.org/syn74277461/imed-nvs-baseline:v1
```

### Build Baseline Image

```bash
docker build -t imed-nvs-baseline:dev .
```

### Run One Sequence

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/iMED_NVS/session_004_scene_2_tool_1:/input:ro \
  -v /path/to/outputs:/output \
  imed-nvs-baseline:dev \
  run-sequence \
  --sequence /input \
  --output /output/session_004_scene_2_tool_1
```

### Run All Detected Sequences

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/iMED_NVS:/input:ro \
  -v /path/to/outputs:/output \
  imed-nvs-baseline:dev \
  run-dataset \
  --data-root /input \
  --output-root /output
```

The root Docker entrypoint is `imed_nvs_baseline.py`. It trains on
`endoscope2`, renders held-out `endoscope1`, and uses a writable
`_input_sequence` view inside the output directory so `/input` can remain
read-only.

## Method 2 B1: Selectable Depth Supervision

This separate `method2_endo4dgs_plus` workspace preserves the challenge B0
training path and adds three independently selectable primary-depth modes:

```text
normalized      exact B0 independently max-normalized L1
metric_l1       strict valid-tissue metric L1 in millimetres
metric_huber    strict valid-tissue Smooth-L1 in millimetres
```

The challenge-compatible entrypoint accepts:

```text
--depth-loss normalized|metric_l1|metric_huber
--depth-weight FLOAT
--depth-huber-beta FLOAT
--depth-diagnostics-interval INT
```

Defaults (`normalized`, weight `0.01`, diagnostics disabled) preserve B0.
Metric experiments should enable diagnostics and use a much smaller weight
because their unweighted loss is expressed in millimetres. Diagnostics are
written to `depth_diagnostics.csv` and `depth_diagnostics/` inside the model
output and use Endoscope 2 training data only.

`--depth-weight` controls only the primary supervision term. The baseline's
separate `depth_weight=0.01` remains fixed for gradient and TV regularization,
so the B1 sweep does not silently change auxiliary objectives.

For local development, use the validated `imed-nvs-baseline:sm86` image and
build `docker/Dockerfile.method2-sm86`. This reuses the CUDA 11.8 extensions
rebuilt for RTX A5000 while copying the Method 2 Python source into a small
derived image. A derived image is required on hosts where the Docker daemon
cannot bind-mount the cluster workspace.

```bash
docker build --pull=false \
  -f docker/Dockerfile.method2-sm86 \
  -t method2-endo4dgs-plus:dev \
  .
```

### Paired multi-GPU ablations

`scripts/run_ablation.sh` runs one complete 300-coarse/1000-fine experiment
for one sequence on one physical GPU. The runner stages only that sequence in
a unique `/tmp` directory because this host's Docker daemon cannot bind-mount
the cluster filesystem. It then preserves the checkpoint, renders, metrics,
diagnostics, configuration, manifest, and log under `results/`. It does not
copy the staged `_input_sequence` back into the repository.

Successful temporary jobs are removed through a network-disabled Docker
cleanup container, which handles the root-owned files produced by training.
Failed jobs are retained and their exact temporary path is printed so that
they can be inspected before manual cleanup. Existing result directories are
never overwritten.

```bash
bash scripts/run_ablation.sh EXPERIMENT SEQUENCE GPU_ID
```

Currently implemented experiment names are:

```text
B0
B1_metric_huber_w1e-5
B1_metric_huber_w1e-4
B1_metric_huber_w1e-3
B1_metric_l1_w5e-6
B1_metric_l1_w5e-5
B1_metric_l1_w5e-4
B2_tool_validmean_d0
B2_tool_dilate3
B2_tool_dilate5
B3_appearance_lr1e-3_reg1e-2
```

The sequence development/holdout roles and fixed physical-GPU assignments
are recorded in `configs/method2_sequence_split.txt`. Keeping each sequence on
the same GPU for B0 and B1 makes the comparisons paired. Two workers can run
concurrently while each worker processes its assigned sequences serially:

```bash
(
  set -e
  bash scripts/run_ablation.sh B1_metric_huber_w1e-3 session_004_scene_2_tool_1 0
  bash scripts/run_ablation.sh B1_metric_huber_w1e-3 session_005_scene_7_tool_2 0
) &

(
  set -e
  bash scripts/run_ablation.sh B1_metric_huber_w1e-3 session_004_scene_6_tool_2 1
  bash scripts/run_ablation.sh B1_metric_huber_w1e-3 session_007_scene_11_tool_3 1
) &

wait
```

Set `METHOD2_KEEP_TMP=1` only when a successful job's temporary files are
needed for debugging. The default cleans them after persistent preservation.

## Method 2 B2: Tool-mask formulation

The challenge B0 implementation is already source-tool-aware: it removes tool
pixels from its initial point cloud and masks RGB, depth, normal, and confidence
losses. B2 therefore does not add a misleading mask on/off switch. It tests two
specific weaknesses of the existing mask formulation:

```text
--tool-aware-loss
    Average primary RGB and normalized-depth errors strictly over valid tissue
    elements. B0 instead averages zero-masked errors over the complete tensor.

--tool-mask-dilation 0|3|5
    Dilate the excluded source-tool region by this radius at the 640x512
    internal training resolution. Radius 3 uses a 7x7 footprint and radius 5
    uses an 11x11 footprint.
```

Both controls default to disabled/zero, preserving B0. Dilation affects only
source training masks; it never reads or modifies the Endoscope 1 evaluation
mask. At the first sampled frame of each training stage, B2 writes the base
tissue mask, effective tissue mask, and newly excluded boundary under
`tool_mask_diagnostics/`.

The initial independent ablations remain on B0 normalized depth:

```text
B2_tool_validmean_d0  strict valid averaging, no dilation
B2_tool_dilate3       B0 averaging, 3-pixel dilation
B2_tool_dilate5       B0 averaging, 5-pixel dilation
```

They must be evaluated independently before any B1+B2 combination is tested.

## Method 2 B3: Training-only appearance correction

B3 optionally learns one affine RGB correction for each Endoscope 2 training
timestamp:

```text
I_corrected = scale[t] * I_rendered + bias[t]
```

Each three-channel scale starts at one and each bias starts at zero. The source
reconstruction objective includes an identity regularizer:

```text
L_app = appearance_reg_weight *
        (mean((scale[t] - 1)^2) + mean(bias[t]^2))
```

The parameters use a separate optimizer and are saved only as training
diagnostics. `render.py` is deliberately unchanged and renders the raw Gaussian
representation for Endoscope 1. Therefore inference cannot require target RGB,
target histograms, or target-specific appearance optimization.

The challenge entrypoint exposes:

```text
--appearance-correction
--appearance-lr FLOAT
--appearance-reg-weight FLOAT
--appearance-diagnostics-interval INT
```

Defaults disable B3 and preserve B0. The first independent experiment is
`B3_appearance_lr1e-3_reg1e-2`, based on B0 depth and baseline tool handling.
It writes `appearance_diagnostics.csv`, `appearance_parameters.csv`, and
`appearance_correction_training_only.pth` to the result directory.

## Method 2 B4-B6 overnight ablations

The next fixed development-set experiments are:

```text
B3_appearance_lr1e-3_reg1e-2
    B0 + training-only affine source appearance correction

B4_source_dssim_w1e-2
    B0 + 0.01 * (1 - source SSIM)

B5_no_confidence
    B0 with both confidence losses disabled

B6_combined_b1_b3_b4
    accepted B1 metric L1 (5e-5)
    + B3 appearance correction
    + B4 source DSSIM (0.01)
    + baseline tool handling and confidence losses
```

B4 uses only Endoscope 2 reconstruction. It exercises the baseline's existing
dormant DSSIM implementation, which evaluates source images after their tool
regions have been zeroed. It is not presented as the official masked target
SSIM implementation. B5 is motivated by the observed negative confidence terms
in B0's total loss and tests their removal without changing any other term.

### Two-GPU Slurm execution

Build the current Docker image before submission. The Slurm script does not
build or pull images on a compute node.

```bash
docker build --pull=false \
  -f docker/Dockerfile.method2-sm86 \
  -t method2-endo4dgs-plus:dev \
  .
```

Submit the fixed four-sequence B3-B6 development suite with:

```bash
sbatch scripts/run_overnight_b3_b6.sbatch
```

The job requests one node with two GPUs. Each GPU processes its fixed pair of
sequences serially while the workers run concurrently. Existing completed
results are skipped, individual failures do not prevent later experiments from
running, and failed temporary directories are retained for inspection. Change
partition/account at submission time if required by the cluster, for example:

```bash
sbatch --partition=GPU_PARTITION --account=PROJECT \
  scripts/run_overnight_b3_b6.sbatch
```

After both workers finish, `scripts/summarize_ablation.py` writes:

```text
results/ablation_results.csv
results/ablation_summary.csv
results/ablation_summary.json
results/slurm_logs/<job_id>/
```

The ranking uses mean PSNR as primary and mean SSIM as secondary. The holdout
sequence is not included in this overnight selection run.

### Submission Scaffold

To build and test the participant submission scaffold:

```bash
cd imednvs_submission
docker build -t my-nvs-submission:dev .
./scripts/local_test.sh my-nvs-submission:dev /path/to/iMED_NVS ./my_test_output
```

The template follows the NVS submission contract directly: the container reads
mounted sequence data from `/input` and writes rendered target-view RGB PNGs to
`/output/<sequence_name>/renders/`.

Unlike the iMED pose-estimation task, many NVS methods, especially 3DGS-style
methods, perform per-sequence optimization on the hidden test sequence before
rendering. The submitted Docker image should therefore include all code,
compiled CUDA extensions, pretrained weights, and other assets needed for
training/optimization and rendering at runtime.
