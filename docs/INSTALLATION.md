# Installation

## Requirements

- Linux x86-64
- Git and [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU and driver for GPU rendering
- Docker with NVIDIA Container Toolkit for GeoSCOPE and Method 2
- Authorized iMED dataset access

## Training-free geometry environment

From the repository root:

```bash
uv sync
source .venv/bin/activate
python -c "import numpy, PIL, scipy, torch; print(torch.__version__, torch.version.cuda)"
```

The project pins Python 3.10, PyTorch 2.1.2+cu118-compatible wheels, NumPy,
Pillow, and SciPy. This environment supports M1, MV1A/M3 research code,
offline analysis, and GIF generation.

If the machine has no compatible NVIDIA GPU, use CPU explicitly for supported
diagnostics. GeoSCOPE itself requires a CUDA GPU because its Method-2 branch
uses compiled Endo-4DGS rasterizers.

## GeoSCOPE

The supported full environment is the submitted Docker context:

```bash
cd methods/geoscope/code/docker_submission/method8_f1
bash scripts/build.sh geoscope:latest
bash scripts/inspect_image.sh geoscope:latest
```

The Docker build needs access to:

```text
docker.synapse.org/syn74277461/imed-nvs-baseline:v1
docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest
```

The second tag is mutable in the historical record. Inspect its digest before
archival builds and replace it with a pinned digest when one is available.

## Method 2

Method 2 shares GeoSCOPE's Endo-4DGS branch. Build it using the instructions in
[`methods/method2_metric_depth/README.md`](../methods/method2_metric_depth/README.md).
The image compiles CUDA 11.8 extensions for several GPU architectures.

## Method 4

Method 4 requires Python 3.11, PyTorch 2.9.1+cu126, a CUDA 12.6 toolkit, and a
pinned gsplat commit. It must use its own uv environment. See
[`methods/method4_gsharp/README.md`](../methods/method4_gsharp/README.md).

## Why there is no single environment

The training-free geometry path and the two learned Gaussian implementations
were validated against different PyTorch/CUDA toolchains. Combining all of
them into one host environment would be less reproducible and risks loading
incompatible compiled extensions. The root uv environment is intentionally
small; Docker isolates the challenge runtime.
