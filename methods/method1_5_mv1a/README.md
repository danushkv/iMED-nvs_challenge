# Method 1.5: MV1A

MV1A is the confirmed Phase-11 refinement of Method 1. “Method 1.5” was only an
informal project label; **MV1A** is the release name.

## Fixed method

```text
input:                  Endoscope2/L RGB + depth
renderer:               depth-aware bilinear soft splatting
visibility tolerance:   1.0 mm + 0.01 × nearest target depth
depth softness:         8.0
source tool masking:    disabled
depth filtering:        none
hole handling:          nearest-valid propagation
fill radius:            3 pixels on the 640×512 grid
internal resolution:    640×512
export resolution:      1280×1024
temporal/stereo fusion: none
```

Historical hidden result:

```text
PSNR: 20.089
SSIM: 0.608
```

The confirmed local and pushed images shared image ID `aaa952c110d1`:

```text
method1-rgbd-reprojection:phase11-candidate
docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest
```

## Recommended reproduction: Docker

After assembling the release:

```bash
cd methods/method1_5_mv1a/code
docker build \
  -f docker_submission/phase11_candidate/Dockerfile \
  -t imed-nvs:mv1a \
  .
```

The standalone snapshot contains the selected geometry modules and submission
adapter. It expects challenge-style mounts:

```bash
docker run --rm --gpus all \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  imed-nvs:mv1a
```

Expected output is `/output/<sequence>/renders/00000.png ...` at native target
resolution.

## Relationship to Method 3

M3-A is an exact MV1A wrapper. M3-B and M3-C preserve MV1A camera geometry,
visibility, radius-3 fill, export, and naming while changing only the projected
surface footprint and then its source-only confidence gate.
