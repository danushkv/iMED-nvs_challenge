# Method 1 Phase-11 RGB-D reprojection Docker candidate

This isolated candidate adapter implements the iMED-NVS Task-2
`/input` to `/output/<sequence>/renders/NNNNN.png` contract. It packages only
the lightweight Method-1 inference path and does not include Endo-4DGS,
evaluation code, metrics, experiment outputs, or target-view data loaders.

## Exact packaged configuration

- renderer: `soft_depth` depth-aware bilinear splatting;
- visibility tolerance: `1.0 mm + 0.01 * nearest_depth`;
- depth softness: `8.0`;
- source tool masking: disabled unconditionally; `endoscope2/toolL` is neither
  required nor discovered;
- hole handling: H2 nearest-valid propagation from raw source support;
- fill radius: 3 pixels on the internal metric-depth grid;
- depth filtering: none;
- internal geometry resolution: source metric-depth resolution (`640x512` on
  the inspected public sequence);
- export: bilinear resize of the filled RGB prediction to native source RGB
  size (`1280x1024` on the inspected public sequence);
- output names: zero-based, sequential, five-digit, three-channel RGB PNGs.

Nearest propagation is prediction-only, limited to radius 3, and leaves pixels
outside raw support plus that bounded fill black. Runtime inference reads only
`endoscope2/L/frame_*.png`, `endoscope2/depthL/frame_*.npy`, `K.txt`, and
`pose.txt`.

The configuration is the one-sequence Phase-11 preliminary winner from
`session_004_scene_2_tool_1`; it is not a cross-sequence-confirmed claim.

## Build

Build from the `method1_rgbd_reprojection` root:

```bash
docker build \
  --build-arg HTTP_PROXY \
  --build-arg HTTPS_PROXY \
  --build-arg NO_PROXY \
  --build-arg http_proxy \
  --build-arg https_proxy \
  --build-arg no_proxy \
  -f docker_submission/phase11_candidate/Dockerfile \
  -t method1-rgbd-reprojection:phase11-candidate \
  .
```

The image is based on `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime` and adds
only `numpy==1.24.4` and `Pillow==10.2.0`. It copies `hole_filling.py`; it does
not copy `depth_filtering.py`.

## One-sequence local contract test

Mount a temporary input parent containing only the permitted source inputs so
the sequence name is preserved beneath `/input`:

```bash
bash docker_submission/phase11_candidate/scripts/local_test.sh \
  method1-rgbd-reprojection:phase11-candidate \
  /tmp/<temporary-input-parent> \
  /tmp/<fresh-output-parent>
```

The checker validates count, exact names, PNG format, RGB mode, and native
source-frame resolution without opening Endoscope-1 data.
