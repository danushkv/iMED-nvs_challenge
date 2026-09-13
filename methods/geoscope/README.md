# GeoSCOPE (Method 8)

**GeoSCOPE — Geometry-first Selective Completion Of Projection Errors** is the
final iMED NVS submission. Its descriptive title is **Geometry-First Novel
View Synthesis with Selective Learned Completion**.

GeoSCOPE combines two frozen predictors:

- **M3-B:** training-free, surface-aware RGB-D reprojection;
- **Method 2:** challenge-adapted Endo-4DGS with metric-depth supervision.

The learned prediction is used only where M3-B remains unsupported after its
existing radius-3 fill:

```python
valid = nearest_resize(m3b_filled_valid_mask)
final = m3b_rgb.copy()
final[~valid] = method2_rgb[~valid]
assert byte_equal(final[valid], m3b_rgb[valid])
```

No black-pixel heuristic, alpha blending, confidence selector, target tool
mask, or target RGB is used at inference.

## Result

| Variant | Mean PSNR | Delta PSNR | Mean SSIM | Delta SSIM | Protected pixels changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| **F1 strict holes** | **19.617677** | **+0.504134** | **0.526226** | **+0.022518** | **0** |
| F2 exclude 1 px | 19.508873 | +0.395330 | 0.513824 | +0.010117 | 0 |
| F2 exclude 2 px | 19.437339 | +0.323795 | 0.512169 | +0.008459 | 0 |

F1 won mean PSNR on all four development sequences. F2 was rejected because
excluding fallback near valid geometry removed useful completions. A hidden
GeoSCOPE score is not recorded in this snapshot.

## Requirements

- Linux x86-64
- NVIDIA GPU with Docker/NVIDIA Container Toolkit
- access to the two historical Synapse parent images
- iMED sequence data mounted read-only

The full method uses Docker because Method 2 depends on compiled CUDA
rasterizers. The root uv environment is sufficient for host-side static and
fusion tests, but not for the complete learned renderer.

## Build

```bash
cd methods/geoscope/code/docker_submission/method8_f1
uv run python tests/static_checks.py
uv run python tests/test_f1_invariant.py
bash scripts/build.sh geoscope:latest
bash scripts/inspect_image.sh geoscope:latest
```

The build is multi-stage. It takes the exact MV1A utilities used by M3-B and
the official challenge Endo-4DGS runtime, then rebuilds CUDA extensions with
native A100 `sm_80` support and additional architectures.

## One-sequence smoke test

Stage only legal source-side inputs:

```bash
bash scripts/stage_one_sequence.sh /path/to/iMED_NVS session_004_scene_2_tool_1
```

Copy the printed temporary path and run:

```bash
TEST_ROOT=/tmp/method8-f1-nvs-test.REPLACE_ME
bash scripts/local_test.sh geoscope:latest "$TEST_ROOT"
```

The test mounts `/input` read-only, disables networking, enforces the
three-hour challenge limit, and validates the PNG contract. The exact
pixel-parity and M3-B-valid invariance commands are in
[`code/docker_submission/method8_f1/README.md`](code/docker_submission/method8_f1/README.md).

## Direct inference

```bash
docker run --rm --gpus all --network=none \
  -v /path/to/imed_nvs:/input:ro \
  -v /path/to/output:/output \
  geoscope:latest
```

The input may be one sequence or a directory containing multiple sequences.
The output contract is:

```text
/output/<sequence>/renders/00000.png
/output/<sequence>/renders/00001.png
...
```

## Checkpoints

No pretrained checkpoint is required. M3-B is training-free; Method 2 trains
a scene model from legal Endoscope2 observations for each input sequence. No
pretrained checkpoint is distributed because these Method-2 states are
sequence-specific rather than universal models.

## Developing GeoSCOPE

The submitted runtime is under `code/docker_submission/method8_f1/`:

```text
combined_entrypoint.py   orchestration and source-only staging
f1_fusion.py             strict masked replacement invariant
m3b/                     frozen surface renderer
method2/                 frozen Endo-4DGS overlay
tests/                   static, parity, and invariant checks
```

The offline experiment code under `code/` accepts pre-rendered M3-B and Method
2 directories and may access target RGB only to evaluate a frozen candidate.
It must never be imported into the inference container.

When proposing a new fallback:

1. keep F1 as an immutable reference;
2. derive selection from legal source observations only;
3. assert exact equality on every protected M3-B pixel;
4. report hole-only quality before global metrics;
5. use a fixed development set and never tune per sequence.

## Source-only guarantee

Sequence discovery prunes every `endoscope1` directory. Runtime reads only
`K.txt`, `pose.txt`, `endoscope2/L`, `endoscope2/depthL`,
`endoscope2/toolL`, and optional frame-name metadata. Method 2 constructs
target camera records from calibration without exposing target pixels.

See [`code/METHOD8_FINAL_REPORT.md`](code/METHOD8_FINAL_REPORT.md) and
[`../../docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).
