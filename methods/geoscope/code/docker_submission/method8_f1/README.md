# Method 8 F1 challenge Docker

This isolated context packages the frozen candidate from
`method8_hole_fallback/DOCKERIZATION_HANDOFF.md`:

```text
native_fallback = NOT nearest_resize(M3B.radius3_filled_valid_mask)
final = M3B
final[native_fallback] = Method2[native_fallback]
```

It uses M3B `surface` (not `surface_confidence`) and the accepted Method 2
configuration (`300` coarse, `1000` fine, `metric_l1`, depth weight `5e-5`,
metrics disabled). No blending, boundary exclusion, confidence gate, target
image access, evaluation gate, saved prediction, or released checkpoint is in
the runtime.

## Files packaged in the image

- `combined_entrypoint.py` and `f1_fusion.py`;
- the four hashed accepted Method-2 overrides under `method2/`;
- the three hashed M3B production modules and frozen M3B JSON under `m3b/`;
- five exact MV1A helper modules copied during the multi-stage build from the
  same parent used by the working M3B image.

Method-2 checkpoints, private M3B renders, and private support masks are made
under a unique `/tmp/method8-f1-*` directory. They are deleted after each
sequence is published and also covered by automatic temporary-directory
cleanup on failure. `/output` receives only final RGB PNG renders.

## Host-only checks (no Docker)

Run these from the repository root:

```bash
cd methods/geoscope/code/docker_submission/method8_f1

uv run python tests/static_checks.py
uv run python tests/test_f1_invariant.py
```

## Build and inspect

Docker commands are manual. Select a single allocated GPU first if needed:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu \
  --format=csv,noheader

export DOCKER_GPUS='device=0'
```

Inspect the two parent images. The known M3B utility parent had local image-ID
prefix `aaa952c110d1`; stop if the mutable `latest` tag no longer identifies
the expected image:

```bash
docker image inspect \
  docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest \
  --format 'id={{.Id}} digests={{json .RepoDigests}}'

docker image inspect \
  docker.synapse.org/syn74277461/imed-nvs-baseline:v1 \
  --format 'id={{.Id}} digests={{json .RepoDigests}}'
```

Build with the same proxy build arguments used by Method 2:

```bash
bash scripts/build.sh imed-method8-f1:dev
```

Inspect entrypoint, labels, source manifest, and CUDA architecture list:

```bash
bash scripts/inspect_image.sh \
  imed-method8-f1:dev
```

Offline CUDA smoke test:

```bash
docker run --rm \
  --gpus "$DOCKER_GPUS" \
  --network=none \
  --entrypoint python \
  imed-method8-f1:dev \
  -c "import sys,torch,diff_gaussian_rasterization_depth; \
from simple_knn import _C as simple_knn_cuda; \
sys.path.insert(0,'/app/method8_m3b'); \
from surface_splat import render_surface_splat; \
print('Torch:',torch.__version__); \
print('CUDA:',torch.version.cuda); \
print('GPU:',torch.cuda.get_device_name(0)); \
print('capability:',torch.cuda.get_device_capability(0)); \
print('M3B and Method2 CUDA imports: OK')"
```

## One-sequence read-only test

Stage a public sequence without copying any Endoscope-1 directory:

```bash
bash scripts/stage_one_sequence.sh \
  /path/to/iMED_NVS \
  session_004_scene_2_tool_1
```

Copy the printed `TEST_ROOT=...` assignment, then run with networking disabled
and `/input` read-only. The script enforces the three-hour challenge timeout:

```bash
TEST_ROOT=/tmp/method8-f1-nvs-test.REPLACE_ME

bash scripts/local_test.sh \
  imed-method8-f1:dev \
  "$TEST_ROOT"
```

Exact decoded-byte comparison with the saved frozen F1 result:

```bash
SEQUENCE=session_004_scene_2_tool_1

uv run python tests/compare_f1_pixels.py \
  --prediction "$TEST_ROOT/output/$SEQUENCE/renders" \
  --reference \
    "/path/to/method8_hole_fallback/results/F1_M2_holes/$SEQUENCE/renders"
```

Independent zero-change check against the original M3B render and its true
radius-3 support mask:

```bash
uv run python tests/check_m3b_valid_pixels.py \
  --prediction "$TEST_ROOT/output/$SEQUENCE/renders" \
  --m3b-renders "/path/to/M3_B_surface/$SEQUENCE/renders" \
  --m3b-filled-mask "/path/to/M3_B_surface/$SEQUENCE/filled_valid_mask"
```

Because the staged tree contains no `endoscope1` directory at all, successful
completion also demonstrates that inference does not require Endoscope-1 RGB,
depth, or masks.

Record the wall time printed by `local_test.sh`. For a conservative projection
to `N` similar sequences, multiply that end-to-end wall time by `N`; the result
must remain below `10800` seconds. The final readiness decision requires the
requested workload to be measured on the evaluator-class A100, not inferred
from a different GPU.

Cleanup is explicit and limited to the marked temporary directory:

```bash
bash scripts/cleanup_test.sh \
  imed-method8-f1:dev \
  "$TEST_ROOT"
```

Do not tag or upload until image inspection, CUDA smoke, output validation,
exact F1 comparison, valid-pixel invariance, and the A100 runtime check all
pass.
