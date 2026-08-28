# Method 3 Docker submission candidates

This folder packages the two candidates selected by
`METHOD3_ABLATION_REPORT.md` as independent Docker tags:

- **M3-C primary candidate:** `surface_confidence`, strongest mean PSNR under
  the baseline-adapted local protocol and no sequence-mean PSNR regression;
- **M3-B structural candidate:** `surface`, strongest corrected-geometry PSNR,
  SSIM, and raw coverage.

The lightweight parent is the confirmed MV1A submission:

```text
docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest
expected local image ID prefix: aaa952c110d1
```

Because `latest` is mutable and the registry digest was not recorded in the
Method-3 report, inspect the pulled parent and confirm its image ID before
building. Replace the base with a digest-pinned reference when the registry
digest is available.

## Packaged runtime

The parent supplies the confirmed source-only MV1A loader, camera geometry,
reprojection, and radius-3 nearest fill. The Method-3 layer adds only:

```text
confidence.py
surface_geometry.py
surface_splat.py
submission_adapter.py
configs/m3_surface_default.json
configs/m3_confidence_default.json
```

Development rendering, evaluation, metrics, stored outputs, diagnostic
exports, Endo-4DGS, stereo, and temporal fusion are absent from the Method-3
layer. The Dockerfile-specific ignore file strictly allowlists the build
context.

Both images read only `endoscope2/L`, `endoscope2/depthL`, `K.txt`, and
`pose.txt`. They write only RGB PNG predictions beneath
`/output/<sequence>/renders/`.

## Exact settings shared by both candidates

```text
visibility tolerance: 1.0 mm + 0.01 * nearest target depth
depth softness: 8.0
source tool masking: disabled
depth filtering: none
nearest-valid fill radius: 3 internal pixels
footprint scale: 1.0
radius range: 0.5 to 2.0 target pixels
support: 5x5
Mahalanobis cutoff: 9.0
depth-discontinuity gate: max(2.0 mm, 0.02 * source depth)
chunk size: 65536
invalid/unreliable surface: exact MV1A bilinear fallback
internal grid: metric-depth resolution
export: bilinear resize of filled RGB to native source resolution
```

M3-C additionally fixes geometry-confidence threshold `0.25` and power `1.0`.
Below-threshold surfels use exact MV1A bilinear fallback.

## Build and inspect

```bash
cd /mnt/cluster/workspaces/venkateda/method3_surface_fusion

docker pull docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest
docker image inspect docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest \
  --format 'id={{.Id}} repo_digests={{json .RepoDigests}}'
```

Confirm the expected `aaa952c110d1` image-ID prefix, then build the primary
M3-C image:

```bash
bash docker_submission/method3_candidates/scripts/build.sh m3c \
  method3-surface-fusion:m3c
```

Build M3-B separately if submission quota permits:

```bash
bash docker_submission/method3_candidates/scripts/build.sh m3b \
  method3-surface-fusion:m3b
```

Inspect the baked renderer before testing or tagging:

```bash
bash docker_submission/method3_candidates/inspect_image.sh \
  method3-surface-fusion:m3c

bash docker_submission/method3_candidates/inspect_image.sh \
  method3-surface-fusion:m3b
```

## One-sequence contract test

Stage only permitted inputs under `/tmp`:

```bash
bash docker_submission/method3_candidates/scripts/stage_one_sequence.sh \
  /mnt/cluster/datasets/iMED_NVS \
  session_004_scene_2_tool_1
```

Copy the printed `TEST_ROOT` assignment, then test M3-C:

```bash
TEST_ROOT=/tmp/method3-nvs-test.REPLACE_WITH_PRINTED_SUFFIX

bash docker_submission/method3_candidates/scripts/local_test.sh \
  method3-surface-fusion:m3c \
  "$TEST_ROOT"
```

The validator checks the expected frame count, exact `00000.png` sequence,
PNG decoding, RGB mode, and 1280x1024 resolution. A failed test is retained.

To test M3-B, stage a fresh test root and pass
`method3-surface-fusion:m3b` to the same command.

Run validation separately:

```bash
python3 docker_submission/method3_candidates/scripts/validate_output.py \
  --input-root "$TEST_ROOT/input" \
  --output-root "$TEST_ROOT/output"
```

Clean only the marked temporary root after review:

```bash
bash docker_submission/method3_candidates/scripts/cleanup_test.sh \
  method3-surface-fusion:m3c \
  "$TEST_ROOT"
```

## Tag and push

Use explicit version tags on a single line. For M3-C:

```bash
docker tag method3-surface-fusion:m3c \
  docker.synapse.org/syn74277461/nct_tso-method3c-nvs:v1

docker login docker.synapse.org

docker push docker.synapse.org/syn74277461/nct_tso-method3c-nvs:v1
```

For M3-B:

```bash
docker tag method3-surface-fusion:m3b \
  docker.synapse.org/syn74277461/nct_tso-method3b-nvs:v1

docker push docker.synapse.org/syn74277461/nct_tso-method3b-nvs:v1
```

Do not submit either image until its one-sequence output contract passes.

After changing any packaged source file, rebuild the same candidate tag before
retesting; Docker containers do not see edits made after the prior image build.
