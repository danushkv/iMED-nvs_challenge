# Method 2 Docker submission: metric L1 depth

This folder packages the accepted Method-2 configuration for MICCAI 2026
iMED Novel View Synthesis Task 2. It layers four required Method-2 source
overrides onto the official organizer image and does not copy experiments,
results, checkpoints, metrics, caches, logs, or unrelated repository assets.

## Fixed submission configuration

The container always invokes the completed Endo-4DGS adapter with:

```text
run-sequence
--iterations 1000
--coarse-iterations 300
--depth-loss metric_l1
--depth-weight 5e-5
--no-metrics
```

All B2-B6 changes remain inactive: baseline tool masking is retained,
tool-aware reduction and dilation are disabled, appearance correction is
disabled, source DSSIM remains zero, and both confidence losses remain enabled.
The baseline RGB loss, auxiliary depth regularizers, camera loading, schedule,
rendering, export resize, and five-digit output naming are preserved.

## Source-only inference boundary

`submission_entrypoint.py` discovers sequences without examining Endoscope 1.
For each sequence it creates an internal temporary view containing only:

- `K.txt` and `pose.txt`;
- `endoscope2/L`;
- `endoscope2/depthL`;
- `endoscope2/toolL`;
- optional `target_frames.txt` metadata.

An empty synthetic `endoscope1` directory and a target frame-name list allow
the existing loader to construct target cameras from `K1_L` and camera id 1
without accessing target RGB, depth, or masks. Training and metric L1 depth
supervision therefore use Endoscope 2 only. Metrics are disabled. If
`target_frames.txt` is present, its ids must match the synchronized Endoscope-2
frame ids required by the completed Method-2 loader.

## CUDA architecture strategy

The parent is the official image:

```text
docker.synapse.org/syn74277461/imed-nvs-baseline:v1
```

The submission rebuilds `simple-knn` and
`diff-gaussian-rasterization-depth` for native architectures 7.0, 7.5, 8.0,
8.6, 8.9, and 9.0, with PTX retained for 9.0. This avoids the known mismatch
on the local RTX A5000 (sm86) and is not tied to that GPU. The trade-off is a
slower build and larger extension binaries. Hardware older than sm70 or a
future architecture unable to JIT the retained PTX remains an evaluator
ambiguity.

## Commands

All commands below are intentionally manual; none of the scripts runs merely
because these files exist.

Set paths and image name:

```bash
cd /mnt/cluster/workspaces/venkateda/method2_endo4dgs_plus
IMAGE=method2-endo4dgs-plus:metric-l1-w5e-5
SUBMISSION=docker_submission/method2_metric_l1
```

Inspect the official parent before building:

```bash
bash "$SUBMISSION/inspect_image.sh" \
  docker.synapse.org/syn74277461/imed-nvs-baseline:v1
```

Build the candidate:

```bash
bash "$SUBMISSION/scripts/build.sh" "$IMAGE"
```

Inspect the result:

```bash
bash "$SUBMISSION/inspect_image.sh" "$IMAGE"
docker image inspect "$IMAGE" --format 'image={{index .RepoTags 0}} id={{.Id}}'
```

Stage one public sequence under a unique `/tmp` test root:

```bash
bash "$SUBMISSION/scripts/stage_one_sequence.sh" \
  /mnt/cluster/datasets/iMED_NVS \
  session_004_scene_2_tool_1
```

Copy the printed `TEST_ROOT=...` assignment, then run inference and validation:

```bash
TEST_ROOT=/tmp/method2-nvs-test.REPLACE_ME
bash "$SUBMISSION/scripts/local_test.sh" "$IMAGE" "$TEST_ROOT"
```

Run the read-only validator separately if desired:

```bash
python3 "$SUBMISSION/scripts/validate_output.py" \
  --input-root "$TEST_ROOT/input" \
  --output-root "$TEST_ROOT/output" \
  --expected-width 1280 \
  --expected-height 1024
```

Compare the exact produced filename list with a generated sequential list:

```bash
SEQUENCE=session_004_scene_2_tool_1
FRAME_COUNT="$(find "$TEST_ROOT/input/$SEQUENCE/endoscope2/L" \
  -maxdepth 1 -type f -name 'frame_*.png' | wc -l)"
seq 0 "$((FRAME_COUNT - 1))" | awk '{printf "%05d.png\\n", $1}' \
  > "$TEST_ROOT/expected_filenames.txt"
find "$TEST_ROOT/output/$SEQUENCE/renders" -maxdepth 1 -type f \
  -printf '%f\n' | sort > "$TEST_ROOT/actual_filenames.txt"
diff -u "$TEST_ROOT/expected_filenames.txt" "$TEST_ROOT/actual_filenames.txt"
```

Clean only the marked temporary test directory, including root-owned Docker
outputs. This is never automatic, including after failure:

```bash
bash "$SUBMISSION/scripts/cleanup_test.sh" "$IMAGE" "$TEST_ROOT"
```

Tag and push only after the candidate passes inspection and validation:

```bash
TEAM_NAME='<team-name>'
VERSION='<version>'
TARGET_IMAGE="docker.synapse.org/syn74277461/${TEAM_NAME}-method2-nvs:${VERSION}"
docker tag "$IMAGE" "$TARGET_IMAGE"
docker login docker.synapse.org
docker push "$TARGET_IMAGE"
docker image inspect "$TARGET_IMAGE" --format 'tag={{index .RepoTags 0}} id={{.Id}}'
```

Do not submit the tag to the challenge until the pushed image is confirmed in
the Synapse Docker repository.
