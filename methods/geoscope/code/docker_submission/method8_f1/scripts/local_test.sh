#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 IMAGE TEST_ROOT" >&2
    exit 2
fi

IMAGE="$1"
TEST_ROOT="$(realpath -- "$2")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_GPU_SELECTION="${DOCKER_GPUS:-all}"

if [[ "$TEST_ROOT" != /tmp/method8-f1-nvs-test.* || \
      ! -f "$TEST_ROOT/.method8-f1-nvs-test-marker" || \
      "$(<"$TEST_ROOT/.method8-f1-nvs-test-marker")" != "method8-f1-nvs-test-v1" ]]; then
    echo "Refusing unrecognized test root: $TEST_ROOT" >&2
    exit 2
fi
if [[ ! -d "$TEST_ROOT/input" || ! -d "$TEST_ROOT/output" ]]; then
    echo "Staged input/output directories are missing under $TEST_ROOT" >&2
    exit 2
fi

STARTED="$(date +%s)"
set +e
timeout --kill-after=60s 10800 \
    docker run --rm \
      --gpus "$DOCKER_GPU_SELECTION" \
      --ipc=host \
      --network=none \
      --memory=120g \
      --pids-limit=1024 \
      -v "$TEST_ROOT/input:/input:ro" \
      -v "$TEST_ROOT/output:/output" \
      "$IMAGE" 2>&1 | tee "$TEST_ROOT/container.log"
CONTAINER_STATUS="${PIPESTATUS[0]}"
set -e
ELAPSED="$(( $(date +%s) - STARTED ))"

if [[ "$CONTAINER_STATUS" -ne 0 ]]; then
    echo "Container failed with status $CONTAINER_STATUS after ${ELAPSED}s." >&2
    echo "Temporary test retained at: $TEST_ROOT" >&2
    exit "$CONTAINER_STATUS"
fi

python3 "$SCRIPT_DIR/validate_output.py" \
    --input-root "$TEST_ROOT/input" \
    --output-root "$TEST_ROOT/output" \
    --expected-width 1280 \
    --expected-height 1024

echo "One-sequence network-disabled test completed in ${ELAPSED}s."
echo "Temporary test retained at: $TEST_ROOT"
