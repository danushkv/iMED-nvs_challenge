#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 IMAGE TEST_ROOT" >&2
    exit 2
fi

IMAGE="$1"
TEST_ROOT="$(realpath -- "$2")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$TEST_ROOT" != /tmp/method3-nvs-test.* || ! -f "$TEST_ROOT/.method3-nvs-test-marker" ]]; then
    echo "Refusing unrecognized test root: $TEST_ROOT" >&2
    exit 2
fi

STARTED="$(date +%s)"
set +e
docker run --rm --gpus all --ipc=host --network=none \
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
    --output-root "$TEST_ROOT/output"

echo "One-sequence test completed in ${ELAPSED}s."
echo "Temporary test retained at: $TEST_ROOT"
