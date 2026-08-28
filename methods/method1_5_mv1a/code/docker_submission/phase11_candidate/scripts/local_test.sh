#!/usr/bin/env sh
set -eu

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 IMAGE_NAME INPUT_ROOT OUTPUT_ROOT [container args...]" >&2
    exit 2
fi

IMAGE_NAME="$1"
INPUT_ROOT="$2"
OUTPUT_ROOT="$3"
shift 3

mkdir -p "$OUTPUT_ROOT"

docker run --rm ${IMED_NVS_DOCKER_GPU_ARGS:---gpus all} --ipc=host \
    -v "$INPUT_ROOT:/input:ro" \
    -v "$OUTPUT_ROOT:/output" \
    "$IMAGE_NAME" "$@"

python3 "$(dirname "$0")/check_outputs.py" "$INPUT_ROOT" "$OUTPUT_ROOT"
