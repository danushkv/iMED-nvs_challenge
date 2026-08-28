#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 IMAGE DATA_ROOT OUTPUT_ROOT [SEQUENCE]" >&2
    exit 2
fi

image=$1
data_root=$(realpath "$2")
output_root=$(realpath -m "$3")
sequence=${4:-}

mkdir -p "$output_root"
if [[ -n "$sequence" ]]; then
    sequence_args=(--sequence "$sequence")
else
    sequence_args=(--max-sequences 1)
fi

exec docker run --rm \
    --gpus "${DOCKER_GPUS:-device=0}" \
    --user "$(id -u):$(id -g)" \
    --shm-size 16g \
    -v "$data_root:/input:ro" \
    -v "$output_root:/output:rw" \
    "$image" \
    run-dataset \
    --data-root /input \
    --output-root /output \
    "${sequence_args[@]}"
