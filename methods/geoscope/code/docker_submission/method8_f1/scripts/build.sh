#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-imed-method8-f1:dev}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

docker build --progress=plain \
    --build-arg HTTP_PROXY \
    --build-arg HTTPS_PROXY \
    --build-arg NO_PROXY \
    --build-arg http_proxy \
    --build-arg https_proxy \
    --build-arg no_proxy \
    -t "$IMAGE" \
    "$SUBMISSION_ROOT"
