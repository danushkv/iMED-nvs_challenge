#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-method2-endo4dgs-plus:metric-l1-w5e-5}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

docker build \
    --build-arg HTTP_PROXY \
    --build-arg HTTPS_PROXY \
    --build-arg NO_PROXY \
    --build-arg http_proxy \
    --build-arg https_proxy \
    --build-arg no_proxy \
    -f "$REPO_ROOT/docker_submission/method2_metric_l1/Dockerfile" \
    -t "$IMAGE" \
    "$REPO_ROOT"
