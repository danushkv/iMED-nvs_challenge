#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 m3b|m3c [IMAGE]" >&2
    exit 2
fi

CANDIDATE="$1"
case "$CANDIDATE" in
    m3b)
        RENDERER=surface
        DEFAULT_IMAGE=method3-surface-fusion:m3b
        ;;
    m3c)
        RENDERER=surface_confidence
        DEFAULT_IMAGE=method3-surface-fusion:m3c
        ;;
    *)
        echo "Candidate must be m3b or m3c" >&2
        exit 2
        ;;
esac

IMAGE="${2:-$DEFAULT_IMAGE}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

docker build \
    --build-arg HTTP_PROXY \
    --build-arg HTTPS_PROXY \
    --build-arg NO_PROXY \
    --build-arg http_proxy \
    --build-arg https_proxy \
    --build-arg no_proxy \
    --build-arg "METHOD3_RENDERER=$RENDERER" \
    -f "$REPO_ROOT/docker_submission/method3_candidates/Dockerfile" \
    -t "$IMAGE" \
    "$REPO_ROOT"

echo "Built $IMAGE with fixed renderer=$RENDERER"
