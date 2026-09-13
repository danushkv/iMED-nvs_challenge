#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 IMAGE TEST_ROOT" >&2
    exit 2
fi

IMAGE="$1"
TEST_ROOT="$(realpath -- "$2")"
if [[ "$TEST_ROOT" != /tmp/method8-f1-nvs-test.* || \
      ! -f "$TEST_ROOT/.method8-f1-nvs-test-marker" || \
      "$(<"$TEST_ROOT/.method8-f1-nvs-test-marker")" != "method8-f1-nvs-test-v1" ]]; then
    echo "Refusing cleanup outside a marked Method-8 test root: $TEST_ROOT" >&2
    exit 2
fi

docker run --rm --network=none --entrypoint /bin/sh \
    -v "$TEST_ROOT:/cleanup-target" \
    "$IMAGE" \
    -c 'find /cleanup-target -xdev -depth -mindepth 1 -delete'
rmdir -- "$TEST_ROOT"
echo "Removed only: $TEST_ROOT"
