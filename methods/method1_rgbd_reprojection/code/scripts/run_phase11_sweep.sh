#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
    echo "Usage: bash scripts/run_phase11_sweep.sh DATA_ROOT SEQUENCE OUTPUT_ROOT [extra args...]" >&2
    exit 2
fi

METHOD1_DATA_ROOT="$1"
METHOD1_SEQUENCE="$2"
METHOD1_OUTPUT_ROOT="$3"
shift 3

python phase11_sweep.py \
    --data_root "$METHOD1_DATA_ROOT" \
    --sequence "$METHOD1_SEQUENCE" \
    --output_root "$METHOD1_OUTPUT_ROOT" \
    "$@"
