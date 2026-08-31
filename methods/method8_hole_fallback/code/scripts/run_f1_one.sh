#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 SEQUENCE [DEVICE]" >&2
    exit 2
fi

METHOD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_ROOT="$(cd "$METHOD_ROOT/.." && pwd)"
SEQUENCE="$1"
DEVICE="${2:-cpu}"
DATA_ROOT="${M8_DATA_ROOT:-/mnt/cluster/datasets/iMED_NVS}"
M3B_ROOT="${M8_M3B_ROOT:-$PARENT_ROOT/outputs/M3_B_surface}"
M2_ROOT="${M8_M2_ROOT:-/mnt/cluster/workspaces/venkateda/method2_endo4dgs_plus/results/B1_metric_l1_w5e-5}"
OUTPUT_ROOT="${M8_OUTPUT_ROOT:-$METHOD_ROOT/results/F1_M2_holes}"
BOUNDARY_EXCLUSION="${M8_BOUNDARY_EXCLUSION:-0}"
EVAL_ACTIVATE="${M8_EVAL_ACTIVATE:-/home/venkateda/gan/bin/activate}"

if [[ ! -f "$EVAL_ACTIVATE" ]]; then
    echo "Evaluation environment activation file not found: $EVAL_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$EVAL_ACTIVATE"

python "$METHOD_ROOT/hole_fallback.py" \
    --data_root "$DATA_ROOT" \
    --sequence "$SEQUENCE" \
    --m3b_dir "$M3B_ROOT/$SEQUENCE" \
    --m2_dir "$M2_ROOT/$SEQUENCE" \
    --output_dir "$OUTPUT_ROOT/$SEQUENCE" \
    --device "$DEVICE" \
    --boundary_exclusion_pixels "$BOUNDARY_EXCLUSION"

