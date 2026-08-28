#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 DATA_ROOT SEQUENCE [OUTPUT_ROOT] [DEVICE]" >&2
    exit 2
fi

data_root=$1
sequence=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
output_root=${3:-$project_root/outputs}
device=${4:-cuda}
python_bin=${METHOD3_PYTHON:-python}

exec "$python_bin" "$project_root/render_sequence.py" \
    --data_root "$data_root" \
    --sequence "$sequence" \
    --output_dir "$output_root/M3_A_MV1A/$sequence" \
    --renderer mv1a \
    --visibility_tolerance_mm 1.0 \
    --visibility_relative 0.01 \
    --depth_softness 8.0 \
    --fill_radius 3 \
    --device "$device"

