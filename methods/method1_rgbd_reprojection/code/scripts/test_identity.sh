#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 DATA_ROOT SEQUENCE [FRAME_ID]" >&2
  exit 64
fi

data_root=$1
sequence=$2
frame_id=${3:-2}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
method_dir=$(cd -- "$script_dir/.." && pwd)

python "$method_dir/identity_test.py" \
  --data_root "$data_root" \
  --sequence "$sequence" \
  --frame_id "$frame_id" \
  --output_json "$method_dir/outputs/identity/${sequence}_frame_$(printf '%06d' "$frame_id").json"
