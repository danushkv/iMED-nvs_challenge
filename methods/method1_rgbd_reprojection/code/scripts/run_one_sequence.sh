#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DATA_ROOT SEQUENCE OUTPUT_ROOT [render_sequence.py options...]" >&2
  exit 64
fi

data_root=$1
sequence=$2
output_root=$3
shift 3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
method_dir=$(cd -- "$script_dir/.." && pwd)

python "$method_dir/render_sequence.py" \
  --data_root "$data_root" \
  --sequence "$sequence" \
  --output_dir "$output_root/$sequence" \
  "$@"
