#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_ROOT [render_sequence.py options...]" >&2
  exit 64
fi

data_root=$1
output_root=$2
shift 2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

found=0
for sequence_dir in "$data_root"/session_*; do
  [[ -d "$sequence_dir" ]] || continue
  found=1
  sequence=${sequence_dir##*/}
  echo "Rendering $sequence"
  bash "$script_dir/run_one_sequence.sh" \
    "$data_root" \
    "$sequence" \
    "$output_root" \
    "$@"
done

if [[ $found -eq 0 ]]; then
  echo "No session_* directories found under $data_root" >&2
  exit 66
fi
