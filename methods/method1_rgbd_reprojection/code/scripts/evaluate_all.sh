#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATA_ROOT PREDICTION_ROOT [evaluate.py options...]" >&2
  exit 64
fi

data_root=$1
prediction_root=$2
shift 2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
method_dir=$(cd -- "$script_dir/.." && pwd)

found=0
for prediction_dir in "$prediction_root"/session_*; do
  [[ -d "$prediction_dir" ]] || continue
  found=1
  sequence=${prediction_dir##*/}
  echo "Evaluating $sequence"
  python "$method_dir/evaluate.py" \
    --data_root "$data_root" \
    --sequence "$sequence" \
    --prediction_dir "$prediction_dir" \
    "$@"
done

if [[ $found -eq 0 ]]; then
  echo "No session_* prediction directories found under $prediction_root" >&2
  exit 66
fi
