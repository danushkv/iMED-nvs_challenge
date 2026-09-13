#!/usr/bin/env bash
set -euo pipefail

METHOD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${M8_PYTHON:-/home/venkateda/gan/bin/python}"

"$PYTHON_BIN" "$METHOD_ROOT/aggregate_fallbacks.py" \
    --results_root "$METHOD_ROOT/results" \
    --output_dir "$METHOD_ROOT/results/summary"
