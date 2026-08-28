#!/usr/bin/env bash

# Run the Endoscope2-only source reconstruction gate for the three additional
# sequences in the fixed Method 3 comparison subset. No Endoscope1 data is
# discovered or loaded by this script.

set -euo pipefail

M4_SOURCE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
M4_SOURCE_ROOT="$(cd -- "$M4_SOURCE_SCRIPT_DIR/.." && pwd)"
M4_SOURCE_REPO_ROOT="$(cd -- "$M4_SOURCE_ROOT/.." && pwd)"
M4_SOURCE_DATA_ROOT="${1:-/mnt/cluster/datasets/iMED_NVS}"
M4_SOURCE_OUTPUT_ROOT="${2:-$M4_SOURCE_ROOT/outputs_m4a_scale_fixed}"

M4_SOURCE_SEQUENCES=(
  session_004_scene_6_tool_2
  session_005_scene_7_tool_2
  session_007_scene_11_tool_3
)

cd "$M4_SOURCE_REPO_ROOT"
source "$M4_SOURCE_ROOT/scripts/activate.sh"

for M4_SOURCE_SEQUENCE in "${M4_SOURCE_SEQUENCES[@]}"; do
  M4_SOURCE_SEQUENCE_OUTPUT="$M4_SOURCE_OUTPUT_ROOT/$M4_SOURCE_SEQUENCE"
  M4_SOURCE_CHECKPOINT="$M4_SOURCE_SEQUENCE_OUTPUT/checkpoints/m4a_final.pt"
  M4_SOURCE_METRICS="$M4_SOURCE_SEQUENCE_OUTPUT/source_reconstruction/source_metrics.json"
  if [[ ! -f "$M4_SOURCE_CHECKPOINT" ]]; then
    echo "Missing checkpoint: $M4_SOURCE_CHECKPOINT" >&2
    exit 1
  fi
  if [[ -e "$M4_SOURCE_METRICS" ]]; then
    echo "Refusing to replace existing source metrics: $M4_SOURCE_METRICS" >&2
    exit 1
  fi

  echo "Source reconstruction gate: $M4_SOURCE_SEQUENCE"
  python method4_gsharp/render_imed.py \
    --view source \
    --data-root "$M4_SOURCE_DATA_ROOT" \
    --sequence "$M4_SOURCE_SEQUENCE" \
    --checkpoint "$M4_SOURCE_CHECKPOINT" \
    --output-dir "$M4_SOURCE_OUTPUT_ROOT" \
    --device cuda \
    2>&1 | tee "$M4_SOURCE_SEQUENCE_OUTPUT/source_render.log"
done

echo "All three source reconstruction gates completed. Stop for inspection."
