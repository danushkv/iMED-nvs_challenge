#!/usr/bin/env bash

# Train the unchanged, millimetre-scale-corrected M4-A configuration on the
# three preselected development sequences. This script stops after training;
# source sanity rendering and target evaluation remain explicit later gates.

set -euo pipefail

M4_DEV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
M4_DEV_ROOT="$(cd -- "$M4_DEV_SCRIPT_DIR/.." && pwd)"
M4_DEV_REPO_ROOT="$(cd -- "$M4_DEV_ROOT/.." && pwd)"
M4_DEV_DATA_ROOT="${1:-/mnt/cluster/datasets/iMED_NVS}"
M4_DEV_OUTPUT_ROOT="${2:-$M4_DEV_ROOT/outputs_m4a_scale_fixed}"

M4_DEV_SEQUENCES=(
  session_004_scene_6_tool_2
  session_005_scene_7_tool_2
  session_007_scene_11_tool_3
)

cd "$M4_DEV_REPO_ROOT"
source "$M4_DEV_ROOT/scripts/activate.sh"

for M4_DEV_SEQUENCE in "${M4_DEV_SEQUENCES[@]}"; do
  M4_DEV_SEQUENCE_OUTPUT="$M4_DEV_OUTPUT_ROOT/$M4_DEV_SEQUENCE"
  M4_DEV_CHECKPOINT="$M4_DEV_SEQUENCE_OUTPUT/checkpoints/m4a_final.pt"
  if [[ -e "$M4_DEV_CHECKPOINT" ]]; then
    echo "Refusing to replace existing checkpoint: $M4_DEV_CHECKPOINT" >&2
    exit 1
  fi
  mkdir -p "$M4_DEV_SEQUENCE_OUTPUT"

  echo "Geometry gate: $M4_DEV_SEQUENCE"
  python method4_gsharp/scripts/verify_geometry_convention.py \
    --data-root "$M4_DEV_DATA_ROOT" \
    --sequence "$M4_DEV_SEQUENCE" \
    --frame-index 0 \
    --samples 16 \
    --world-scale 1.0 \
    --report "$M4_DEV_SEQUENCE_OUTPUT/geometry_report.json" \
    2>&1 | tee "$M4_DEV_SEQUENCE_OUTPUT/geometry.log"

  echo "Training unchanged corrected M4-A: $M4_DEV_SEQUENCE"
  python method4_gsharp/train_imed.py \
    --data-root "$M4_DEV_DATA_ROOT" \
    --sequence "$M4_DEV_SEQUENCE" \
    --output-dir "$M4_DEV_OUTPUT_ROOT" \
    --coarse-steps 500 \
    --fine-steps 3000 \
    --init-max-points 50000 \
    --world-scale 1.0 \
    --device cuda \
    2>&1 | tee "$M4_DEV_SEQUENCE_OUTPUT/train.log"
done

echo "All three M4-A development trainings completed. Stop for log inspection."
