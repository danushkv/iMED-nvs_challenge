#!/usr/bin/env bash

# Offline GT evaluation for the exact four-sequence Method 3 comparison set.
# This is the only batch stage that opens Endoscope1 RGB/tool masks. Training
# checkpoints and target renders are already frozen and are never modified.

set -euo pipefail

M4_EVAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
M4_EVAL_ROOT="$(cd -- "$M4_EVAL_SCRIPT_DIR/.." && pwd)"
M4_EVAL_REPO_ROOT="$(cd -- "$M4_EVAL_ROOT/.." && pwd)"
M4_EVAL_DATA_ROOT="${1:-/mnt/cluster/datasets/iMED_NVS}"
M4_EVAL_TRAIN_ROOT="${2:-$M4_EVAL_ROOT/outputs_m4a_scale_fixed}"
M4_EVAL_RENDER_ROOT="${3:-$M4_EVAL_ROOT/target_eval_m4a_640x512}"

M4_EVAL_SEQUENCES=(
  session_004_scene_2_tool_1
  session_004_scene_6_tool_2
  session_005_scene_7_tool_2
  session_007_scene_11_tool_3
)

cd "$M4_EVAL_REPO_ROOT"
source "$M4_EVAL_ROOT/scripts/activate.sh"

for M4_EVAL_SEQUENCE in "${M4_EVAL_SEQUENCES[@]}"; do
  M4_EVAL_PREDICTION_DIR="$M4_EVAL_RENDER_ROOT/$M4_EVAL_SEQUENCE"
  M4_EVAL_SEQUENCE_OUTPUT="$M4_EVAL_TRAIN_ROOT/$M4_EVAL_SEQUENCE"
  M4_EVAL_OUTPUT="$M4_EVAL_SEQUENCE_OUTPUT/target_evaluation_both"
  M4_EVAL_RESULTS="$M4_EVAL_OUTPUT/results.json"
  if [[ ! -f "$M4_EVAL_PREDICTION_DIR/render_summary.json" ]]; then
    echo "Missing frozen target render: $M4_EVAL_PREDICTION_DIR" >&2
    exit 1
  fi
  if [[ -e "$M4_EVAL_RESULTS" ]]; then
    echo "Refusing to replace existing evaluation: $M4_EVAL_RESULTS" >&2
    exit 1
  fi

  echo "Offline two-protocol evaluation: $M4_EVAL_SEQUENCE"
  python method1_rgbd_reprojection/evaluate.py \
    --data_root "$M4_EVAL_DATA_ROOT" \
    --sequence "$M4_EVAL_SEQUENCE" \
    --prediction_dir "$M4_EVAL_PREDICTION_DIR" \
    --evaluation_dir "$M4_EVAL_OUTPUT" \
    --mask_protocol both \
    --device cuda \
    2>&1 | tee "$M4_EVAL_SEQUENCE_OUTPUT/evaluate_both.log"
done

echo "All four offline evaluations completed. Stop for M3 comparison."
