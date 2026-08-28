#!/usr/bin/env bash

# Render legal 640x512 Endoscope1 target cameras for the three additional
# sequences in the fixed Method 3 comparison subset. Endoscope1 RGB is never
# discovered or loaded here; offline evaluation remains a separate command.

set -euo pipefail

M4_TARGET_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
M4_TARGET_ROOT="$(cd -- "$M4_TARGET_SCRIPT_DIR/.." && pwd)"
M4_TARGET_REPO_ROOT="$(cd -- "$M4_TARGET_ROOT/.." && pwd)"
M4_TARGET_DATA_ROOT="${1:-/mnt/cluster/datasets/iMED_NVS}"
M4_TARGET_TRAIN_ROOT="${2:-$M4_TARGET_ROOT/outputs_m4a_scale_fixed}"
M4_TARGET_RENDER_ROOT="${3:-$M4_TARGET_ROOT/target_eval_m4a_640x512}"

M4_TARGET_SEQUENCES=(
  session_004_scene_6_tool_2
  session_005_scene_7_tool_2
  session_007_scene_11_tool_3
)

cd "$M4_TARGET_REPO_ROOT"
source "$M4_TARGET_ROOT/scripts/activate.sh"

for M4_TARGET_SEQUENCE in "${M4_TARGET_SEQUENCES[@]}"; do
  M4_TARGET_TRAIN_OUTPUT="$M4_TARGET_TRAIN_ROOT/$M4_TARGET_SEQUENCE"
  M4_TARGET_CHECKPOINT="$M4_TARGET_TRAIN_OUTPUT/checkpoints/m4a_final.pt"
  M4_TARGET_SEQUENCE_OUTPUT="$M4_TARGET_RENDER_ROOT/$M4_TARGET_SEQUENCE"
  M4_TARGET_SUMMARY="$M4_TARGET_SEQUENCE_OUTPUT/target_render_summary.json"
  if [[ ! -f "$M4_TARGET_CHECKPOINT" ]]; then
    echo "Missing checkpoint: $M4_TARGET_CHECKPOINT" >&2
    exit 1
  fi
  if [[ -e "$M4_TARGET_SUMMARY" ]]; then
    echo "Refusing to replace existing target render: $M4_TARGET_SUMMARY" >&2
    exit 1
  fi
  mkdir -p "$M4_TARGET_TRAIN_OUTPUT"

  echo "Legal 640x512 target render: $M4_TARGET_SEQUENCE"
  python method4_gsharp/render_imed.py \
    --view target \
    --target-size 640 512 \
    --data-root "$M4_TARGET_DATA_ROOT" \
    --sequence "$M4_TARGET_SEQUENCE" \
    --checkpoint "$M4_TARGET_CHECKPOINT" \
    --output-dir "$M4_TARGET_RENDER_ROOT" \
    --device cuda \
    2>&1 | tee "$M4_TARGET_TRAIN_OUTPUT/target_render_640x512.log"
done

echo "All three target-camera renders completed. Stop before GT evaluation."
