#!/usr/bin/env bash

# Controlled M4-B experiment on the original representative sequence. Each
# stage is explicit and refuses to replace an existing result.

set -euo pipefail

M4_B_STAGE="${1:-}"
M4_B_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
M4_B_ROOT="$(cd -- "$M4_B_SCRIPT_DIR/.." && pwd)"
M4_B_REPO_ROOT="$(cd -- "$M4_B_ROOT/.." && pwd)"
M4_B_DATA_ROOT="${2:-/mnt/cluster/datasets/iMED_NVS}"
M4_B_TRAIN_ROOT="${3:-$M4_B_ROOT/outputs_m4b_surface}"
M4_B_RENDER_ROOT="${4:-$M4_B_ROOT/target_eval_m4b_surface_640x512}"
M4_B_SEQUENCE="session_004_scene_2_tool_1"
M4_B_SEQUENCE_OUTPUT="$M4_B_TRAIN_ROOT/$M4_B_SEQUENCE"
M4_B_VALIDATION="$M4_B_SEQUENCE_OUTPUT/initialization_validation.json"
M4_B_CHECKPOINT="$M4_B_SEQUENCE_OUTPUT/checkpoints/m4b_final.pt"
M4_B_SOURCE_METRICS="$M4_B_SEQUENCE_OUTPUT/source_reconstruction/source_metrics.json"
M4_B_TARGET_OUTPUT="$M4_B_RENDER_ROOT/$M4_B_SEQUENCE"
M4_B_TARGET_SUMMARY="$M4_B_TARGET_OUTPUT/target_render_summary.json"
M4_B_EVAL_OUTPUT="$M4_B_SEQUENCE_OUTPUT/target_evaluation_both"
M4_B_EVAL_RESULTS="$M4_B_EVAL_OUTPUT/results.json"

usage() {
  echo "Usage: bash method4_gsharp/scripts/run_m4b_one.sh {validate|train|source|target|evaluate} [DATA_ROOT] [TRAIN_ROOT] [RENDER_ROOT]" >&2
}

if [[ -z "$M4_B_STAGE" ]]; then
  usage
  exit 2
fi

cd "$M4_B_REPO_ROOT"
source "$M4_B_ROOT/scripts/activate.sh"

case "$M4_B_STAGE" in
  validate)
    if [[ -e "$M4_B_VALIDATION" ]]; then
      echo "Refusing to replace existing M4-B validation: $M4_B_VALIDATION" >&2
      exit 1
    fi
    mkdir -p "$M4_B_SEQUENCE_OUTPUT"
    python method4_gsharp/scripts/verify_m4b_initialization.py \
      --data-root "$M4_B_DATA_ROOT" \
      --sequence "$M4_B_SEQUENCE" \
      --frame-index 0 \
      --samples 16 \
      --world-scale 1.0 \
      --voxel-size 0.5 \
      --thickness-ratio 0.2 \
      --device cuda \
      --report "$M4_B_VALIDATION" \
      2>&1 | tee "$M4_B_SEQUENCE_OUTPUT/initialization_validation.log"
    ;;

  train)
    if [[ ! -f "$M4_B_VALIDATION" ]]; then
      echo "Missing M4-B initialization validation: $M4_B_VALIDATION" >&2
      exit 1
    fi
    if ! grep -q '"status": "PASS"' "$M4_B_VALIDATION"; then
      echo "M4-B initialization validation did not pass" >&2
      exit 1
    fi
    if [[ -e "$M4_B_CHECKPOINT" ]]; then
      echo "Refusing to replace existing M4-B checkpoint: $M4_B_CHECKPOINT" >&2
      exit 1
    fi
    python method4_gsharp/train_imed_m4b.py \
      --data-root "$M4_B_DATA_ROOT" \
      --sequence "$M4_B_SEQUENCE" \
      --output-dir "$M4_B_TRAIN_ROOT" \
      --coarse-steps 500 \
      --fine-steps 3000 \
      --init-max-points 50000 \
      --voxel-size 0.5 \
      --thickness-ratio 0.2 \
      --candidate-multiplier 4.0 \
      --world-scale 1.0 \
      --device cuda \
      --seed 42 \
      2>&1 | tee "$M4_B_SEQUENCE_OUTPUT/train.log"
    echo "M4-B training complete. Inspect it before source reconstruction."
    ;;

  source)
    if [[ ! -f "$M4_B_CHECKPOINT" ]]; then
      echo "Missing M4-B checkpoint: $M4_B_CHECKPOINT" >&2
      exit 1
    fi
    if [[ -e "$M4_B_SOURCE_METRICS" ]]; then
      echo "Refusing to replace existing M4-B source metrics: $M4_B_SOURCE_METRICS" >&2
      exit 1
    fi
    python method4_gsharp/render_imed.py \
      --view source \
      --data-root "$M4_B_DATA_ROOT" \
      --sequence "$M4_B_SEQUENCE" \
      --checkpoint "$M4_B_CHECKPOINT" \
      --output-dir "$M4_B_TRAIN_ROOT" \
      --device cuda \
      2>&1 | tee "$M4_B_SEQUENCE_OUTPUT/source_render.log"
    ;;

  target)
    if [[ ! -f "$M4_B_CHECKPOINT" ]]; then
      echo "Missing M4-B checkpoint: $M4_B_CHECKPOINT" >&2
      exit 1
    fi
    if [[ ! -f "$M4_B_SOURCE_METRICS" ]]; then
      echo "Missing M4-B source gate: $M4_B_SOURCE_METRICS" >&2
      exit 1
    fi
    if [[ -e "$M4_B_TARGET_SUMMARY" ]]; then
      echo "Refusing to replace existing M4-B target render: $M4_B_TARGET_SUMMARY" >&2
      exit 1
    fi
    python method4_gsharp/render_imed.py \
      --view target \
      --target-size 640 512 \
      --data-root "$M4_B_DATA_ROOT" \
      --sequence "$M4_B_SEQUENCE" \
      --checkpoint "$M4_B_CHECKPOINT" \
      --output-dir "$M4_B_RENDER_ROOT" \
      --device cuda \
      2>&1 | tee "$M4_B_SEQUENCE_OUTPUT/target_render_640x512.log"
    ;;

  evaluate)
    if [[ ! -f "$M4_B_TARGET_SUMMARY" ]]; then
      echo "Missing frozen M4-B target render: $M4_B_TARGET_SUMMARY" >&2
      exit 1
    fi
    if [[ -e "$M4_B_EVAL_RESULTS" ]]; then
      echo "Refusing to replace existing M4-B evaluation: $M4_B_EVAL_RESULTS" >&2
      exit 1
    fi
    # The only M4-B stage that accesses Endoscope1 RGB, strictly offline.
    python method1_rgbd_reprojection/evaluate.py \
      --data_root "$M4_B_DATA_ROOT" \
      --sequence "$M4_B_SEQUENCE" \
      --prediction_dir "$M4_B_TARGET_OUTPUT" \
      --evaluation_dir "$M4_B_EVAL_OUTPUT" \
      --mask_protocol both \
      --device cuda \
      2>&1 | tee "$M4_B_SEQUENCE_OUTPUT/evaluate_both.log"
    ;;

  *)
    usage
    exit 2
    ;;
esac
