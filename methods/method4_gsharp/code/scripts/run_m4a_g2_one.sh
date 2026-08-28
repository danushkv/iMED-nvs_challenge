#!/usr/bin/env bash

# Controlled final M4-A ablation on the original representative sequence.
# G2 changes only init_max_points from 50k to 100k. Each stage is explicit so
# training/source/target/evaluation gates can be inspected before continuing.

set -euo pipefail

M4_G2_STAGE="${1:-}"
M4_G2_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
M4_G2_ROOT="$(cd -- "$M4_G2_SCRIPT_DIR/.." && pwd)"
M4_G2_REPO_ROOT="$(cd -- "$M4_G2_ROOT/.." && pwd)"
M4_G2_DATA_ROOT="${2:-/mnt/cluster/datasets/iMED_NVS}"
M4_G2_TRAIN_ROOT="${3:-$M4_G2_ROOT/outputs_m4a_g2_100k}"
M4_G2_RENDER_ROOT="${4:-$M4_G2_ROOT/target_eval_m4a_g2_100k_640x512}"
M4_G2_SEQUENCE="session_004_scene_2_tool_1"
M4_G2_SEQUENCE_OUTPUT="$M4_G2_TRAIN_ROOT/$M4_G2_SEQUENCE"
M4_G2_CHECKPOINT="$M4_G2_SEQUENCE_OUTPUT/checkpoints/m4a_final.pt"
M4_G2_SOURCE_METRICS="$M4_G2_SEQUENCE_OUTPUT/source_reconstruction/source_metrics.json"
M4_G2_TARGET_OUTPUT="$M4_G2_RENDER_ROOT/$M4_G2_SEQUENCE"
M4_G2_TARGET_SUMMARY="$M4_G2_TARGET_OUTPUT/target_render_summary.json"
M4_G2_EVAL_OUTPUT="$M4_G2_SEQUENCE_OUTPUT/target_evaluation_both"
M4_G2_EVAL_RESULTS="$M4_G2_EVAL_OUTPUT/results.json"

usage() {
  echo "Usage: bash method4_gsharp/scripts/run_m4a_g2_one.sh {train|source|target|evaluate} [DATA_ROOT] [TRAIN_ROOT] [RENDER_ROOT]" >&2
}

if [[ -z "$M4_G2_STAGE" ]]; then
  usage
  exit 2
fi

cd "$M4_G2_REPO_ROOT"
source "$M4_G2_ROOT/scripts/activate.sh"

case "$M4_G2_STAGE" in
  train)
    if [[ -e "$M4_G2_CHECKPOINT" ]]; then
      echo "Refusing to replace existing G2 checkpoint: $M4_G2_CHECKPOINT" >&2
      exit 1
    fi
    mkdir -p "$M4_G2_SEQUENCE_OUTPUT"

    echo "G2 geometry gate: $M4_G2_SEQUENCE"
    python method4_gsharp/scripts/verify_geometry_convention.py \
      --data-root "$M4_G2_DATA_ROOT" \
      --sequence "$M4_G2_SEQUENCE" \
      --frame-index 0 \
      --samples 16 \
      --world-scale 1.0 \
      --report "$M4_G2_SEQUENCE_OUTPUT/geometry_report.json" \
      2>&1 | tee "$M4_G2_SEQUENCE_OUTPUT/geometry.log"

    echo "Training controlled M4-A G2: 100k init, 500 coarse, 3000 fine"
    python method4_gsharp/train_imed.py \
      --data-root "$M4_G2_DATA_ROOT" \
      --sequence "$M4_G2_SEQUENCE" \
      --output-dir "$M4_G2_TRAIN_ROOT" \
      --coarse-steps 500 \
      --fine-steps 3000 \
      --init-max-points 100000 \
      --world-scale 1.0 \
      --device cuda \
      2>&1 | tee "$M4_G2_SEQUENCE_OUTPUT/train.log"
    echo "G2 training complete. Inspect train_summary.json and train.log before source."
    ;;

  source)
    if [[ ! -f "$M4_G2_CHECKPOINT" ]]; then
      echo "Missing G2 checkpoint: $M4_G2_CHECKPOINT" >&2
      exit 1
    fi
    if [[ -e "$M4_G2_SOURCE_METRICS" ]]; then
      echo "Refusing to replace existing G2 source metrics: $M4_G2_SOURCE_METRICS" >&2
      exit 1
    fi

    python method4_gsharp/render_imed.py \
      --view source \
      --data-root "$M4_G2_DATA_ROOT" \
      --sequence "$M4_G2_SEQUENCE" \
      --checkpoint "$M4_G2_CHECKPOINT" \
      --output-dir "$M4_G2_TRAIN_ROOT" \
      --device cuda \
      2>&1 | tee "$M4_G2_SEQUENCE_OUTPUT/source_render.log"
    echo "G2 source gate complete. Inspect source_metrics.json before target."
    ;;

  target)
    if [[ ! -f "$M4_G2_CHECKPOINT" ]]; then
      echo "Missing G2 checkpoint: $M4_G2_CHECKPOINT" >&2
      exit 1
    fi
    if [[ ! -f "$M4_G2_SOURCE_METRICS" ]]; then
      echo "Missing G2 source gate: $M4_G2_SOURCE_METRICS" >&2
      exit 1
    fi
    if [[ -e "$M4_G2_TARGET_SUMMARY" ]]; then
      echo "Refusing to replace existing G2 target render: $M4_G2_TARGET_SUMMARY" >&2
      exit 1
    fi

    python method4_gsharp/render_imed.py \
      --view target \
      --target-size 640 512 \
      --data-root "$M4_G2_DATA_ROOT" \
      --sequence "$M4_G2_SEQUENCE" \
      --checkpoint "$M4_G2_CHECKPOINT" \
      --output-dir "$M4_G2_RENDER_ROOT" \
      --device cuda \
      2>&1 | tee "$M4_G2_SEQUENCE_OUTPUT/target_render_640x512.log"
    echo "G2 target render complete. Inspect its summary before evaluation."
    ;;

  evaluate)
    if [[ ! -f "$M4_G2_TARGET_SUMMARY" ]]; then
      echo "Missing frozen G2 target render: $M4_G2_TARGET_SUMMARY" >&2
      exit 1
    fi
    if [[ -e "$M4_G2_EVAL_RESULTS" ]]; then
      echo "Refusing to replace existing G2 evaluation: $M4_G2_EVAL_RESULTS" >&2
      exit 1
    fi

    # This is the only G2 stage that may access Endoscope1 RGB, strictly for
    # offline evaluation after training and target rendering are frozen.
    python method1_rgbd_reprojection/evaluate.py \
      --data_root "$M4_G2_DATA_ROOT" \
      --sequence "$M4_G2_SEQUENCE" \
      --prediction_dir "$M4_G2_TARGET_OUTPUT" \
      --evaluation_dir "$M4_G2_EVAL_OUTPUT" \
      --mask_protocol both \
      --device cuda \
      2>&1 | tee "$M4_G2_SEQUENCE_OUTPUT/evaluate_both.log"
    echo "Final M4-A G2 evaluation complete. Stop M4-A ablations and compare with the frozen 50k result."
    ;;

  *)
    usage
    exit 2
    ;;
esac
