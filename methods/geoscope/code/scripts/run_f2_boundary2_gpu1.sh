#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export M8_BOUNDARY_EXCLUSION=2
# Preserve the interrupted CUDA attempts. This branch only mixes images and
# computes metrics, so use CPU in a fresh directory.
export M8_OUTPUT_ROOT="$METHOD_ROOT/results/F2_boundary2_cpu"

for sequence in \
    session_004_scene_2_tool_1 \
    session_004_scene_6_tool_2 \
    session_005_scene_7_tool_2 \
    session_007_scene_11_tool_3
do
    bash "$SCRIPT_DIR/run_f1_one.sh" "$sequence" cpu
done
