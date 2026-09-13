#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_f1_one.sh" session_004_scene_6_tool_2 cuda:0
bash "$SCRIPT_DIR/run_f1_one.sh" session_007_scene_11_tool_3 cuda:0

