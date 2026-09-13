#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/method8_hole_fallback/results/F1_M2_holes" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
render_root=$1
output_root=$release_root/assets/qualitative

for sequence in session_004_scene_2_tool_1 session_004_scene_6_tool_2; do
    if [[ ! -d "$render_root/$sequence/renders" ]]; then
        echo "Missing frozen GeoSCOPE renders: $render_root/$sequence/renders" >&2
        exit 1
    fi
done

uv run python "$script_dir/create_release_gif.py" \
    --input "$render_root/session_004_scene_2_tool_1/renders" \
    --output "$output_root/geoscope_scene2.gif" \
    --label session_004_scene_2_tool_1 \
    --stride 6 \
    --width 360

uv run python "$script_dir/create_release_gif.py" \
    --input "$render_root/session_004_scene_6_tool_2/renders" \
    --output "$output_root/geoscope_scene6.gif" \
    --label session_004_scene_6_tool_2 \
    --stride 6 \
    --width 360

echo "Created two prediction-only GeoSCOPE GIFs under $output_root"
