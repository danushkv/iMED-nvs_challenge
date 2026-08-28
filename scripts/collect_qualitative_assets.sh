#!/usr/bin/env bash
set -Eeuo pipefail

# Copy one already-generated prediction per retained/diagnostic method without
# modifying image content. Endoscope1 GT is intentionally never collected.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
workspace_root=$(cd -- "$release_root/.." && pwd)

sequence=${1:-session_004_scene_2_tool_1}
frame=${2:-00000}

if [[ ! "$sequence" =~ ^session_[A-Za-z0-9_]+$ ]]; then
    echo "Invalid sequence: $sequence" >&2
    exit 2
fi
if [[ ! "$frame" =~ ^[0-9]{5}$ ]]; then
    echo "Frame must be a five-digit output index, found: $frame" >&2
    exit 2
fi

method1_source=${METHOD1_SOURCE:-$workspace_root/method1_rgbd_reprojection}
method2_source=${METHOD2_SOURCE:-$workspace_root/../method2_endo4dgs_plus}
method3_source=${METHOD3_SOURCE:-$workspace_root/../method3_surface_fusion}
method4_source=${METHOD4_SOURCE:-$workspace_root/method4_gsharp}
gallery=$release_root/assets/qualitative

mkdir -p -- "$gallery"

copy_prediction() {
    local label=$1
    local source_file=$2
    local destination=$gallery/${sequence}__${frame}__${label}.png

    if [[ ! -f "$source_file" ]]; then
        echo "[MISSING] $label: $source_file" >&2
        return 1
    fi
    if [[ -e "$destination" ]]; then
        if cmp -s -- "$source_file" "$destination"; then
            echo "[KEEP] $destination"
            return
        fi
        echo "Refusing to overwrite a differing gallery asset: $destination" >&2
        return 1
    fi
    cp --preserve=mode,timestamps -- "$source_file" "$destination"
    echo "[COPY] $destination"
}

status=0

copy_prediction mv1a \
    "$method1_source/outputs/phase11/$sequence/runs/soft_depth__mask_off__h2_nearest_r3__depth_none/rgb/$frame.png" || status=1
copy_prediction method2_metric_l1 \
    "$method2_source/results/B1_metric_l1_w5e-5/$sequence/test/ours_1000/renders/$frame.png" || status=1
copy_prediction m3b_surface \
    "$method3_source/outputs/M3_B_surface/$sequence/renders/$frame.png" || status=1
copy_prediction m3c_confidence \
    "$method3_source/outputs/M3_C_confidence/$sequence/renders/$frame.png" || status=1
copy_prediction m4a_gsharp \
    "$method4_source/target_eval_m4a_640x512/$sequence/rgb/$frame.png" || status=1

# M4-B exists only for the representative sequence and remains a dropped
# ablation; include it when present so its failure is visually inspectable.
m4b_source=$method4_source/target_eval_m4b_surface_640x512/$sequence/rgb/$frame.png
if [[ -f "$m4b_source" ]]; then
    copy_prediction m4b_surface_init "$m4b_source" || status=1
fi

if [[ $status -ne 0 ]]; then
    echo "One or more requested predictions were unavailable; existing files were left untouched." >&2
    exit 1
fi

echo
echo "Qualitative predictions collected without resizing or composition."
echo "No Endoscope1 GT, target mask, dataset file, or error map was copied."
echo "Review redistribution rights before forcing these Git-ignored images into the repository."
