#!/usr/bin/env bash
set -Eeuo pipefail

# Assemble a curated, text-only code snapshot from the frozen experiment roots.
# This script does not train, evaluate, read the dataset, or copy outputs.
# It accepts identical existing files and refuses to overwrite differing files.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
workspace_root=$(cd -- "$release_root/.." && pwd)

method1_source=${METHOD1_SOURCE:-$workspace_root/method1_rgbd_reprojection}
method2_source=${METHOD2_SOURCE:-$workspace_root/../method2_endo4dgs_plus}
method3_source=${METHOD3_SOURCE:-$workspace_root/../method3_surface_fusion}
method4_source=${METHOD4_SOURCE:-$workspace_root/method4_gsharp}

for command_name in cp cmp dirname find mkdir mktemp sha256sum sort xargs; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command is unavailable: $command_name" >&2
        exit 1
    fi
done

for source_dir in "$method1_source" "$method2_source" "$method3_source" "$method4_source"; do
    if [[ ! -d "$source_dir" ]]; then
        echo "Missing experiment source root: $source_dir" >&2
        exit 1
    fi
done

copied=0
identical=0

copy_file() {
    local source_file=$1
    local destination_relative=$2
    local destination_file=$release_root/$destination_relative

    if [[ ! -f "$source_file" ]]; then
        echo "Missing allowlisted source file: $source_file" >&2
        exit 1
    fi

    mkdir -p -- "$(dirname -- "$destination_file")"
    if [[ -e "$destination_file" ]]; then
        if cmp -s -- "$source_file" "$destination_file"; then
            identical=$((identical + 1))
            return
        fi
        echo "Refusing to overwrite a differing release file: $destination_file" >&2
        echo "Source was: $source_file" >&2
        exit 1
    fi

    cp --preserve=mode,timestamps -- "$source_file" "$destination_file"
    copied=$((copied + 1))
}

copy_tree_text() {
    local source_dir=$1
    local destination_prefix=$2
    local path relative

    while IFS= read -r -d '' path; do
        relative=${path#"$source_dir"/}
        copy_file "$path" "$destination_prefix/$relative"
    done < <(
        find "$source_dir" -type f \
            \( -name '*.py' -o -name '*.sh' -o -name '*.md' \
               -o -name '*.txt' -o -name '*.json' -o -name '*.yml' \
               -o -name '*.yaml' -o -name 'Dockerfile' \
               -o -name 'Dockerfile.*' -o -name '.dockerignore' \) \
            -print0
    )
}

# Method 1 research implementation.
method1_files=(
    README.md
    DATASET_NOTES.md
    camera.py
    compare_baseline.py
    depth_filtering.py
    evaluate.py
    hole_filling.py
    identity_test.py
    inspect_dataset.py
    metrics_compat.py
    phase11_sweep.py
    render_sequence.py
    reprojection.py
    requirements.txt
    soft_splatting.py
    visualize_debug.py
)
for relative in "${method1_files[@]}"; do
    copy_file "$method1_source/$relative" \
        "methods/method1_rgbd_reprojection/code/$relative"
done
copy_tree_text "$method1_source/scripts" \
    "methods/method1_rgbd_reprojection/code/scripts"

# MV1A is kept as a small self-contained submission snapshot.
for relative in camera.py reprojection.py soft_splatting.py hole_filling.py; do
    copy_file "$method1_source/$relative" "methods/method1_5_mv1a/code/$relative"
done
mv1a_source=$method1_source/docker_submission/phase11_candidate
copy_tree_text "$mv1a_source" \
    "methods/method1_5_mv1a/code/docker_submission/phase11_candidate"

# Method 2 is an overlay on the official Endo-4DGS challenge image.
method2_files=(
    LICENSE.md
    METHOD2_CHANGES.md
    METHOD2_NOTES.md
    README.md
    imed_nvs_baseline.py
    train.py
    arguments/__init__.py
    scene/imed_loader.py
    configs/method2_sequence_split.txt
    scripts/run_ablation.sh
    scripts/summarize_ablation.py
)
for relative in "${method2_files[@]}"; do
    copy_file "$method2_source/$relative" "methods/method2_metric_depth/code/$relative"
done
copy_tree_text "$method2_source/docker_submission/method2_metric_l1" \
    "methods/method2_metric_depth/code/docker_submission/method2_metric_l1"

# Method 3 source tree is read-only; this operation copies from it only.
method3_files=(
    README.md
    METHOD3_NOTES.md
    METHOD3_ABLATION_REPORT.md
    confidence.py
    render_sequence.py
    surface_geometry.py
    surface_splat.py
    configs/m3_surface_default.json
    configs/m3_confidence_default.json
    scripts/run_mv1a_reference.sh
    scripts/run_surface.sh
    scripts/run_confidence.sh
)
for relative in "${method3_files[@]}"; do
    copy_file "$method3_source/$relative" "methods/method3_surface_fusion/code/$relative"
done
copy_tree_text "$method3_source/docker_submission/method3_candidates" \
    "methods/method3_surface_fusion/code/docker_submission/method3_candidates"

# Method 4 adapter and configs. gsplat itself remains a pinned external source.
method4_files=(
    README.md
    DOCKERIZATION_HANDOFF.md
    GSHARP_NOTES.md
    GSHARP_GSPLAT_COMMIT.txt
    M4A_G2_RESULT.md
    METHOD4A_SPEC.md
    METHOD4B_PLAN.md
    METHOD4B_RESULT.md
    compare_m3.py
    compare_m4ab.py
    render_imed.py
    surface_initialization.py
    train_imed.py
    train_imed_m4b.py
)
for relative in "${method4_files[@]}"; do
    copy_file "$method4_source/$relative" "methods/method4_gsharp/code/$relative"
done
copy_tree_text "$method4_source/configs" "methods/method4_gsharp/code/configs"
copy_tree_text "$method4_source/datasets" "methods/method4_gsharp/code/datasets"
copy_tree_text "$method4_source/scripts" "methods/method4_gsharp/code/scripts"
copy_tree_text "$method4_source/docker_submission" \
    "methods/method4_gsharp/code/docker_submission"
copy_file "$method4_source/third_party/gsplat/LICENSE" \
    "methods/method4_gsharp/code/licenses/GSPLAT_APACHE_2.0.txt"

# Create a stable content manifest without embedding machine-specific paths.
manifest=$release_root/SOURCE_SNAPSHOT.sha256
temporary_manifest=$(mktemp "$release_root/.source-snapshot.XXXXXX")
trap 'rm -f -- "$temporary_manifest"' EXIT
(
    cd "$release_root"
    find methods -type f -path '*/code/*' -print0 \
        | sort -z \
        | xargs -0 sha256sum
) > "$temporary_manifest"

if [[ -e "$manifest" ]]; then
    if ! cmp -s -- "$temporary_manifest" "$manifest"; then
        echo "Source snapshot differs from existing manifest: $manifest" >&2
        echo "Review/remove the old manifest explicitly; it will not be overwritten." >&2
        exit 1
    fi
else
    cp -- "$temporary_manifest" "$manifest"
fi

echo "Release source assembly complete."
echo "New files copied:       $copied"
echo "Identical files kept:   $identical"
echo "Content manifest:       $manifest"
echo "Next: bash '$release_root/scripts/validate_release.sh'"
