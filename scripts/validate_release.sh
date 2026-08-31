#!/usr/bin/env bash
set -Eeuo pipefail

# Static release audit only. This does not import Python, render, train, build,
# evaluate, access the dataset, or contact the network.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)

required_files=(
    README.md
    CONTRIBUTING.md
    THIRD_PARTY.md
    docs/DATA_AND_LEGALITY.md
    docs/EVALUATION.md
    docs/QUALITATIVE.md
    docs/REPRODUCIBILITY.md
    docs/RESULTS.md
    experiments/results.csv
    methods/method1_rgbd_reprojection/README.md
    methods/method1_5_mv1a/README.md
    methods/method2_metric_depth/README.md
    methods/method3_surface_fusion/README.md
    methods/method4_gsharp/README.md
    methods/method8_hole_fallback/README.md
    methods/method1_rgbd_reprojection/code/render_sequence.py
    methods/method1_5_mv1a/code/docker_submission/phase11_candidate/Dockerfile
    methods/method2_metric_depth/code/train.py
    methods/method3_surface_fusion/code/surface_splat.py
    methods/method4_gsharp/code/train_imed.py
    methods/method4_gsharp/code/GSHARP_GSPLAT_COMMIT.txt
    methods/method8_hole_fallback/code/hole_fallback.py
    methods/method8_hole_fallback/code/METHOD8_FINAL_REPORT.md
    experiments/method8_ablation.csv
    experiments/method8_summary.json
    SOURCE_SNAPSHOT.sha256
)

failed=0
for relative in "${required_files[@]}"; do
    if [[ ! -f "$release_root/$relative" ]]; then
        echo "[MISSING] $relative" >&2
        failed=1
    fi
done

for forbidden_dir in envs toolkits downloads checkpoints __pycache__ .venv; do
    while IFS= read -r path; do
        echo "[FORBIDDEN DIRECTORY] ${path#"$release_root"/}" >&2
        failed=1
    done < <(find "$release_root" -type d -name "$forbidden_dir" -print)
done

while IFS= read -r path; do
    echo "[FORBIDDEN BINARY] ${path#"$release_root"/}" >&2
    failed=1
done < <(find "$release_root" -type f \
    \( -name '*.so' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \
       -o -name '*.npy' -o -name '*.pyc' \) -print)

while IFS= read -r path; do
    echo "[FILE OVER 10 MiB] ${path#"$release_root"/}" >&2
    failed=1
done < <(find "$release_root" -type f -size +10M -print)

expected_commit=846c07932a77a901b474c40dd7fbfe42965ab354
commit_file=$release_root/methods/method4_gsharp/code/GSHARP_GSPLAT_COMMIT.txt
if [[ -f "$commit_file" ]]; then
    recorded=$(tr -d '\r\n' < "$commit_file")
    if [[ "$recorded" != "$expected_commit" ]]; then
        echo "[GSPLAT COMMIT MISMATCH] expected $expected_commit, found $recorded" >&2
        failed=1
    fi
fi

if [[ -f "$release_root/SOURCE_SNAPSHOT.sha256" ]]; then
    if ! (cd "$release_root" && sha256sum --check --quiet SOURCE_SNAPSHOT.sha256); then
        echo "[HASH CHECK FAILED] SOURCE_SNAPSHOT.sha256" >&2
        failed=1
    fi
fi

if [[ $failed -ne 0 ]]; then
    echo "Release validation failed." >&2
    exit 1
fi

echo "Release validation passed."
echo "No environments, checkpoints, compiled extensions, arrays, or files over 10 MiB were found."
echo "This static audit does not replace method-specific runtime or leakage tests."
