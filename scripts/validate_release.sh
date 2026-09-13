#!/usr/bin/env bash
set -Eeuo pipefail

# Static release audit only. This does not import Python, render, train, build,
# evaluate, access the dataset, or contact the network.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)

required_files=(
    README.md
    LICENSE
    CONTRIBUTING.md
    THIRD_PARTY.md
    CITATION.cff
    pyproject.toml
    uv.lock
    assets/geoscope_overview.svg
    assets/method_lineage.svg
    assets/qualitative/geoscope_scene2.gif
    assets/qualitative/geoscope_scene2.json
    assets/qualitative/geoscope_scene6.gif
    assets/qualitative/geoscope_scene6.json
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
    methods/geoscope/README.md
    methods/method1_rgbd_reprojection/code/render_sequence.py
    methods/method1_5_mv1a/code/docker_submission/phase11_candidate/Dockerfile
    methods/method2_metric_depth/code/train.py
    methods/method3_surface_fusion/code/surface_splat.py
    methods/method4_gsharp/code/train_imed.py
    methods/method4_gsharp/code/GSHARP_GSPLAT_COMMIT.txt
    methods/geoscope/code/hole_fallback.py
    methods/geoscope/code/METHOD8_FINAL_REPORT.md
    methods/geoscope/code/docker_submission/method8_f1/Dockerfile
    methods/geoscope/code/docker_submission/method8_f1/combined_entrypoint.py
    methods/geoscope/code/docker_submission/method8_f1/f1_fusion.py
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

# Audit only files Git would publish: tracked files plus untracked files that
# are not ignored. This deliberately permits a local ignored `.venv` created
# by `uv sync`, while still rejecting one if it could enter a commit.
if ! git -C "$release_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[GIT REQUIRED] release validation must run inside the release repository" >&2
    failed=1
else
    while IFS= read -r -d '' relative; do
        path=$release_root/$relative
        [[ -f "$path" ]] || continue

        case "/$relative/" in
            */envs/*|*/toolkits/*|*/downloads/*|*/checkpoints/*|*/__pycache__/*|*/.venv/*)
                echo "[FORBIDDEN DIRECTORY] $relative" >&2
                failed=1
                ;;
        esac

        case "$relative" in
            *.so|*.pt|*.pth|*.ckpt|*.npy|*.pyc)
                echo "[FORBIDDEN BINARY] $relative" >&2
                failed=1
                ;;
        esac

        if (( $(stat -c '%s' "$path") > 10485760 )); then
            echo "[FILE OVER 10 MiB] $relative" >&2
            failed=1
        fi
    done < <(git -C "$release_root" ls-files --cached --others --exclude-standard -z)
fi

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
echo "No publishable environments, checkpoints, compiled extensions, arrays, or files over 10 MiB were found."
echo "Ignored local environments and generated outputs were not audited because Git will not publish them."
echo "This static audit does not replace method-specific runtime or leakage tests."
