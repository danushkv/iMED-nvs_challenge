#!/usr/bin/env bash
set -Eeuo pipefail

# Run one complete Method 2 ablation for one sequence on one physical GPU.
#
# The cluster dataset is copied into a unique /tmp directory because the local
# Docker daemon cannot bind-mount /mnt/cluster. Successful results are copied
# back to this repository before the root-owned temporary tree is removed by a
# short, network-disabled Docker cleanup container.

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_ablation.sh EXPERIMENT SEQUENCE GPU_ID

Implemented experiments:
  B0
  B1_metric_huber_w1e-5
  B1_metric_huber_w1e-4
  B1_metric_huber_w1e-3
  B1_metric_l1_w5e-6
  B1_metric_l1_w5e-5
  B1_metric_l1_w5e-4
  B2_tool_validmean_d0
  B2_tool_dilate3
  B2_tool_dilate5
  B3_appearance_lr1e-3_reg1e-2
  B4_source_dssim_w1e-2
  B5_no_confidence
  B6_combined_b1_b3_b4

Example:
  bash scripts/run_ablation.sh B0 session_005_scene_7_tool_2 0

Optional environment variables:
  METHOD2_IMAGE         Docker image (default: method2-endo4dgs-plus:dev)
  METHOD2_DATA_ROOT     Dataset root (default: /mnt/cluster/datasets/iMED_NVS)
  METHOD2_RESULTS_ROOT  Persistent result root (default: <repo>/results)
  METHOD2_KEEP_TMP=1    Keep successful temporary data for debugging
  METHOD2_SKIP_EXISTING=1  Exit successfully when a result already exists
EOF
}

if [[ $# -ne 3 ]]; then
    usage >&2
    exit 2
fi

experiment=$1
sequence=$2
gpu_id=$3

if [[ ! "$sequence" =~ ^session_[A-Za-z0-9_]+$ ]]; then
    echo "Invalid sequence name: $sequence" >&2
    exit 2
fi
if [[ ! "$gpu_id" =~ ^[0-9]+$ && ! "$gpu_id" =~ ^GPU-[A-Za-z0-9-]+$ ]]; then
    echo "GPU_ID must be a non-negative index or NVIDIA GPU UUID, found: $gpu_id" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
data_root=${METHOD2_DATA_ROOT:-/mnt/cluster/datasets/iMED_NVS}
results_root=${METHOD2_RESULTS_ROOT:-$project_root/results}
docker_image=${METHOD2_IMAGE:-method2-endo4dgs-plus:dev}

depth_mode=
depth_weight=
depth_huber_beta=5.0
diagnostics_interval=0
result_group=
tool_aware_loss=0
tool_mask_dilation=0
appearance_correction=0
appearance_lr=0.001
appearance_reg_weight=0.01
appearance_diagnostics_interval=0
ssim_weight=0.0
use_confidence=1

case "$experiment" in
    B0)
        depth_mode=normalized
        depth_weight=0.01
        diagnostics_interval=0
        result_group=B0_baseline
        ;;
    B1_metric_huber_w1e-5)
        depth_mode=metric_huber
        depth_weight=0.00001
        diagnostics_interval=100
        result_group=$experiment
        ;;
    B1_metric_huber_w1e-4)
        depth_mode=metric_huber
        depth_weight=0.0001
        diagnostics_interval=100
        result_group=$experiment
        ;;
    B1_metric_huber_w1e-3)
        depth_mode=metric_huber
        depth_weight=0.001
        diagnostics_interval=100
        result_group=$experiment
        ;;
    B1_metric_l1_w5e-5)
        depth_mode=metric_l1
        depth_weight=0.00005
        diagnostics_interval=100
        result_group=$experiment
        ;;
    B1_metric_l1_w5e-6)
        depth_mode=metric_l1
        depth_weight=0.000005
        diagnostics_interval=100
        result_group=$experiment
        ;;
    B1_metric_l1_w5e-4)
        depth_mode=metric_l1
        depth_weight=0.0005
        diagnostics_interval=100
        result_group=$experiment
        ;;
    B2_tool_validmean_d0)
        depth_mode=normalized
        depth_weight=0.01
        tool_aware_loss=1
        tool_mask_dilation=0
        result_group=$experiment
        ;;
    B2_tool_dilate3)
        depth_mode=normalized
        depth_weight=0.01
        tool_aware_loss=0
        tool_mask_dilation=3
        result_group=$experiment
        ;;
    B2_tool_dilate5)
        depth_mode=normalized
        depth_weight=0.01
        tool_aware_loss=0
        tool_mask_dilation=5
        result_group=$experiment
        ;;
    B3_appearance_lr1e-3_reg1e-2)
        depth_mode=normalized
        depth_weight=0.01
        appearance_correction=1
        appearance_lr=0.001
        appearance_reg_weight=0.01
        appearance_diagnostics_interval=100
        result_group=$experiment
        ;;
    B4_source_dssim_w1e-2)
        depth_mode=normalized
        depth_weight=0.01
        ssim_weight=0.01
        result_group=$experiment
        ;;
    B5_no_confidence)
        depth_mode=normalized
        depth_weight=0.01
        use_confidence=0
        result_group=$experiment
        ;;
    B6_combined_b1_b3_b4)
        depth_mode=metric_l1
        depth_weight=0.00005
        diagnostics_interval=100
        appearance_correction=1
        appearance_lr=0.001
        appearance_reg_weight=0.01
        appearance_diagnostics_interval=100
        ssim_weight=0.01
        result_group=$experiment
        ;;
    *)
        echo "Unknown or not-yet-implemented experiment: $experiment" >&2
        usage >&2
        exit 2
        ;;
esac

for required_command in docker cp mktemp tee find; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "Required command is unavailable: $required_command" >&2
        exit 1
    fi
done

source_sequence=$data_root/$sequence
for required_path in K.txt pose.txt endoscope1 endoscope2; do
    if [[ ! -e "$source_sequence/$required_path" ]]; then
        echo "Missing required sequence input: $source_sequence/$required_path" >&2
        exit 1
    fi
done

result_parent=$results_root/$result_group
result_dir=$result_parent/$sequence
if [[ -e "$result_dir" ]]; then
    if [[ ${METHOD2_SKIP_EXISTING:-0} == 1 && -f "$result_dir/results.json" ]]; then
        echo "[SKIP] Completed result already exists: $result_dir"
        exit 0
    fi
    echo "Refusing to overwrite completed or partial result: $result_dir" >&2
    exit 1
fi

mkdir -p -- "$result_parent"
tmp_root=$(mktemp -d "/tmp/method2-${experiment}-${sequence}-XXXXXX")
tmp_input=$tmp_root/input
tmp_output=$tmp_root/output
run_log=$tmp_root/${experiment}.log
result_stage=$result_parent/.${sequence}.partial.$$
cleanup_allowed=0

cleanup_on_exit() {
    original_status=$?
    final_status=$original_status
    trap - EXIT
    set +e

    if [[ $cleanup_allowed -eq 1 && ${METHOD2_KEEP_TMP:-0} != 1 ]]; then
        docker run --rm --network none \
            -v "$tmp_root:/cleanup" \
            --entrypoint bash \
            "$docker_image" \
            -c 'find /cleanup -mindepth 1 -delete'
        cleanup_status=$?
        if [[ $cleanup_status -eq 0 ]]; then
            rmdir -- "$tmp_root"
            cleanup_status=$?
        fi
        if [[ $cleanup_status -eq 0 ]]; then
            echo "[CLEANUP] Removed temporary job directory: $tmp_root"
        else
            echo "[WARNING] Results were preserved, but temporary cleanup failed: $tmp_root" >&2
            final_status=1
        fi
    else
        echo "[RETAINED] Temporary job directory: $tmp_root" >&2
        if [[ $cleanup_allowed -eq 1 ]]; then
            echo "[RETAINED] METHOD2_KEEP_TMP=1 requested retention." >&2
        else
            echo "[RETAINED] The run did not complete preservation; inspect it before cleanup." >&2
        fi
    fi

    exit "$final_status"
}
trap cleanup_on_exit EXIT

mkdir -p -- "$tmp_input" "$tmp_output"
echo "[STAGE] Copying $source_sequence to $tmp_input"
cp -a -- "$source_sequence" "$tmp_input/"

docker_command=(
    docker run --rm
    --gpus "device=$gpu_id"
    --shm-size=16g
    -v "$tmp_input:/input:ro"
    -v "$tmp_output:/output"
    "$docker_image"
    run-dataset
    --data-root /input
    --output-root /output
    --max-sequences 1
    --coarse-iterations 300
    --iterations 1000
    --depth-loss "$depth_mode"
    --depth-weight "$depth_weight"
    --depth-huber-beta "$depth_huber_beta"
    --depth-diagnostics-interval "$diagnostics_interval"
    --tool-mask-dilation "$tool_mask_dilation"
    --appearance-lr "$appearance_lr"
    --appearance-reg-weight "$appearance_reg_weight"
    --appearance-diagnostics-interval "$appearance_diagnostics_interval"
    --ssim-weight "$ssim_weight"
)

if [[ $tool_aware_loss -eq 1 ]]; then
    docker_command+=(--tool-aware-loss)
fi
if [[ $appearance_correction -eq 1 ]]; then
    docker_command+=(--appearance-correction)
fi
if [[ $use_confidence -eq 0 ]]; then
    docker_command+=(--disable-confidence)
fi

echo "[EXPERIMENT] $experiment"
echo "[SEQUENCE]   $sequence"
echo "[GPU]        physical device $gpu_id"
echo "[RESULT]     $result_dir"
printf '[RUN]       '
printf '%q ' "${docker_command[@]}"
printf '\n'

start_epoch=$(date +%s)
set +e
"${docker_command[@]}" 2>&1 | tee "$run_log"
docker_status=${PIPESTATUS[0]}
set -e
end_epoch=$(date +%s)
elapsed_seconds=$((end_epoch - start_epoch))

if [[ $docker_status -ne 0 ]]; then
    echo "[FAILED] Docker exited with status $docker_status after ${elapsed_seconds}s." >&2
    exit "$docker_status"
fi

container_result=$tmp_output/$sequence
for required_output in results.json per_view.json renders point_cloud/iteration_1000; do
    if [[ ! -e "$container_result/$required_output" ]]; then
        echo "[FAILED] Successful Docker exit but output is missing: $container_result/$required_output" >&2
        exit 1
    fi
done

mkdir -p -- "$result_stage"
while IFS= read -r -d '' output_item; do
    if [[ $(basename -- "$output_item") == _input_sequence ]]; then
        continue
    fi
    cp -a --no-preserve=ownership -- "$output_item" "$result_stage/"
done < <(find "$container_result" -mindepth 1 -maxdepth 1 -print0)

cp -- "$run_log" "$result_stage/${experiment}.log"
{
    printf 'experiment=%s\n' "$experiment"
    printf 'sequence=%s\n' "$sequence"
    printf 'physical_gpu_id=%s\n' "$gpu_id"
    printf 'docker_image=%s\n' "$docker_image"
    printf 'depth_loss=%s\n' "$depth_mode"
    printf 'depth_weight=%s\n' "$depth_weight"
    printf 'depth_huber_beta_mm=%s\n' "$depth_huber_beta"
    printf 'tool_aware_loss=%s\n' "$tool_aware_loss"
    printf 'tool_mask_dilation_radius_pixels=%s\n' "$tool_mask_dilation"
    printf 'appearance_correction=%s\n' "$appearance_correction"
    printf 'appearance_lr=%s\n' "$appearance_lr"
    printf 'appearance_reg_weight=%s\n' "$appearance_reg_weight"
    printf 'ssim_weight=%s\n' "$ssim_weight"
    printf 'use_confidence=%s\n' "$use_confidence"
    printf 'coarse_iterations=300\n'
    printf 'fine_iterations=1000\n'
    printf 'elapsed_seconds=%s\n' "$elapsed_seconds"
    printf 'source_sequence=%s\n' "$source_sequence"
    printf 'command='
    printf '%q ' "${docker_command[@]}"
    printf '\n'
} > "$result_stage/run_manifest.txt"

mv -- "$result_stage" "$result_dir"
cleanup_allowed=1

echo "[PRESERVED] $result_dir"
echo "[COMPLETE] $experiment / $sequence finished in ${elapsed_seconds}s"
