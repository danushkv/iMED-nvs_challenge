#!/usr/bin/env bash

# Source this file; do not execute it in a child shell:
#   source method4_gsharp/scripts/activate.sh

M4_ACTIVATE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export M4_GSHARP_ROOT="$(cd -- "$M4_ACTIVATE_DIR/.." && pwd)"

source "$M4_GSHARP_ROOT/envs/gsplat/bin/activate"

export CUDA_HOME="$M4_GSHARP_ROOT/toolkits/cuda-12.6"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-2}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

unset M4_ACTIVATE_DIR

echo "Method 4 G-SHARP environment activated"
echo "  root:      $M4_GSHARP_ROOT"
echo "  python:    $(command -v python)"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  arch:      $TORCH_CUDA_ARCH_LIST"
