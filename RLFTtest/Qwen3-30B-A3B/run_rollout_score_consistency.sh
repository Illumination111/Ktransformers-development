#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../../ktransformers-RLFT" && pwd)"
conda_bin="${CONDA_BIN:-/mnt/data2/wbw/miniconda3/bin/conda}"

# Keep model/compiler/package caches off the root filesystem.  These defaults
# can still be overridden by the caller, but every directory is rooted in the
# writable data volume by default.
cache_root="${KT_RLFT_CACHE_ROOT:-/mnt/data2/wbw/.cache/kt-RLFT}"
mkdir -p "$cache_root" "$cache_root/huggingface" "$cache_root/torch" "$cache_root/triton" "$cache_root/pip" "$cache_root/tmp"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$cache_root/xdg}"
export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$cache_root/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$cache_root/triton}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$cache_root/pip}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$cache_root/cuda}"
export TMPDIR="${TMPDIR:-$cache_root/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$CUDA_CACHE_PATH"

exec "$conda_bin" run --no-capture-output -n kt-RLFT python \
  "$script_dir/test_rollout_score_consistency.py" \
  --repo-root "$repo_root" "$@"
