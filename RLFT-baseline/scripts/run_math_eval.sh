#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

[ "$#" -eq 2 ] || die "usage: $0 <phase-label> <hf-model-path>"
phase=$1
model_path=$2
tokenizer_path="${MATH_EVAL_TOKENIZER_PATH:-$B0_MATH_MODEL}"
tp_size="${MATH_EVAL_TP_SIZE:-2}"
dp_size="${MATH_EVAL_DP_SIZE:-1}"
[[ "$phase" =~ ^[A-Za-z0-9._-]+$ ]] || die "phase label contains unsafe characters: $phase"
require_conda_env
require_clean_worktree
require_dir "$model_path"
require_dir "$tokenizer_path"
require_file "$B0_MATH500_DATA"
require_file "$B0_AIME2024_DATA"

output_dir="$B0_ROOT/eval/$phase"
[ ! -e "$output_dir" ] || die "evaluation output directory already exists: $output_dir"
mkdir -p "$output_dir" "$B0_ROOT/logs/eval-$phase"
export PATH="$B0_CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$B0_ROOT/scripts/peft_compat:$B0_ROOT/scripts:$B0_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$B0_CUDA_VISIBLE_DEVICES}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
export RLFT_DISABLE_CUDA_GRAPH="${RLFT_DISABLE_CUDA_GRAPH:-0}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

for benchmark in math500 aime2024; do
    "$B0_CONDA_PREFIX/bin/python" "$B0_ROOT/scripts/evaluate_math.py" \
        --benchmark "$benchmark" \
        --model-path "$model_path" \
        --tokenizer-path "$tokenizer_path" \
        --output "$output_dir/$benchmark.json" \
        --tensor-parallel-size "$tp_size" \
        --data-parallel-size "$dp_size" \
        --gpu-memory-utilization 0.6 \
        --max-new-tokens 8192 \
        --temperature 1.0 \
        2>&1 | tee "$B0_ROOT/logs/eval-$phase/$benchmark.log"
done
