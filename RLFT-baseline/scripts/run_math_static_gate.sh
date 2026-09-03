#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

[ "$#" -eq 0 ] || die "usage: $0"
require_conda_env
require_clean_worktree
math_model="${MATH_MODEL_PATH:-$B0_MATH_MODEL}"
train_data="$B0_HARD_MATH_TRAIN_DATA"
require_dir "$math_model"
require_file "$train_data"

output_dir="$B0_ROOT/eval/math_static_gate"
metrics_dir="$B0_ROOT/metrics/math_static_gate"
mkdir -p "$output_dir" "$metrics_dir"
export PATH="$B0_CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$B0_ROOT/scripts/peft_compat:$B0_ROOT/scripts:$B0_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$B0_CUDA_VISIBLE_DEVICES}"

reports=()
for length in 4096 8192 10240; do
    report="$output_dir/hardmath-${length}.json"
    [ ! -e "$report" ] || die "static gate output already exists: $report"
    "$B0_CONDA_PREFIX/bin/python" "$B0_ROOT/scripts/evaluate_math.py" \
        --benchmark hardmath \
        --model-path "$math_model" \
        --data "$train_data" \
        --output "$report" \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.6 \
        --max-new-tokens "$length" \
        --temperature 1.0 \
        --limit 64 \
        --seed-count 8
    reports+=("$report")
done

"$B0_CONDA_PREFIX/bin/python" "$B0_ROOT/scripts/compare_math_static_gates.py" \
    "${reports[@]}" \
    --output "$metrics_dir/selection.json" \
    --max-clip-rate 0.5
