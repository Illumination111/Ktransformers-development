#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

[ "$#" -ge 1 ] || die "usage: $0 <global_step> [target_dir]"
step=$1
[[ "$step" =~ ^[0-9]+$ ]] || die "global_step must be numeric"
shift
[ "$#" -le 1 ] || die "usage: $0 <global_step> [target_dir]"

experiment="${MATH_SFT_EXPERIMENT:-math_sft_qwen3_30b_a3b_lora}"
checkpoint_dir="${MATH_SFT_CHECKPOINT_DIR:-$B0_ROOT/checkpoints/$experiment}"
actor_dir="$checkpoint_dir/global_step_$step/actor"
target_dir="${1:-$actor_dir/huggingface}"

require_clean_worktree
require_conda_env
require_dir "$actor_dir"
[ ! -e "$target_dir" ] || die "target already exists: $target_dir"

export PATH="$B0_CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$B0_ROOT/scripts/peft_compat:$B0_ROOT/scripts:$B0_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
cd "$B0_WORKTREE"
python -m verl.model_merger merge \
    --backend fsdp \
    --use_cpu_initialization \
    --local_dir "$actor_dir" \
    --target_dir "$target_dir"

printf 'Merged SFT checkpoint: %s\n' "$target_dir"
printf 'LoRA adapter (for GRPO): %s/lora_adapter\n' "$target_dir"
