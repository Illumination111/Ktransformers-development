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
step_dir="$checkpoint_dir/global_step_$step"
if [ -d "$step_dir/actor" ]; then
    # GRPO-style checkpoints wrap the FSDP shards in actor/.
    local_dir="$step_dir/actor"
    default_target="$local_dir/huggingface"
else
    # The standalone SFT trainer writes FSDP shards directly in global_step_N/.
    local_dir="$step_dir"
    default_target="$step_dir/merged_hf"
fi
target_dir="${1:-$default_target}"

require_clean_worktree
require_conda_env
require_dir "$local_dir"
find "$local_dir" -maxdepth 1 -name 'model_world_size_*_rank_*.pt' -print -quit | grep -q . \
    || die "no FSDP model shards found in: $local_dir"
[ ! -e "$target_dir" ] || die "target already exists: $target_dir"

export PATH="$B0_CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$B0_ROOT/scripts/peft_compat:$B0_ROOT/scripts:$B0_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
cd "$B0_WORKTREE"
python -m verl.model_merger merge \
    --backend fsdp \
    --use_cpu_initialization \
    --local_dir "$local_dir" \
    --target_dir "$target_dir"

printf 'Exported SFT FSDP bundle: %s\n' "$target_dir"
printf 'LoRA adapter (for GRPO): %s/lora_adapter\n' "$target_dir"
printf 'NOTE: top-level HF weights are the base model, not a standalone LoRA merge.\n'
printf 'Use scripts/materialize_lora_model.py with this lora_adapter before standalone evaluation.\n'
