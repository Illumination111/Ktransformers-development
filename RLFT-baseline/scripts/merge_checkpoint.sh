#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

[ "$#" -eq 2 ] || die "usage: $0 <smoke|formal> <global_step_number>"
stage=$1
step=$2
case "$stage" in
    smoke) experiment=b0_smoke_2step ;;
    formal) experiment=b0_formal_60step ;;
    *) die "only smoke/formal checkpoints are expected; pilot does not save" ;;
esac
[[ "$step" =~ ^[0-9]+$ ]] || die "step must be numeric"

require_clean_worktree
require_conda_env
actor_dir="$B0_ROOT/checkpoints/$experiment/global_step_$step/actor"
target_dir="$B0_ROOT/checkpoints/$experiment/global_step_$step/actor/huggingface"
require_dir "$actor_dir"
[ ! -e "$target_dir" ] || die "target already exists: $target_dir"

export PATH="$B0_CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$B0_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
cd "$B0_WORKTREE"
python -m verl.model_merger merge \
    --backend fsdp \
    --use_cpu_initialization \
    --local_dir "$actor_dir" \
    --target_dir "$target_dir"

printf 'Merged checkpoint: %s\n' "$target_dir"
