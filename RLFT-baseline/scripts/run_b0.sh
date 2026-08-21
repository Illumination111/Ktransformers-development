#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

usage() {
    printf 'Usage: %s {smoke|pilot|formal} [--dry-run]\n' "$0"
}

[ "$#" -ge 1 ] || { usage >&2; exit 2; }
stage=$1
shift
dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=1
    shift
fi
[ "$#" -eq 0 ] || die "unexpected arguments: $*"

case "$stage" in
    smoke)
        experiment=b0_smoke_2step
        stage_overrides=(trainer.total_training_steps=2 trainer.save_freq=1 trainer.test_freq=1)
        ;;
    pilot)
        experiment=b0_pilot_10step
        stage_overrides=(trainer.total_training_steps=10 trainer.save_freq=-1 trainer.test_freq=5)
        ;;
    formal)
        experiment=b0_formal_60step
        stage_overrides=(trainer.total_training_steps=null trainer.save_freq="$B0_SAVE_FREQ" trainer.test_freq="$B0_TEST_FREQ")
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

require_clean_worktree
require_dir "$B0_MODEL"

checkpoint_dir="$B0_ROOT/checkpoints/$experiment"
rollout_dir="$B0_ROOT/logs/$experiment/rollouts"
validation_dir="$B0_ROOT/logs/$experiment/validation"
log_file="$B0_ROOT/logs/$experiment/console.log"

base_overrides=(
    "data.train_files=['$B0_TRAIN_DATA']"
    "data.val_files=['$B0_VAL_DATA']"
    "data.seed=$B0_SEED"
    "actor_rollout_ref.actor.data_loader_seed=$B0_SEED"
    "trainer.project_name=$B0_PROJECT"
    "trainer.experiment_name=$experiment"
    "trainer.default_local_dir=$checkpoint_dir"
    "trainer.rollout_data_dir=$rollout_dir"
    "trainer.validation_data_dir=$validation_dir"
    trainer.resume_mode=disable
    trainer.max_actor_ckpt_to_keep=2
    trainer.max_critic_ckpt_to_keep=0
    trainer.val_before_train=True
)

command=(
    bash "$B0_WORKTREE/examples/grpo_trainer/run_qwen3_30b_a3b_fsdp.sh"
    "${base_overrides[@]}"
    "${stage_overrides[@]}"
)

if [ "$dry_run" -eq 1 ]; then
    printf 'B0 stage: %s\n' "$stage"
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

require_conda_env
require_file "$B0_TRAIN_DATA"
require_file "$B0_VAL_DATA"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
visible_gpus=$(count_visible_gpus)
[ "$visible_gpus" -eq "$B0_NGPUS" ] || die "visible GPU count is $visible_gpus, expected $B0_NGPUS"

if [ "${B0_SKIP_GPU_IDLE_CHECK:-0}" != 1 ]; then
    while IFS= read -r used_mib; do
        [ "$used_mib" -lt 1024 ] || die "at least one GPU already uses ${used_mib} MiB; all 8 GPUs are required"
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
fi

if [ -d "$checkpoint_dir" ] && [ -n "$(find "$checkpoint_dir" -mindepth 1 -print -quit)" ]; then
    die "checkpoint directory is not empty: $checkpoint_dir"
fi
mkdir -p "$checkpoint_dir" "$rollout_dir" "$validation_dir" "$(dirname -- "$log_file")"

export PATH="$B0_CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$B0_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export WANDB_DIR="$B0_ROOT/logs/$experiment/wandb"
export RAY_TMPDIR="$B0_ROOT/logs/$experiment/ray"
export PYTHONHASHSEED="$B0_SEED"
export MODEL_PATH="$B0_MODEL"
export NGPUS_PER_NODE="$B0_NGPUS"
export TRAIN_BATCH_SIZE="$B0_TRAIN_BATCH_SIZE"
export PPO_MINI_BATCH_SIZE="$B0_PPO_MINI_BATCH_SIZE"
export MAX_PROMPT_LENGTH="$B0_MAX_PROMPT_LENGTH"
export MAX_RESPONSE_LENGTH="$B0_MAX_RESPONSE_LENGTH"
export PPO_MAX_TOKEN_LEN_PER_GPU="$B0_PPO_MAX_TOKEN_LEN_PER_GPU"
export ACTOR_LR="$B0_ACTOR_LR"
export KL_LOSS_COEF="$B0_KL_LOSS_COEF"
export ROLLOUT_TP="$B0_ROLLOUT_TP"
export ROLLOUT_GPU_MEM_UTIL="$B0_ROLLOUT_GPU_MEMORY_UTILIZATION"
export ROLLOUT_N="$B0_ROLLOUT_N"
export TOTAL_EPOCHS="$B0_TOTAL_EPOCHS"
export PROJECT_NAME="$B0_PROJECT"
export EXPERIMENT_NAME="$experiment"

cd "$B0_WORKTREE"
printf 'Starting stage %s; log=%s\n' "$stage" "$log_file"
"${command[@]}" 2>&1 | tee "$log_file"
