#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

# The host has eight GPUs, but this profile intentionally owns only two.
# Respect an explicit caller selection; otherwise use the pinned pair.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$B0_CUDA_VISIBLE_DEVICES}"

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
        experiment=b0_puregpu_2gpu_nekoqa_persona_lora_smoke_2step
        stage_overrides=(
            trainer.total_training_steps=2
            trainer.save_freq=1
            trainer.test_freq=1
            trainer.val_before_train=False
            data.train_batch_size=8
            actor_rollout_ref.rollout.n=2
            actor_rollout_ref.rollout.max_num_batched_tokens=2048
            actor_rollout_ref.rollout.max_num_seqs=8
            actor_rollout_ref.actor.ppo_mini_batch_size=4
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
        )
        ;;
    pilot)
        experiment=b0_puregpu_2gpu_nekoqa_persona_lora_pilot_10step
        stage_overrides=(trainer.total_training_steps=10 trainer.save_freq=-1 trainer.test_freq=5)
        ;;
    formal)
        experiment=b0_puregpu_2gpu_nekoqa_persona_lora_formal_60step
        stage_overrides=(trainer.total_training_steps=60 trainer.save_freq="$B0_SAVE_FREQ" trainer.test_freq="$B0_TEST_FREQ")
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
    # NekoQA targets concise persona answers rather than chain-of-thought.
    # Without this Qwen3 spends most of the 512-token budget inside <think>.
    +data.apply_chat_template_kwargs.enable_thinking=False
    # Ray kills multiprocessing DataLoader children during normal teardown,
    # which produces a misleading final traceback despite exit code 0.  This
    # small local dataset does not need parallel DataLoader workers.
    data.dataloader_num_workers=0
    "actor_rollout_ref.actor.data_loader_seed=$B0_SEED"
    "actor_rollout_ref.rollout.name=$B0_ROLLOUT_BACKEND"
    "reward.custom_reward_function.path=$B0_REWARD_FUNCTION"
    reward.custom_reward_function.name=compute_score
    # Two GPUs cannot colocate an FP32 actor shard with a TP=2 rollout shard.
    # Store the FSDP2 actor in BF16; all model compute remains on GPU, while
    # offload only parks inactive parameters/optimizer state in host memory.
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.model.lora_rank=$B0_LORA_RANK
    actor_rollout_ref.model.lora_alpha=$B0_LORA_ALPHA
    actor_rollout_ref.model.target_modules=$B0_LORA_TARGET_MODULES
    # The historical LoRA/SGLang run used padded batches.  Remove-padding
    # produces a 3-D activation for the 0.5.8 LoRA Triton path, which requires
    # a 2-D token matrix.
    actor_rollout_ref.model.use_remove_padding=False
    # Historical Torch-2.8 environment had no FlashAttention wheel; use the
    # native SDPA implementation instead of loading a Torch-2.9 extension.
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.fsdp2_checkpoint_load_mode=hf_safetensors
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=True
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.fsdp2_checkpoint_load_mode=hf_safetensors
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=True
    actor_rollout_ref.ref.use_torch_compile=True
    # Let SGLang load the base model once.  With LoRA this avoids a first
    # full-parameter base sync from FSDP into the rollout engine.
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.max_model_len=1536
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.max_num_seqs=32
    +actor_rollout_ref.rollout.engine_kwargs.sglang.lora_backend=torch_native
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_fused_qk_norm_rope=False
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$B0_ROOT/scripts/peft_compat:$B0_WORKTREE"
    # Ray 2.57 does not populate CUDA_VISIBLE_DEVICES for these fractional
    # placement-group actors.  Select the physical device from Ray's assigned
    # accelerator id before FSDP/NCCL initializes instead of letting every
    # worker default to physical GPU 0.
    '+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1"'
    "trainer.project_name=${B0_PROJECT}_puregpu"
    "trainer.experiment_name=$experiment"
    "trainer.default_local_dir=$checkpoint_dir"
    "trainer.rollout_data_dir=$rollout_dir"
    "trainer.validation_data_dir=$validation_dir"
    trainer.resume_mode=disable
    trainer.max_actor_ckpt_to_keep=2
    trainer.max_critic_ckpt_to_keep=0
    trainer.val_before_train=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

command=(
    bash "$B0_WORKTREE/examples/grpo_trainer/run_qwen3_30b_a3b_fsdp.sh"
    "${base_overrides[@]}"
    "${stage_overrides[@]}"
)

if [ "$dry_run" -eq 1 ]; then
    printf 'B0 pure-GPU stage: %s\n' "$stage"
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
        [ "$used_mib" -lt 1024 ] || die "at least one selected GPU already uses ${used_mib} MiB; all $B0_NGPUS selected GPUs are required"
    done < <(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
fi

if [ -d "$checkpoint_dir" ] && [ -n "$(find "$checkpoint_dir" -mindepth 1 -print -quit)" ]; then
    die "checkpoint directory is not empty: $checkpoint_dir"
fi
mkdir -p "$checkpoint_dir" "$rollout_dir" "$validation_dir" "$(dirname -- "$log_file")"

export PATH="$B0_CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$B0_ROOT/scripts/peft_compat:$B0_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export WANDB_DIR="$B0_ROOT/logs/$experiment/wandb"
# Ray appends a long session/sockets suffix and AF_UNIX paths are limited to
# 107 bytes on Linux. Keep only Ray's disposable runtime files under /tmp;
# persistent console, rollout, validation, and checkpoint outputs stay in B0_ROOT.
export RAY_TMPDIR="/tmp/verl-b0-ray/$stage"
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
export PROJECT_NAME="${B0_PROJECT}_puregpu"
export EXPERIMENT_NAME="$experiment"

cd "$B0_WORKTREE"
printf 'Starting pure-GPU stage %s; log=%s\n' "$stage" "$log_file"
"${command[@]}" 2>&1 | tee "$log_file"
