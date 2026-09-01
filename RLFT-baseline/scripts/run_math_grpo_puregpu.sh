#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

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

math_root="$B0_ROOT/data/processed/math_grpo"
train_data="$math_root/train.parquet"
val_data="$math_root/validation.parquet"
math_model="${MATH_MODEL_PATH:-$B0_MODEL}"
math_compile="${MATH_ENABLE_TORCH_COMPILE:-0}"

case "$stage" in
    smoke)
        experiment=math_grpo_2gpu_smoke_2step
        stage_overrides=(
            trainer.total_training_steps=2
            trainer.save_freq=1
            trainer.test_freq=-1
            data.train_batch_size=8
            data.val_batch_size=8
            data.max_prompt_length=512
            data.max_response_length=512
            actor_rollout_ref.rollout.n=2
            actor_rollout_ref.rollout.max_model_len=1024
            actor_rollout_ref.rollout.max_num_batched_tokens=1024
            actor_rollout_ref.rollout.max_num_seqs=8
            actor_rollout_ref.actor.ppo_mini_batch_size=4
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2048
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048
        )
        ;;
    pilot)
        experiment=math_grpo_2gpu_pilot_10step
        stage_overrides=(
            trainer.total_training_steps=10
            trainer.save_freq=10
            trainer.test_freq=-1
            data.train_batch_size=4
            data.max_response_length=1024
            actor_rollout_ref.rollout.n=2
            actor_rollout_ref.rollout.max_model_len=2048
            actor_rollout_ref.rollout.max_num_batched_tokens=2048
            actor_rollout_ref.rollout.max_num_seqs=8
            actor_rollout_ref.actor.ppo_mini_batch_size=4
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
        )
        ;;
    formal)
        experiment=math_grpo_2gpu_formal_30step
        stage_overrides=(
            trainer.total_training_steps=30
            trainer.save_freq=10
            trainer.test_freq=10
            data.train_batch_size=4
            data.max_response_length=2048
            actor_rollout_ref.rollout.n=2
            actor_rollout_ref.rollout.max_model_len=3072
            actor_rollout_ref.rollout.max_num_batched_tokens=4096
            actor_rollout_ref.rollout.max_num_seqs=8
            actor_rollout_ref.actor.ppo_mini_batch_size=4
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
        )
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

require_clean_worktree
require_dir "$math_model"
require_file "$train_data"
require_file "$val_data"

checkpoint_dir="$B0_ROOT/checkpoints/$experiment"
rollout_dir="$B0_ROOT/logs/$experiment/rollouts"
validation_dir="$B0_ROOT/logs/$experiment/validation"
log_file="$B0_ROOT/logs/$experiment/console.log"

base_overrides=(
    "data.train_files=['$train_data']"
    "data.val_files=['$val_data']"
    data.seed="$B0_SEED"
    data.train_batch_size=16
    data.val_batch_size=16
    data.max_prompt_length=1024
    data.max_response_length=2048
    data.filter_overlong_prompts=True
    data.truncation=error
    +data.apply_chat_template_kwargs.enable_thinking=True
    data.dataloader_num_workers=0
    actor_rollout_ref.actor.data_loader_seed="$B0_SEED"
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.model.path="$math_model"
    actor_rollout_ref.model.lora_rank=32
    actor_rollout_ref.model.lora_alpha=64
    "actor_rollout_ref.model.target_modules=['q_proj','k_proj','v_proj','o_proj']"
    actor_rollout_ref.model.use_remove_padding=False
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.003
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.fsdp2_checkpoint_load_mode=hf_safetensors
    actor_rollout_ref.actor.fsdp_config.use_torch_compile="$math_compile"
    actor_rollout_ref.actor.use_torch_compile="$math_compile"
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.fsdp2_checkpoint_load_mode=hf_safetensors
    actor_rollout_ref.ref.fsdp_config.use_torch_compile="$math_compile"
    actor_rollout_ref.ref.use_torch_compile="$math_compile"
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
    actor_rollout_ref.rollout.max_model_len=3072
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.max_num_seqs=16
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
    +actor_rollout_ref.rollout.engine_kwargs.sglang.lora_backend=torch_native
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_fused_qk_norm_rope=False
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$B0_ROOT/scripts/peft_compat:$B0_WORKTREE"
    '+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1"'
    ray_kwargs.ray_init.num_cpus=8
    trainer.balance_batch=True
    'trainer.logger=["console","wandb"]'
    "trainer.project_name=${B0_PROJECT}_math_grpo"
    "trainer.experiment_name=$experiment"
    "trainer.default_local_dir=$checkpoint_dir"
    "trainer.rollout_data_dir=$rollout_dir"
    "trainer.validation_data_dir=$validation_dir"
    trainer.resume_mode=disable
    trainer.max_actor_ckpt_to_keep=2
    trainer.max_critic_ckpt_to_keep=0
    trainer.val_before_train=False
    trainer.total_epochs=2
)

if [ -n "${MATH_LORA_ADAPTER_PATH:-}" ]; then
    require_dir "$MATH_LORA_ADAPTER_PATH"
    base_overrides+=(
        "actor_rollout_ref.model.lora_adapter_path=$MATH_LORA_ADAPTER_PATH"
        actor_rollout_ref.model.use_shm=True
    )
fi

command=(
    bash "$B0_WORKTREE/examples/grpo_trainer/run_qwen3_30b_a3b_fsdp.sh"
    "${base_overrides[@]}"
    "${stage_overrides[@]}"
)

if [ "$dry_run" -eq 1 ]; then
    printf 'Math GRPO stage: %s\nCommand:' "$stage"
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

require_conda_env
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
visible_gpus=$(count_visible_gpus)
[ "$visible_gpus" -eq "$B0_NGPUS" ] || die "visible GPU count is $visible_gpus, expected $B0_NGPUS"

if [ "${B0_SKIP_GPU_IDLE_CHECK:-0}" != 1 ]; then
    while IFS= read -r used_mib; do
        [ "$used_mib" -lt 1024 ] || die "selected GPU already uses ${used_mib} MiB"
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
export RAY_TMPDIR="/tmp/verl-math-grpo-ray/$stage"
export PYTHONHASHSEED="$B0_SEED"
export MODEL_PATH="$math_model"
export NGPUS_PER_NODE="$B0_NGPUS"
export PYTHONUNBUFFERED=1

cd "$B0_WORKTREE"
printf 'Starting math GRPO stage %s; log=%s\n' "$stage" "$log_file"
"${command[@]}" 2>&1 | tee "$log_file"
