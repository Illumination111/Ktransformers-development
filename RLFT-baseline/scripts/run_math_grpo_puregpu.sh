#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

# Do not inherit the 2-GPU CUDA_VISIBLE_DEVICES=0,1 from NekoQA / the failed
# formal tmux session. Override with MATH_CUDA_VISIBLE_DEVICES if needed.
math_ngpus="${MATH_NGPUS:-4}"
export CUDA_VISIBLE_DEVICES="${MATH_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

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

train_data="${MATH_TRAIN_DATA:-$B0_HARD_MATH_TRAIN_DATA}"
val_data="${MATH_VALIDATION_DATA:-$B0_MATH_VALIDATION_GATE_DATA}"
math_reward="${MATH_REWARD_FUNCTION:-$B0_MATH_REWARD_FUNCTION}"
math_model="${MATH_MODEL_PATH:-$B0_MATH_MODEL}"
math_compile="${MATH_ENABLE_TORCH_COMPILE:-0}"
# MultiTurnSFTDataset tokenized each completed assistant turn without the
# non-thinking generation prefix, so retain the raw Qwen assistant prefix at
# rollout time.  This matches the actual SFT token stream.
math_enable_thinking="${MATH_ENABLE_THINKING:-true}"
case "$math_enable_thinking" in
    true|false) ;;
    *) die "MATH_ENABLE_THINKING must be true or false" ;;
esac
# The sharded HF loader initializes the model on meta tensors and only
# supports creating a fresh LoRA adapter.  Warm-starting an existing adapter
# must load a real full state before FSDP2 shards it.
fsdp2_load_mode=hf_safetensors
if [ -n "${MATH_LORA_ADAPTER_PATH:-}" ]; then
    fsdp2_load_mode=full_state
fi

case "$stage" in
    smoke)
        experiment="${MATH_EXPERIMENT:-math_grpo_v2_4gpu_smoke_2step}"
        stage_overrides=(
            trainer.total_training_steps=2
            trainer.save_freq=1
            trainer.test_freq=-1
            trainer.val_before_train=False
            data.train_batch_size=2
            data.val_batch_size=8
            data.max_prompt_length=1024
            data.max_response_length=4096
            actor_rollout_ref.rollout.n=4
            actor_rollout_ref.rollout.max_model_len=6144
            actor_rollout_ref.rollout.max_num_batched_tokens=4096
            actor_rollout_ref.rollout.max_num_seqs=8
            actor_rollout_ref.actor.ppo_mini_batch_size=2
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240
        )
        ;;
    pilot)
        experiment="${MATH_EXPERIMENT:-math_grpo_v2_4gpu_pilot_8step}"
        stage_overrides=(
            trainer.total_training_steps=8
            trainer.save_freq=8
            trainer.test_freq=8
            trainer.val_before_train=True
            data.train_batch_size=4
            data.max_response_length=4096
            actor_rollout_ref.rollout.n=4
            actor_rollout_ref.rollout.max_model_len=6144
            actor_rollout_ref.rollout.max_num_batched_tokens=4096
            actor_rollout_ref.rollout.max_num_seqs=8
            actor_rollout_ref.actor.ppo_mini_batch_size=4
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240
        )
        ;;
    formal)
        experiment="${MATH_EXPERIMENT:-math_grpo_v2_4gpu_formal_80step}"
        stage_overrides=(
            trainer.total_training_steps=80
            trainer.save_freq=20
            trainer.test_freq=20
            trainer.val_before_train=True
            data.train_batch_size=4
            data.max_response_length=8192
            actor_rollout_ref.rollout.n=8
            actor_rollout_ref.rollout.max_model_len=10240
            actor_rollout_ref.rollout.max_num_batched_tokens=8192
            actor_rollout_ref.rollout.max_num_seqs=8
            actor_rollout_ref.actor.ppo_mini_batch_size=4
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240
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
require_file "$math_reward"

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
    data.max_response_length=8192
    data.filter_overlong_prompts=True
    data.truncation=error
    +data.apply_chat_template_kwargs.enable_thinking="$math_enable_thinking"
    data.dataloader_num_workers=0
    actor_rollout_ref.actor.data_loader_seed="$B0_SEED"
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.model.path="$math_model"
    actor_rollout_ref.model.lora_rank=32
    actor_rollout_ref.model.lora_alpha=64
    "actor_rollout_ref.model.target_modules=['q_proj','k_proj','v_proj','o_proj']"
    actor_rollout_ref.model.use_remove_padding=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=512
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.fsdp2_checkpoint_load_mode="$fsdp2_load_mode"
    actor_rollout_ref.actor.fsdp_config.use_torch_compile="$math_compile"
    actor_rollout_ref.actor.use_torch_compile="$math_compile"
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.fsdp2_checkpoint_load_mode="$fsdp2_load_mode"
    actor_rollout_ref.ref.fsdp_config.use_torch_compile="$math_compile"
    actor_rollout_ref.ref.use_torch_compile="$math_compile"
    actor_rollout_ref.rollout.tensor_model_parallel_size="$math_ngpus"
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240
    actor_rollout_ref.rollout.max_model_len=10240
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.max_num_seqs=8
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.entropy_from_logits_chunk_size=512
    "reward.custom_reward_function.path=$math_reward"
    reward.custom_reward_function.name=compute_score
    +actor_rollout_ref.rollout.engine_kwargs.sglang.lora_backend=torch_native
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_fused_qk_norm_rope=False
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$B0_ROOT/scripts/peft_compat:$B0_ROOT/scripts:$B0_WORKTREE"
    '+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1"'
    ray_kwargs.ray_init.num_cpus=16
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
    trainer.val_before_train=True
    trainer.total_epochs=2
    trainer.n_gpus_per_node="$math_ngpus"
    trainer.nnodes=1
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
if [ "$stage" = formal ] && [ "${MATH_SKIP_STATIC_GATE:-0}" != 1 ]; then
    static_gate="$B0_ROOT/metrics/math_static_gate/selection.json"
    require_file "$static_gate"
    "$B0_CONDA_PREFIX/bin/python" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("gate_passed"):
    raise SystemExit("math static length gate did not pass")
' "$static_gate"
fi
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
visible_gpus=$(count_visible_gpus)
[ "$visible_gpus" -eq "$math_ngpus" ] || die "visible GPU count is $visible_gpus, expected $math_ngpus"

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
export PYTHONPATH="$B0_ROOT/scripts/peft_compat:$B0_ROOT/scripts:$B0_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export WANDB_DIR="$B0_ROOT/logs/$experiment/wandb"
export RAY_TMPDIR="/tmp/verl-math-grpo-ray/$stage"
export PYTHONHASHSEED="$B0_SEED"
export MODEL_PATH="$math_model"
export NGPUS_PER_NODE="$math_ngpus"
export PYTHONUNBUFFERED=1

cd "$B0_WORKTREE"
printf 'Starting math GRPO stage %s; log=%s\n' "$stage" "$log_file"
"${command[@]}" 2>&1 | tee "$log_file"

audit_output="$B0_ROOT/metrics/$experiment/rollout_audit.json"
audit_args=(
    "$B0_CONDA_PREFIX/bin/python"
    "$B0_ROOT/scripts/audit_math_rollouts.py"
    --rollout-dir "$rollout_dir"
    --output "$audit_output"
    --tokenizer-path "$math_model"
)
case "$stage" in
    smoke) audit_args+=(--max-response-tokens 4096 --min-format-rate 0.5 --min-answer-extracted-rate 0.5 --max-clip-rate 0.5) ;;
    pilot) audit_args+=(--max-response-tokens 4096 --min-effective-steps 1) ;;
    formal) audit_args+=(--max-response-tokens 8192 --min-effective-steps 50 --max-clip-rate 0.5) ;;
esac
"${audit_args[@]}"
