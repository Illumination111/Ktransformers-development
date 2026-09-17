#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

usage() {
    printf 'Usage: %s [--dry-run]\n' "$0"
}

dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=1
    shift
fi
[ "$#" -eq 0 ] || die "unexpected arguments: $*"

sft_ngpus="${MATH_SFT_NGPUS:-4}"
export CUDA_VISIBLE_DEVICES="${MATH_SFT_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
sft_model="${MATH_SFT_MODEL_PATH:-$B0_MATH_MODEL}"
sft_data_dir="${MATH_SFT_DATA_DIR:-$B0_ROOT/data/processed/math_sft}"
train_data="${MATH_SFT_TRAIN_DATA:-$sft_data_dir/train.parquet}"
val_data="${MATH_SFT_VALIDATION_DATA:-$sft_data_dir/validation.parquet}"
experiment="${MATH_SFT_EXPERIMENT:-math_sft_qwen3_30b_a3b_lora}"
checkpoint_dir="${MATH_SFT_CHECKPOINT_DIR:-$B0_ROOT/checkpoints/$experiment}"
log_file="$B0_ROOT/logs/$experiment/console.log"
sft_train_batch_size="${MATH_SFT_TRAIN_BATCH_SIZE:-16}"
sft_micro_batch_size="${MATH_SFT_MICRO_BATCH_SIZE_PER_GPU:-2}"
sft_max_length="${MATH_SFT_MAX_LENGTH:-2048}"
sft_max_token_len_per_gpu="${MATH_SFT_MAX_TOKEN_LEN_PER_GPU:-8192}"
sft_num_workers="${MATH_SFT_NUM_WORKERS:-4}"
sft_enable_thinking="${MATH_SFT_ENABLE_THINKING:-false}"
sft_lr="${MATH_SFT_LR:-1e-5}"
sft_total_epochs="${MATH_SFT_TOTAL_EPOCHS:-1}"
sft_save_freq="${MATH_SFT_SAVE_FREQ:-500}"
sft_test_freq="${MATH_SFT_TEST_FREQ:-500}"

require_clean_worktree
require_dir "$sft_model"
require_file "$train_data"
require_file "$val_data"
[[ "$sft_ngpus" =~ ^[1-9][0-9]*$ ]] || die "MATH_SFT_NGPUS must be a positive integer"
case "$sft_enable_thinking" in
    true|false) ;;
    *) die "MATH_SFT_ENABLE_THINKING must be true or false" ;;
esac

command=(
    torchrun --standalone --nnodes=1 --nproc_per_node="$sft_ngpus"
    -m verl.trainer.sft_trainer
    "data.train_files=['$train_data']"
    "data.val_files=['$val_data']"
    data.train_batch_size="$sft_train_batch_size"
    data.micro_batch_size_per_gpu="$sft_micro_batch_size"
    data.max_length="$sft_max_length"
    data.truncation=error
    data.use_dynamic_bsz=True
    data.max_token_len_per_gpu="$sft_max_token_len_per_gpu"
    data.messages_key=messages
    data.ignore_input_ids_mismatch=True
    data.num_workers="$sft_num_workers"
    +data.apply_chat_template_kwargs.enable_thinking="$sft_enable_thinking"
    model=hf_model
    "model.path=$sft_model"
    model.trust_remote_code=True
    +model.override_config.attn_implementation=sdpa
    model.use_remove_padding=True
    model.enable_gradient_checkpointing=True
    model.lora_rank=32
    model.lora_alpha=64
    "model.target_modules=['q_proj','k_proj','v_proj','o_proj']"
    engine=fsdp
    engine.strategy=fsdp2
    engine.model_dtype=bf16
    engine.param_offload=True
    engine.optimizer_offload=True
    engine.fsdp2_checkpoint_load_mode=hf_safetensors
    engine.use_torch_compile=False
    optim=fsdp
    optim.lr="$sft_lr"
    optim.lr_warmup_steps_ratio=0.05
    optim.weight_decay=0.01
    trainer.default_local_dir="$checkpoint_dir"
    "trainer.project_name=${B0_PROJECT}_math_sft"
    "trainer.experiment_name=$experiment"
    'trainer.logger=["console","wandb"]'
    trainer.total_epochs="$sft_total_epochs"
    trainer.save_freq="$sft_save_freq"
    trainer.test_freq="$sft_test_freq"
    trainer.max_ckpt_to_keep=2
    trainer.resume_mode=disable
    trainer.device=cuda
    trainer.nnodes=1
    trainer.n_gpus_per_node="$sft_ngpus"
)

if [ "$dry_run" -eq 1 ]; then
    printf 'Math SFT command:\n'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

require_conda_env
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
visible_gpus=$(count_visible_gpus)
[ "$visible_gpus" -eq "$sft_ngpus" ] || die "visible GPU count is $visible_gpus, expected $sft_ngpus"
if [ -d "$checkpoint_dir" ] && [ -n "$(find "$checkpoint_dir" -mindepth 1 -print -quit)" ]; then
    die "checkpoint directory is not empty: $checkpoint_dir"
fi
mkdir -p "$checkpoint_dir" "$(dirname -- "$log_file")"

export PATH="$B0_CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$B0_ROOT/scripts/peft_compat:$B0_ROOT/scripts:$B0_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export WANDB_DIR="$B0_ROOT/logs/$experiment/wandb"
export PYTHONHASHSEED="$B0_SEED"
export PYTHONUNBUFFERED=1

cd "$B0_WORKTREE"
printf 'Starting math SFT; log=%s\n' "$log_file"
"${command[@]}" 2>&1 | tee "$log_file"
printf 'SFT finished. Checkpoints are under: %s\n' "$checkpoint_dir"
