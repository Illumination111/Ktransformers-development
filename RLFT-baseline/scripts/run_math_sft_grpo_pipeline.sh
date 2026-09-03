#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

usage() {
    printf 'Usage: %s [--dry-run]\n' "$0"
    printf '\n'
    printf 'Runs math SFT, checkpoint export, static length gate, and GRPO formal.\n'
}

dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=1
    shift
fi
[ "$#" -eq 0 ] || die "unexpected arguments: $*"

sft_ngpus="${MATH_SFT_NGPUS:-4}"
sft_cuda="${MATH_SFT_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
grpo_ngpus="${MATH_NGPUS:-4}"
grpo_cuda="${MATH_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
static_cuda="${MATH_STATIC_GATE_CUDA_VISIBLE_DEVICES:-0,1}"
sft_data_dir="${MATH_SFT_DATA_DIR:-$B0_ROOT/data/processed/math_sft}"
sft_ckpt="${MATH_SFT_CHECKPOINT_DIR:-$B0_ROOT/checkpoints/${MATH_SFT_EXPERIMENT:-math_sft_qwen3_30b_a3b_lora}}"

if [ "$dry_run" -eq 1 ]; then
    printf 'Pipeline commands:\n'
    printf '  1. prepare_math_sft_data.py (only when SFT parquet is absent)\n'
    printf '  2. MATH_SFT_NGPUS=%s MATH_SFT_CUDA_VISIBLE_DEVICES=%s bash scripts/run_math_sft_puregpu.sh\n' "$sft_ngpus" "$sft_cuda"
    printf '  3. bash scripts/merge_math_sft_checkpoint.sh <latest_step>\n'
    printf '  4. CUDA_VISIBLE_DEVICES=%s bash scripts/run_math_static_gate.sh\n' "$static_cuda"
    printf '  5. GRPO formal on CUDA_VISIBLE_DEVICES=%s\n' "$grpo_cuda"
    exit 0
fi

require_clean_worktree
require_conda_env
require_file "$B0_HARD_MATH_TRAIN_DATA"
require_dir "$B0_MATH_MODEL"

train_sft="$sft_data_dir/train.parquet"
val_sft="$sft_data_dir/validation.parquet"
if [ ! -f "$train_sft" ] || [ ! -f "$val_sft" ]; then
    [ ! -e "$sft_data_dir" ] || [ -z "$(find "$sft_data_dir" -mindepth 1 -print -quit)" ] || \
        die "SFT data directory is incomplete and non-empty: $sft_data_dir"
    "$B0_CONDA_PREFIX/bin/python" "$B0_ROOT/scripts/prepare_math_sft_data.py" \
        --input "$B0_HARD_MATH_TRAIN_DATA" \
        --output-dir "$sft_data_dir"
fi
require_file "$train_sft"
require_file "$val_sft"

printf '[1/5] Starting SFT\n'
MATH_SFT_NGPUS="$sft_ngpus" \
MATH_SFT_CUDA_VISIBLE_DEVICES="$sft_cuda" \
MATH_SFT_DATA_DIR="$sft_data_dir" \
MATH_SFT_CHECKPOINT_DIR="$sft_ckpt" \
bash "$B0_ROOT/scripts/run_math_sft_puregpu.sh"

require_file "$sft_ckpt/latest_checkpointed_iteration.txt"
sft_step=$(tr -d '[:space:]' < "$sft_ckpt/latest_checkpointed_iteration.txt")
[[ "$sft_step" =~ ^[0-9]+$ ]] || die "invalid SFT checkpoint step: $sft_step"

printf '[2/5] Exporting SFT checkpoint step %s\n' "$sft_step"
MATH_SFT_CHECKPOINT_DIR="$sft_ckpt" \
bash "$B0_ROOT/scripts/merge_math_sft_checkpoint.sh" "$sft_step"
sft_hf="$sft_ckpt/global_step_${sft_step}/actor/huggingface"
require_dir "$sft_hf"
require_dir "$sft_hf/lora_adapter"

printf '[3/5] Running static length gate\n'
CUDA_VISIBLE_DEVICES="$static_cuda" \
MATH_MODEL_PATH="$B0_MATH_MODEL" \
bash "$B0_ROOT/scripts/run_math_static_gate.sh"

printf '[4/4] Running GRPO formal\n'
MATH_NGPUS="$grpo_ngpus" \
MATH_CUDA_VISIBLE_DEVICES="$grpo_cuda" \
MATH_MODEL_PATH="$B0_MATH_MODEL" \
MATH_LORA_ADAPTER_PATH="$sft_hf/lora_adapter" \
bash "$B0_ROOT/scripts/run_math_grpo_puregpu.sh" formal

printf 'Pipeline completed successfully. SFT step=%s, adapter=%s\n' "$sft_step" "$sft_hf/lora_adapter"
