#!/usr/bin/env bash
# Multi-step Qwen3.5-122B-A10B VLM LoRA functional/stability test.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
LLAMA_FACTORY_DIR="${VLM_LLAMA_FACTORY_DIR:-/mnt/data2/wbw/LLaMA-Factory}"
KT_SOURCE_DIR="${VLM_KT_SOURCE_DIR:-/mnt/data2/wbw/ktransformers/kt-kernel}"
MODEL_PATH="${VLM_MODEL_PATH:-/mnt/data2/models/Qwen3.5-122B-A10B}"
DATASET_DIR="${VLM_DATASET_DIR:-${LLAMA_FACTORY_DIR}/data}"
DATASET_NAME="${VLM_DATASET_NAME:-mllm_demo}"
LOG_BASE="${VLM_FORMAL_LOG_BASE:-${SCRIPT_DIR}/formal_test_log}"
DEVICES="${VLM_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_STEPS=20
CUTOFF_LEN=512
PREFLIGHT_ONLY=0
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

  --model-path PATH       default: ${MODEL_PATH}
  --dataset-dir PATH      default: ${DATASET_DIR}
  --dataset-name NAME     default: ${DATASET_NAME}
  --devices LIST          exactly eight comma-separated GPU ids
  --max-steps N           default: 20; formal runs require N >= 10
  --cutoff-len N          default: 512
  --log-base PATH         default: ${LOG_BASE}
  --preflight-only        validate checkpoint, demo data, Processor and Conv3D patch
  --dry-run               preflight, render files and print the launch command
  -h, --help

The six-row mllm_demo is split deterministically into four training rows and
two evaluation rows. This validates function and stability, not model quality.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 ]] || die "missing value for $1"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) need_value "$@"; MODEL_PATH="$2"; shift ;;
        --dataset-dir) need_value "$@"; DATASET_DIR="$2"; shift ;;
        --dataset-name) need_value "$@"; DATASET_NAME="$2"; shift ;;
        --devices) need_value "$@"; DEVICES="$2"; shift ;;
        --max-steps) need_value "$@"; MAX_STEPS="$2"; shift ;;
        --cutoff-len) need_value "$@"; CUTOFF_LEN="$2"; shift ;;
        --log-base) need_value "$@"; LOG_BASE="$2"; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
    shift
done

[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "--max-steps must be a positive integer"
[[ "${MAX_STEPS}" -ge 10 ]] || die "formal tests require --max-steps >= 10"
[[ "${CUTOFF_LEN}" =~ ^[1-9][0-9]*$ ]] || die "--cutoff-len must be a positive integer"
IFS=',' read -r -a DEVICE_IDS <<< "${DEVICES}"
[[ ${#DEVICE_IDS[@]} -eq 8 ]] || die "the formal server profile requires exactly eight GPU ids"
[[ -d "${LLAMA_FACTORY_DIR}" ]] || die "LLaMA-Factory not found: ${LLAMA_FACTORY_DIR}"
[[ -f "${KT_SOURCE_DIR}/python/sft/conv3d_compat.py" ]] || die "KT source not found: ${KT_SOURCE_DIR}"

PYTHON="${VLM_PYTHON:-/mnt/data2/wbw/conda/envs/Kllama/bin/python}"
[[ -x "${PYTHON}" ]] || die "Kllama Python not found: ${PYTHON}"
ACCELERATE="$(dirname "${PYTHON}")/accelerate"
[[ -x "${ACCELERATE}" ]] || die "accelerate not found beside ${PYTHON}"

export CUDA_VISIBLE_DEVICES="${DEVICES}"
export VLM_KT_CONV3D_COMPAT="${KT_SOURCE_DIR}/python/sft/conv3d_compat.py"
export PYTHONPATH="${SCRIPT_DIR}:${LLAMA_FACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

preflight=(
    "${PYTHON}" "${SCRIPT_DIR}/validate_vlm_setup.py"
    --model-path "${MODEL_PATH}"
    --dataset-dir "${DATASET_DIR}"
    --dataset-name "${DATASET_NAME}"
)
if [[ "${PREFLIGHT_ONLY}" -eq 0 && "${DRY_RUN}" -eq 0 ]]; then
    preflight+=(--require-cuda)
fi
"${preflight[@]}"
[[ "${PREFLIGHT_ONLY}" -eq 0 ]] || exit 0

RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')"
RUN_DIR="${LOG_BASE}/${RUN_ID}"
mkdir -p "${RUN_DIR}/model_output"
TRAIN_CONFIG="${RUN_DIR}/train.yaml"
"${PYTHON}" "${SCRIPT_DIR}/render_train_config.py" \
    --template "${CONFIG_DIR}/train_vlm_lora_qwen35_122b_formal.yaml.template" \
    --output "${TRAIN_CONFIG}" \
    --model-path "${MODEL_PATH}" \
    --dataset-dir "${DATASET_DIR}" \
    --dataset-name "${DATASET_NAME}" \
    --model-output "${RUN_DIR}/model_output" \
    --max-steps "${MAX_STEPS}" \
    --cutoff-len "${CUTOFF_LEN}"

launch=(
    "${ACCELERATE}" launch
    --config_file "${CONFIG_DIR}/accelerate_ktransformers_bf16_8gpu.yaml"
    "${SCRIPT_DIR}/train_vlm_contract.py"
    "${TRAIN_CONFIG}"
)
printf 'run_dir=%s\n' "${RUN_DIR}"
printf 'command:'
printf ' %q' "${launch[@]}"
printf '\n'
[[ "${DRY_RUN}" -eq 0 ]] || exit 0

cd "${LLAMA_FACTORY_DIR}"
"${launch[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
"${PYTHON}" "${SCRIPT_DIR}/validate_formal_run.py" \
    --run-dir "${RUN_DIR}" \
    --expected-steps "${MAX_STEPS}" \
    2>&1 | tee "${RUN_DIR}/formal_validation.log"
