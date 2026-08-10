#!/usr/bin/env bash
# Staged Qwen3.5-122B-A10B VLM LoRA smoke test. No 122B weights are loaded in
# --preflight-only/--dry-run mode.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
LLAMA_FACTORY_DIR="${VLM_LLAMA_FACTORY_DIR:-/mnt/data2/wbw/LLaMA-Factory}"
MODEL_PATH="${VLM_MODEL_PATH:-/mnt/data2/models/Qwen3.5-122B-A10B}"
DATASET_DIR="${VLM_DATASET_DIR:-${LLAMA_FACTORY_DIR}/data}"
DATASET_NAME="${VLM_DATASET_NAME:-mllm_demo}"
LOG_BASE="${VLM_LOG_BASE:-${SCRIPT_DIR}/test_log}"
DEVICES="${VLM_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_STEPS=1
CUTOFF_LEN=512
PREFLIGHT_ONLY=0
DRY_RUN=0
ALLOW_TORCH29_CONV3D=0

usage() {
    sed -n '2,35p' "$0" | sed -n '/^# Staged/,$p' >/dev/null
    cat <<EOF
Usage: bash $(basename "$0") [options]

  --model-path PATH       default: ${MODEL_PATH}
  --dataset-dir PATH      default: ${DATASET_DIR}
  --dataset-name NAME     default: ${DATASET_NAME}
  --devices LIST          exactly eight comma-separated GPU ids
  --max-steps N           default: 1
  --cutoff-len N          default: 512
  --log-base PATH         default: ${LOG_BASE}
  --preflight-only        validate config, checkpoint index, data and Processor
  --dry-run               preflight, render files and print the launch command
  --allow-torch29-conv3d  explicitly accept the torch 2.9.x Conv3D+AMP risk
  -h, --help

Without --allow-torch29-conv3d the runner fails closed on torch 2.9.x.
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
        --allow-torch29-conv3d) ALLOW_TORCH29_CONV3D=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
    shift
done

[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "--max-steps must be a positive integer"
[[ "${CUTOFF_LEN}" =~ ^[1-9][0-9]*$ ]] || die "--cutoff-len must be a positive integer"
IFS=',' read -r -a DEVICE_IDS <<< "${DEVICES}"
[[ ${#DEVICE_IDS[@]} -eq 8 ]] || die "this server profile requires exactly eight GPU ids"
[[ -d "${LLAMA_FACTORY_DIR}" ]] || die "LLaMA-Factory not found: ${LLAMA_FACTORY_DIR}"

PYTHON="${VLM_PYTHON:-/mnt/data2/wbw/conda/envs/Kllama/bin/python}"
[[ -x "${PYTHON}" ]] || die "Kllama Python not found: ${PYTHON}"
ACCELERATE="$(dirname "${PYTHON}")/accelerate"
[[ -x "${ACCELERATE}" ]] || die "accelerate not found beside ${PYTHON}"

# Apply the requested device contract to preflight as well as launch.
export CUDA_VISIBLE_DEVICES="${DEVICES}"

preflight=(
    "${PYTHON}" "${SCRIPT_DIR}/validate_vlm_setup.py"
    --model-path "${MODEL_PATH}"
    --dataset-dir "${DATASET_DIR}"
    --dataset-name "${DATASET_NAME}"
)
if [[ "${ALLOW_TORCH29_CONV3D}" -eq 1 ]]; then
    preflight+=(--allow-torch29-conv3d)
fi
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
    --template "${CONFIG_DIR}/train_vlm_lora_qwen35_122b.yaml.template" \
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

export PYTHONPATH="${SCRIPT_DIR}:${LLAMA_FACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${ALLOW_TORCH29_CONV3D}" -eq 1 ]]; then
    export ALLOW_TORCH29_CONV3D=1
fi
cd "${LLAMA_FACTORY_DIR}"
"${launch[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
