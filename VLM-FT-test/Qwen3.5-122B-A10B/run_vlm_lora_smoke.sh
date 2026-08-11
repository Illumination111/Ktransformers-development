#!/usr/bin/env bash
# Staged Qwen3.5-122B-A10B VLM LoRA smoke test. No 122B weights are loaded in
# --preflight-only/--dry-run mode.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
LLAMA_FACTORY_DIR="${VLM_LLAMA_FACTORY_DIR:-/mnt/data2/wbw/LLaMA-Factory}"
KT_SOURCE_DIR="${VLM_KT_SOURCE_DIR:-/mnt/data2/wbw/ktransformers/kt-kernel}"
MODEL_PATH="${VLM_MODEL_PATH:-/mnt/data2/models/Qwen3.5-122B-A10B}"
DATASET_DIR="${VLM_DATASET_DIR:-${LLAMA_FACTORY_DIR}/data}"
DATASET_NAME="${VLM_DATASET_NAME:-mllm_demo}"
LOG_BASE="${VLM_LOG_BASE:-${SCRIPT_DIR}/test_log}"
DEVICES="${VLM_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_STEPS=1
CUTOFF_LEN=512
LORA_SCOPE="${VLM_LORA_SCOPE:-text}"
EXPECTED_LAYERS="${VLM_EXPECTED_LAYERS:-48}"
EXPECTED_EXPERTS="${VLM_EXPECTED_EXPERTS:-256}"
EXPECTED_TOP_K="${VLM_EXPECTED_TOP_K:-8}"
EXPECTED_WRAPPERS="${VLM_EXPECTED_KT_WRAPPERS:-${EXPECTED_LAYERS}}"
PREFLIGHT_ONLY=0
DRY_RUN=0

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
  --lora-scope SCOPE      text, vision or all; default: ${LORA_SCOPE}
  --log-base PATH         default: ${LOG_BASE}
  --preflight-only        validate config, checkpoint index, data and Processor
  --dry-run               preflight, render files and print the launch command
  -h, --help

On torch 2.9.x the runner requires ms-swift>=4.4.2,<4.5 and verifies its
Conv3D replacement before loading the VLM.
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
        --lora-scope) need_value "$@"; LORA_SCOPE="$2"; shift ;;
        --log-base) need_value "$@"; LOG_BASE="$2"; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
    shift
done

[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "--max-steps must be a positive integer"
[[ "${CUTOFF_LEN}" =~ ^[1-9][0-9]*$ ]] || die "--cutoff-len must be a positive integer"
[[ "${LORA_SCOPE}" =~ ^(text|vision|all)$ ]] || die "--lora-scope must be text, vision or all"
IFS=',' read -r -a DEVICE_IDS <<< "${DEVICES}"
[[ ${#DEVICE_IDS[@]} -eq 8 ]] || die "this server profile requires exactly eight GPU ids"
[[ -d "${LLAMA_FACTORY_DIR}" ]] || die "LLaMA-Factory not found: ${LLAMA_FACTORY_DIR}"
[[ -f "${LLAMA_FACTORY_DIR}/src/llamafactory/model/model_utils/vlm_lora.py" ]] ||
    die "LLaMA-Factory lacks scoped VLM LoRA support: ${LLAMA_FACTORY_DIR}"
[[ -f "${KT_SOURCE_DIR}/python/sft/conv3d_compat.py" ]] || die "KT source not found: ${KT_SOURCE_DIR}"

PYTHON="${VLM_PYTHON:-/mnt/data2/wbw/conda/envs/Kllama/bin/python}"
[[ -x "${PYTHON}" ]] || die "Kllama Python not found: ${PYTHON}"
ACCELERATE="$(dirname "${PYTHON}")/accelerate"
[[ -x "${ACCELERATE}" ]] || die "accelerate not found beside ${PYTHON}"

# Apply the requested device contract to preflight as well as launch.
export CUDA_VISIBLE_DEVICES="${DEVICES}"
export VLM_KT_CONV3D_COMPAT="${KT_SOURCE_DIR}/python/sft/conv3d_compat.py"
export VLM_LORA_SCOPE="${LORA_SCOPE}"
export VLM_EXPECTED_KT_WRAPPERS="${EXPECTED_WRAPPERS}"
export PYTHONPATH="${SCRIPT_DIR}:${LLAMA_FACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

preflight=(
    "${PYTHON}" "${SCRIPT_DIR}/validate_vlm_setup.py"
    --model-path "${MODEL_PATH}"
    --dataset-dir "${DATASET_DIR}"
    --dataset-name "${DATASET_NAME}"
    --expected-layers "${EXPECTED_LAYERS}"
    --expected-experts "${EXPECTED_EXPERTS}"
    --expected-top-k "${EXPECTED_TOP_K}"
)
if [[ "${PREFLIGHT_ONLY}" -eq 0 && "${DRY_RUN}" -eq 0 ]]; then
    preflight+=(--require-cuda)
fi
"${preflight[@]}"
[[ "${PREFLIGHT_ONLY}" -eq 0 ]] || exit 0

if [[ "${LORA_SCOPE}" == "vision" && "${DRY_RUN}" -eq 0 ]]; then
    "${PYTHON}" -c 'from kt_kernel.sft.config import KTConfig; raise SystemExit(0 if "kt_freeze_experts" in KTConfig.__dataclass_fields__ else 1)' ||
        die "vision scope requires an installed kt-kernel with kt_freeze_experts support"
fi

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
    --cutoff-len "${CUTOFF_LEN}" \
    --lora-scope "${LORA_SCOPE}"

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

RESOURCE_SAMPLES="${RUN_DIR}/resource_samples.jsonl"
RESOURCE_SUMMARY="${RUN_DIR}/resource_summary.json"
RESOURCE_MONITOR_PID=""
stop_resource_monitor() {
    if [[ -n "${RESOURCE_MONITOR_PID}" ]]; then
        kill -TERM "${RESOURCE_MONITOR_PID}" 2>/dev/null || true
        wait "${RESOURCE_MONITOR_PID}" 2>/dev/null || true
        RESOURCE_MONITOR_PID=""
    fi
}
trap stop_resource_monitor EXIT INT TERM
"${PYTHON}" "${TEST_ROOT}/resource_monitor.py" \
    --output "${RESOURCE_SAMPLES}" --devices "${DEVICES}" --interval 1 &
RESOURCE_MONITOR_PID=$!

cd "${LLAMA_FACTORY_DIR}"
set +e
"${launch[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
stop_resource_monitor
"${PYTHON}" "${TEST_ROOT}/summarize_resources.py" \
    --input "${RESOURCE_SAMPLES}" --output "${RESOURCE_SUMMARY}" --require-gpu \
    2>&1 | tee "${RUN_DIR}/resource_summary.log"
[[ "${TRAIN_STATUS}" -eq 0 ]] || die "training command failed with status ${TRAIN_STATUS}"
"${PYTHON}" "${SCRIPT_DIR}/validate_adapter_output.py" --output-dir "${RUN_DIR}/model_output" \
    --lora-scope "${LORA_SCOPE}" \
    2>&1 | tee "${RUN_DIR}/adapter_validation.log"
