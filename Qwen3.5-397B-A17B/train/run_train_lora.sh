#!/usr/bin/env bash
# Train KT LoRA adapters on Qwen3.5-397B-A17B from the three Nemotron datasets.
#
# Tasks map to datasets / adapter names:
#   cuda -> nemotron_cuda -> lora-adapter/.../cuda
#   swe  -> nemotron_swe  -> lora-adapter/.../swe
#   cpp  -> nemotron_cpp  -> lora-adapter/.../cpp
#
# Usage:
#   # Prepare data (once), then train one task
#   bash run_prepare_data.sh
#   bash run_train_lora.sh cuda
#
#   # Train all three sequentially (skips missing prepared jsonl)
#   bash run_train_lora.sh all
#
#   # Smoke: tiny sample count + few steps (LLaMA-Factory key=value overrides)
#   bash run_train_lora.sh cuda max_samples=64 max_steps=20
#
#   # Skip prepare / skip convert
#   SKIP_PREPARE=1 SKIP_CONVERT=1 bash run_train_lora.sh swe
#   FORCE_PREPARE=1 bash run_train_lora.sh cuda   # rebuild jsonl even if present
#
# Env overrides (see configs/default_env.sh):
#   DEVICES, NUM_GPUS, MODEL_PATH, KT_WEIGHT_PATH, MLS_CONDA_ENV, ACCELERATE_CONFIG

set -euo pipefail

# Log / filename timestamps use China local time (script-only; host TZ unchanged).
export MLS_TIMEZONE="${MLS_TIMEZONE:-Asia/Shanghai}"
export TZ="${MLS_TIMEZONE}"

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${TRAIN_DIR}/configs/default_env.sh"

usage() {
  cat <<'EOF'
Usage: run_train_lora.sh <cuda|swe|cpp|all> [extra llamafactory CLI overrides...]

Examples:
  bash run_train_lora.sh cuda
  bash run_train_lora.sh all
  bash run_train_lora.sh swe max_samples=10000 learning_rate=5e-5
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

TASK_ARG="$1"
shift
EXTRA_OVERRIDES=("$@")

case "${TASK_ARG}" in
  -h|--help) usage; exit 0 ;;
esac

declare -A TASK_YAML=(
  [cuda]="train_lora_cuda.yaml"
  [swe]="train_lora_swe.yaml"
  [cpp]="train_lora_cpp.yaml"
)
declare -A TASK_JSONL=(
  [cuda]="nemotron_cuda.jsonl"
  [swe]="nemotron_swe.jsonl"
  [cpp]="nemotron_cpp.jsonl"
)
declare -A TASK_RUN=(
  [cuda]="nemotron_cuda"
  [swe]="nemotron_swe"
  [cpp]="nemotron_cpp"
)
declare -A TASK_ADAPTER=(
  [cuda]="cuda"
  [swe]="swe"
  [cpp]="cpp"
)

if [[ "${TASK_ARG}" == "all" ]]; then
  TASKS=(cuda swe cpp)
else
  if [[ -z "${TASK_YAML[${TASK_ARG}]+x}" ]]; then
    echo "[error] unknown task: ${TASK_ARG} (expected cuda|swe|cpp|all)" >&2
    exit 2
  fi
  TASKS=("${TASK_ARG}")
fi

# Activate conda env that has llamafactory + accelerate.
if [[ -f "${MLS_CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${MLS_CONDA_SH}"
  conda activate "${MLS_CONDA_ENV}"
fi

PYTHON_BIN="${MLS_CONDA_PYTHON}"
ACCELERATE_BIN="${MLS_CONDA_BIN}/accelerate"
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
  ACCELERATE_BIN="$(command -v accelerate || true)"
fi
if [[ -z "${ACCELERATE_BIN}" ]]; then
  echo "[error] accelerate not found in ${MLS_CONDA_BIN}; set MLS_CONDA_ENV" >&2
  exit 1
fi

# Prefer INT8 accelerate config when KT_WEIGHT_PATH is set.
if [[ -n "${KT_WEIGHT_PATH}" ]]; then
  if [[ "${ACCELERATE_CONFIG}" == *"bf16"* ]]; then
    ACCELERATE_CONFIG="${TRAIN_DIR}/configs/accelerate_fsdp2_kt_int8_8gpu.yaml"
  fi
fi

# Align num_processes with NUM_GPUS if accelerate config still says 8.
if [[ "${NUM_GPUS}" != "8" ]]; then
  ACCEL_RUNTIME="${LOG_DIR}/accelerate_runtime_${NUM_GPUS}gpu.yaml"
  mkdir -p "${LOG_DIR}"
  sed "s/^num_processes:.*/num_processes: ${NUM_GPUS}/" "${ACCELERATE_CONFIG}" > "${ACCEL_RUNTIME}"
  ACCELERATE_CONFIG="${ACCEL_RUNTIME}"
fi

mkdir -p "${RUN_ROOT}" "${ADAPTER_ROOT}" "${LOG_DIR}"

SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

prepare_one() {
  local task="$1"
  if [[ "${SKIP_PREPARE}" == "1" ]]; then
    return 0
  fi
  local jsonl="${DATASET_DIR}/${TASK_JSONL[${task}]}"
  if [[ -f "${jsonl}" && "${FORCE_PREPARE:-0}" != "1" ]]; then
    echo "[prepare] reuse existing ${jsonl}"
    return 0
  fi
  echo "[prepare] building ${task} ..."
  bash "${TRAIN_DIR}/run_prepare_data.sh" "${task}"
}

train_one() {
  local task="$1"
  local yaml="${TRAIN_DIR}/configs/${TASK_YAML[${task}]}"
  local run_dir="${RUN_ROOT}/${TASK_RUN[${task}]}"
  local jsonl="${DATASET_DIR}/${TASK_JSONL[${task}]}"
  local log_file="${LOG_DIR}/train_${task}_$(date +%Y%m%d_%H%M%S).log"

  if [[ ! -f "${jsonl}" ]]; then
    echo "[error] missing prepared data: ${jsonl}" >&2
    return 1
  fi
  if [[ ! -f "${yaml}" ]]; then
    echo "[error] missing train yaml: ${yaml}" >&2
    return 1
  fi

  if [[ "${SKIP_TRAIN}" == "1" ]]; then
    echo "[train] SKIP_TRAIN=1, skip ${task}"
    return 0
  fi

  local -a overrides=(
    "model_name_or_path=${MODEL_PATH}"
    "dataset_dir=${DATASET_DIR}"
    "output_dir=${run_dir}"
  )
  if [[ -n "${KT_WEIGHT_PATH}" ]]; then
    overrides+=("kt_weight_path=${KT_WEIGHT_PATH}")
  fi
  overrides+=("${EXTRA_OVERRIDES[@]}")

  # Text-only: load Qwen3_5MoeForCausalLM from text_config (no visual / Conv3d).
  # Same contract as FFTtest/Qwen3.5-35B-A3B/qwen35_text_only.py.
  local train_entry="${TRAIN_DIR}/scripts/train_lora_text_only.py"
  if [[ ! -f "${train_entry}" ]]; then
    echo "[error] missing text-only train entry: ${train_entry}" >&2
    return 1
  fi

  echo "[train] task=${task}"
  echo "[train] yaml=${yaml}"
  echo "[train] entry=${train_entry} (MLS_TEXT_ONLY=1)"
  echo "[train] accelerate=${ACCELERATE_CONFIG}"
  echo "[train] devices=${DEVICES}"
  echo "[train] output=${run_dir}"
  echo "[train] TMPDIR=${MLS_TMPDIR}"
  echo "[train] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
  echo "[train] log=${log_file}"

  mkdir -p "${run_dir}" \
    "${MLS_TMPDIR}" \
    "${TRITON_CACHE_DIR}/autotune" \
    "${CUDA_CACHE_PATH}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${MPLCONFIGDIR}"
  (
    cd "${LLAMA_FACTORY_DIR}"
    export TZ="${TZ:-Asia/Shanghai}"
    export MLS_TIMEZONE="${MLS_TIMEZONE:-Asia/Shanghai}"
    export MLS_TEXT_ONLY=1
    export USE_KT=1
    export ACCELERATE_USE_KT=true
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export CUDA_VISIBLE_DEVICES="${DEVICES}"
    # Keep all JIT/compile caches off the full root filesystem (FFTtest pattern).
    export TMPDIR="${MLS_TMPDIR}"
    export TMP="${MLS_TMPDIR}"
    export TEMP="${MLS_TMPDIR}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
    export CUDA_CACHE_PATH="${CUDA_CACHE_PATH}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}"
    export MPLCONFIGDIR="${MPLCONFIGDIR}"
    # scripts/ first so qwen35_text_only is importable; then LLaMA-Factory src.
    export PYTHONPATH="${TRAIN_DIR}/scripts:${LLAMA_FACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

    set -x
    "${ACCELERATE_BIN}" launch \
      --config_file "${ACCELERATE_CONFIG}" \
      "${train_entry}" \
      "${yaml}" \
      "${overrides[@]}"
  ) 2>&1 | tee "${log_file}"
}

convert_one() {
  local task="$1"
  local run_dir="${RUN_ROOT}/${TASK_RUN[${task}]}"
  local adapter_name="${TASK_ADAPTER[${task}]}"

  if [[ "${SKIP_CONVERT}" == "1" ]]; then
    echo "[convert] SKIP_CONVERT=1, skip ${task}"
    return 0
  fi
  if [[ ! -f "${run_dir}/fused_expert_lora.safetensors" ]]; then
    echo "[warn] no fused_expert_lora.safetensors in ${run_dir}; skip convert for ${task}" >&2
    return 0
  fi
  bash "${TRAIN_DIR}/scripts/convert_kt_adapter.sh" "${run_dir}" "${adapter_name}"
}

FAILED=0
for task in "${TASKS[@]}"; do
  echo "======== task: ${task} ========"
  if ! prepare_one "${task}"; then
    echo "[error] prepare failed: ${task}" >&2
    FAILED=1
    continue
  fi
  if ! train_one "${task}"; then
    echo "[error] train failed: ${task}" >&2
    FAILED=1
    continue
  fi
  if ! convert_one "${task}"; then
    echo "[error] convert failed: ${task}" >&2
    FAILED=1
    continue
  fi
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "[done] finished with errors" >&2
  exit 1
fi

echo "[done] adapters under ${ADAPTER_ROOT}"
ls -la "${ADAPTER_ROOT}" || true
