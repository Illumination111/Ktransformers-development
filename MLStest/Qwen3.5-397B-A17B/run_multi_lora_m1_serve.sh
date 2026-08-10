#!/usr/bin/env bash
# Launch sglang-kt multi-LoRA serving (M1) for Qwen3.5-397B-A17B.
# M1 contract: multiple composite adapters resident; max_loras_per_batch=1.

set -Eeuo pipefail

export TZ="${MLS_TIMEZONE:-Asia/Shanghai}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/configs/default_env.sh"

DRY_RUN=0
SKIP_READY_WAIT=0
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Launch Qwen3.5-397B-A17B multi-LoRA serving under the M1 contract:
  - multiple merged KT composite adapters via --lora-paths
  - M1 (default): --kt-lora-dispatch single, --max-loras-per-batch 1
  - M2: --kt-lora-dispatch grouped, --kt-max-loras-per-batch N (N concurrent adapters/batch)
  - --kt-num-gpu-experts 0, --disable-cuda-graph
  - dense LoRA backend: triton (397B correctness baseline)

Required:
  --kt-weight-path PATH     Verified KT CPU expert weight pack for 397B
  --lora-paths LIST         Comma-separated name=path pairs
                            Example: L0=/data/L0,L1=/data/L1

Options:
  --model-path PATH         Default: ${MODEL_PATH}
  --tokenizer-path PATH     Default: same as --model-path
  --kt-method METHOD        Default: ${KT_METHOD} (AMXBF16|AMXINT4|AMXINT8|BF16)
  --devices LIST            Physical GPU ids; TP uses first --tp-size ids
  --tp-size N               Default: ${TP_SIZE}
  --host ADDR               Default: ${HOST}
  --port N                  Default: ${PORT}
  --served-model-name NAME  Default: ${SERVED_MODEL_NAME}
  --max-loaded-loras N      Default: ${MAX_LOADED_LORAS}
  --kt-max-loaded-loras N   Default: same as --max-loaded-loras
  --max-lora-rank N         Default: ${MAX_LORA_RANK}
  --chunked-prefill-size N  Default: ${CHUNKED_PREFILL_SIZE}
  --max-running-requests N  Default: ${MAX_RUNNING_REQUESTS}
  --max-total-tokens N      Default: ${MAX_TOTAL_TOKENS}
  --context-length N        Default: ${CONTEXT_LENGTH}
  --kt-cpuinfer N           Default: ${KT_CPUINFER}
  --kt-threadpool-count N   Default: ${KT_THREADPOOL_COUNT}
  --kt-numa-nodes LIST      Space-separated; default: ${KT_NUMA_NODES}
  --attention-backend NAME  Default: ${ATTENTION_BACKEND}
  --mem-fraction-static F   Default: ${MEM_FRACTION_STATIC}
  --log-base PATH           Run log root (default: test_log under case dir)
  --conda-env NAME          Default: ${MLS_CONDA_ENV}
  --dry-run                 Print the launch command only
  --skip-ready-wait         Do not wait for /health (used by e2e wrapper)
  -h, --help                Show this help

Environment overrides: MODEL_PATH, KT_WEIGHT_PATH, LORA_PATHS, PORT, etc.
See: ${MLS_ROOT}/docs/task_bash_Qwen3.5-397B-A17B.md
EOF
}

need_value() {
    local flag="$1" count="$2"
    (( count >= 2 )) || {
        printf 'Missing value for %s\n' "${flag}" >&2
        exit 2
    }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) need_value "$1" "$#"; MODEL_PATH="$2"; TOKENIZER_PATH="${TOKENIZER_PATH:-$2}"; shift ;;
        --tokenizer-path) need_value "$1" "$#"; TOKENIZER_PATH="$2"; shift ;;
        --kt-weight-path) need_value "$1" "$#"; KT_WEIGHT_PATH="$2"; shift ;;
        --kt-method) need_value "$1" "$#"; KT_METHOD="$2"; shift ;;
        --lora-paths) need_value "$1" "$#"; LORA_PATHS="$2"; shift ;;
        --devices) need_value "$1" "$#"; DEVICES="$2"; shift ;;
        --tp-size) need_value "$1" "$#"; TP_SIZE="$2"; shift ;;
        --host) need_value "$1" "$#"; HOST="$2"; shift ;;
        --port) need_value "$1" "$#"; PORT="$2"; shift ;;
        --served-model-name) need_value "$1" "$#"; SERVED_MODEL_NAME="$2"; shift ;;
        --max-loaded-loras) need_value "$1" "$#"; MAX_LOADED_LORAS="$2"; KT_MAX_LOADED_LORAS="${KT_MAX_LOADED_LORAS:-$2}"; shift ;;
        --kt-max-loaded-loras) need_value "$1" "$#"; KT_MAX_LOADED_LORAS="$2"; shift ;;
        --kt-lora-dispatch) need_value "$1" "$#"; KT_LORA_DISPATCH="$2"; shift ;;
        --kt-max-loras-per-batch) need_value "$1" "$#"; KT_MAX_LORAS_PER_BATCH="$2"; shift ;;
        --max-loras-per-batch) need_value "$1" "$#"; MAX_LORAS_PER_BATCH="$2"; shift ;;
        --max-lora-rank) need_value "$1" "$#"; MAX_LORA_RANK="$2"; shift ;;
        --chunked-prefill-size) need_value "$1" "$#"; CHUNKED_PREFILL_SIZE="$2"; shift ;;
        --max-running-requests) need_value "$1" "$#"; MAX_RUNNING_REQUESTS="$2"; shift ;;
        --max-total-tokens) need_value "$1" "$#"; MAX_TOTAL_TOKENS="$2"; shift ;;
        --context-length) need_value "$1" "$#"; CONTEXT_LENGTH="$2"; shift ;;
        --kt-cpuinfer) need_value "$1" "$#"; KT_CPUINFER="$2"; shift ;;
        --kt-threadpool-count) need_value "$1" "$#"; KT_THREADPOOL_COUNT="$2"; shift ;;
        --kt-numa-nodes) need_value "$1" "$#"; KT_NUMA_NODES="$2"; shift ;;
        --attention-backend) need_value "$1" "$#"; ATTENTION_BACKEND="$2"; shift ;;
        --mem-fraction-static) need_value "$1" "$#"; MEM_FRACTION_STATIC="$2"; shift ;;
        --log-base) need_value "$1" "$#"; LOG_BASE="$2"; shift ;;
        --conda-env) need_value "$1" "$#"; MLS_CONDA_ENV="$2"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        --skip-ready-wait) SKIP_READY_WAIT=1 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

# Reconcile M1/M2 batch limits after CLI overrides.
KT_LORA_DISPATCH="${KT_LORA_DISPATCH:-single}"
if [[ "${KT_LORA_DISPATCH}" == "grouped" ]]; then
    MAX_LORAS_PER_BATCH="${MAX_LORAS_PER_BATCH:-4}"
    KT_MAX_LORAS_PER_BATCH="${KT_MAX_LORAS_PER_BATCH:-${MAX_LORAS_PER_BATCH}}"
    if (( MAX_LORAS_PER_BATCH < KT_MAX_LORAS_PER_BATCH )); then
        printf 'ERROR: --max-loras-per-batch (%s) < --kt-max-loras-per-batch (%s)\n' \
            "${MAX_LORAS_PER_BATCH}" "${KT_MAX_LORAS_PER_BATCH}" >&2
        exit 2
    fi
else
    MAX_LORAS_PER_BATCH=1
    KT_MAX_LORAS_PER_BATCH=1
fi

if [[ -z "${KT_WEIGHT_PATH}" ]]; then
    printf 'ERROR: --kt-weight-path is required (verified KT expert pack for 397B).\n' >&2
    exit 2
fi
if [[ -z "${LORA_PATHS}" ]]; then
    printf 'ERROR: --lora-paths is required (e.g. L0=/path/L0,L1=/path/L1).\n' >&2
    exit 2
fi
if (( ! DRY_RUN )); then
    if [[ ! -d "${MODEL_PATH}" ]]; then
        printf 'ERROR: model path not found: %s\n' "${MODEL_PATH}" >&2
        exit 2
    fi
    if [[ ! -d "${KT_WEIGHT_PATH}" ]]; then
        printf 'ERROR: kt-weight-path not found: %s\n' "${KT_WEIGHT_PATH}" >&2
        exit 2
    fi
fi

# Parse LORA_PATHS=name=path,name=path into array for --lora-paths args
IFS=',' read -r -a _LORA_PAIRS <<< "${LORA_PATHS}"
LORA_ARGS=()
ADAPTER_COUNT=0
for pair in "${_LORA_PAIRS[@]}"; do
    pair="${pair#"${pair%%[![:space:]]*}"}"
    pair="${pair%"${pair##*[![:space:]]}"}"
    [[ -n "${pair}" ]] || continue
    if [[ "${pair}" != *"="* ]]; then
        printf 'ERROR: invalid --lora-paths entry %q (expected name=path)\n' "${pair}" >&2
        exit 2
    fi
    name="${pair%%=*}"
    path="${pair#*=}"
    if (( ! DRY_RUN )); then
        if [[ ! -d "${path}" ]]; then
            printf 'ERROR: adapter dir not found for %s: %s\n' "${name}" "${path}" >&2
            exit 2
        fi
        if [[ ! -f "${path}/adapter_model.safetensors" ]]; then
            printf 'ERROR: %s missing adapter_model.safetensors under %s\n' "${name}" "${path}" >&2
            exit 2
        fi
        if [[ ! -f "${path}/adapter_config.json" ]]; then
            printf 'ERROR: %s missing adapter_config.json under %s\n' "${name}" "${path}" >&2
            exit 2
        fi
    fi
    LORA_ARGS+=("${name}=${path}")
    ADAPTER_COUNT=$((ADAPTER_COUNT + 1))
done
if (( ADAPTER_COUNT < 1 )); then
    printf 'ERROR: need at least one adapter in --lora-paths\n' >&2
    exit 2
fi
if (( ADAPTER_COUNT > MAX_LOADED_LORAS )); then
    printf 'ERROR: adapter count (%d) > --max-loaded-loras (%s)\n' \
        "${ADAPTER_COUNT}" "${MAX_LOADED_LORAS}" >&2
    exit 2
fi
if (( MAX_LORAS_PER_BATCH != 1 )); then
    printf 'WARN: M1 forces max_loras_per_batch=1 (was %s)\n' "${MAX_LORAS_PER_BATCH}" >&2
    MAX_LORAS_PER_BATCH=1
fi
KT_MAX_LOADED_LORAS="${KT_MAX_LOADED_LORAS:-${MAX_LOADED_LORAS}}"
if (( KT_MAX_LOADED_LORAS < ADAPTER_COUNT )); then
    printf 'ERROR: --kt-max-loaded-loras (%s) < adapter count (%d)\n' \
        "${KT_MAX_LOADED_LORAS}" "${ADAPTER_COUNT}" >&2
    exit 2
fi

# Resolve CUDA_VISIBLE_DEVICES from --devices (first TP_SIZE ids)
IFS=',' read -r -a _DEV_ARR <<< "${DEVICES}"
if (( ${#_DEV_ARR[@]} < TP_SIZE )); then
    printf 'ERROR: need at least %s devices in --devices, got %s\n' \
        "${TP_SIZE}" "${DEVICES}" >&2
    exit 2
fi
VISIBLE=()
for ((i = 0; i < TP_SIZE; i++)); do
    VISIBLE+=("${_DEV_ARR[$i]}")
done
CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${VISIBLE[*]}")"
export CUDA_VISIBLE_DEVICES

RUN_ID="$(date +%Y%m%d_%H%M%S)_m1_serve_tp${TP_SIZE}_n${ADAPTER_COUNT}"
RUN_DIR="${LOG_BASE}/${RUN_ID}"
mkdir -p "${RUN_DIR}" "${SGLANG_KT_LORA_CACHE_DIR}" "${TMPDIR}"
export SGLANG_KT_LORA_CACHE_DIR TMPDIR
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
# sglang's startup check_server_args() hard-fails on torch 2.9.1 + cudnn<9.15 due to
# an nn.Conv3d perf bug; MLS text-only MoE serving never builds Conv3d, so bypass.
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"

# Prefer case-local fork on PYTHONPATH
export PYTHONPATH="${KT_KERNEL_PYTHON}:${SGLANG_KT_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -x "${MLS_CONDA_PYTHON}" && "${MLS_CONDA_ENV}" == "kt-kernel" ]]; then
    PYTHON_BIN="${MLS_CONDA_PYTHON}"
elif [[ -f "${MLS_CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${MLS_CONDA_SH}"
    conda activate "${MLS_CONDA_ENV}"
    PYTHON_BIN="$(command -v python)"
else
    PYTHON_BIN="$(command -v python3)"
fi

# shellcheck disable=SC2206
NUMA_ARR=(${KT_NUMA_NODES})

CMD=(
    "${PYTHON_BIN}" -m sglang.launch_server
    --host "${HOST}"
    --port "${PORT}"
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --kt-weight-path "${KT_WEIGHT_PATH}"
    --kt-method "${KT_METHOD}"
    --kt-cpuinfer "${KT_CPUINFER}"
    --kt-threadpool-count "${KT_THREADPOOL_COUNT}"
    --kt-numa-nodes "${NUMA_ARR[@]}"
    --kt-num-gpu-experts "${KT_NUM_GPU_EXPERTS}"
    --tensor-parallel-size "${TP_SIZE}"
    --dtype bfloat16
    --trust-remote-code
    --served-model-name "${SERVED_MODEL_NAME}"
    --attention-backend "${ATTENTION_BACKEND}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --context-length "${CONTEXT_LENGTH}"
    --disable-cuda-graph
    --disable-custom-all-reduce
    --enable-lora
    --lora-backend "${LORA_BACKEND}"
    --max-lora-rank "${MAX_LORA_RANK}"
    --max-loaded-loras "${MAX_LOADED_LORAS}"
    --max-loras-per-batch "${MAX_LORAS_PER_BATCH}"
    --kt-max-loaded-loras "${KT_MAX_LOADED_LORAS}"
    --kt-max-loras-per-batch "${KT_MAX_LORAS_PER_BATCH}"
    --kt-lora-dispatch "${KT_LORA_DISPATCH}"
    --lora-paths "${LORA_ARGS[@]}"
    --log-level info
)
if ((${#EXTRA_ARGS[@]})); then
    CMD+=("${EXTRA_ARGS[@]}")
fi

cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "milestone": "${KT_LORA_DISPATCH}",
  "model_path": "${MODEL_PATH}",
  "kt_weight_path": "${KT_WEIGHT_PATH}",
  "kt_method": "${KT_METHOD}",
  "lora_paths": "${LORA_PATHS}",
  "adapter_count": ${ADAPTER_COUNT},
  "max_loaded_loras": ${MAX_LOADED_LORAS},
  "max_loras_per_batch": ${MAX_LORAS_PER_BATCH},
  "kt_max_loaded_loras": ${KT_MAX_LOADED_LORAS},
  "kt_max_loras_per_batch": ${KT_MAX_LORAS_PER_BATCH},
  "kt_lora_dispatch": "${KT_LORA_DISPATCH}",
  "tp_size": ${TP_SIZE},
  "devices": "${CUDA_VISIBLE_DEVICES}",
  "host": "${HOST}",
  "port": ${PORT},
  "served_model_name": "${SERVED_MODEL_NAME}",
  "chunked_prefill_size": ${CHUNKED_PREFILL_SIZE},
  "conda_env": "${MLS_CONDA_ENV}",
  "python": "${PYTHON_BIN}",
  "pythonpath": "${PYTHONPATH}"
}
EOF

printf 'Writing launch command to %s/launch_cmd.txt\n' "${RUN_DIR}"
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_cmd.txt"
printf '\n' >> "${RUN_DIR}/launch_cmd.txt"

if (( DRY_RUN )); then
    printf '[dry-run] would exec:\n'
    cat "${RUN_DIR}/launch_cmd.txt"
    exit 0
fi

printf 'Starting M1 multi-LoRA server; logs -> %s/server.log\n' "${RUN_DIR}"
# shellcheck disable=SC2094
"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/server.log"
