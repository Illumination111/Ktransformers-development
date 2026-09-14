#!/usr/bin/env bash
# GLM-4.5-Air PyTorch TORCH full-FT using the accepted sap4 stack.
# The xmy venv is used read-only; vendor trees are injected only via PYTHONPATH.

set -Eeuo pipefail

export TZ="${FFT_TIMEZONE:-Asia/Shanghai}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${FFT_ROOT}/pytorch_torch/env.sh"

CONFIGS_DIR="${SCRIPT_DIR}/configs"
LOG_BASE="${FFT_LOG_BASE:-${SCRIPT_DIR}/test_log}"
MODEL_PATH="${FFT_MODEL_PATH:-/mnt/data2/models/GLM-4.5-Air}"
DATASET_DIR="${FFT_DATASET_DIR:-${FFT_ROOT}/dataset}"
DATASET_NAME="${FFT_DATASET_NAME:-fft_real_100}"
SHARED_FLOW_DIR="${FFT_SHARED_FLOW_DIR:-${FFT_ROOT}/Qwen3.5-35B-A3B}"
MIN_AVAILABLE_GIB="${FFT_TORCH_MIN_AVAILABLE_GIB:-1200}"
MAIN_PORT="${FFT_TORCH_MAIN_PORT:-29851}"

TRAIN_ENTRY_MODULE="finetune_train_with_timing"
TRAIN_CONFIG_BASE="${CONFIGS_DIR}/train_full_bf16_glm45_air.yaml"
ACCEL_TEMPLATE="${CONFIGS_DIR}/accelerate_pytorch_torch_bf16_8gpu.yaml"
VALIDATOR="${SCRIPT_DIR}/validate_benchmark_dataset.py"
AGGREGATOR="${SCRIPT_DIR}/aggregate_sweep_results.py"
TIMING_VALIDATOR="${SHARED_FLOW_DIR}/validate_step_timing.py"
RESOURCE_EXEC="${SHARED_FLOW_DIR}/resource_scope_exec.py"
MONITOR_SCRIPT="${SHARED_FLOW_DIR}/monitor.py"
MEMORY_ANALYZER="${SHARED_FLOW_DIR}/analyze_memory_usage.py"

readonly -a SERVER_SEQUENCE_LENGTHS=(32 64 128 256 512 1024 2048 4096)

SEQUENCE_LENGTHS_CSV=""
SEQUENCE_LENGTHS_OVERRIDE_SET=0
SEQUENCE_LENGTH=""
SEQUENCE_LENGTH_SET=0
STEPS=15
WARMUP_STEPS=5
GRAD_ACCUM_STEPS=1
LEARNING_RATE="1.0e-5"
DEVICES_OVERRIDE=""
CPU_THREADS_OVERRIDE="${FFT_CPU_THREADS:-64}"
DRY_RUN=0
CONTINUE_ON_ERROR=0
KEEP_MODEL_OUTPUT=0
SKIP_DATASET_CHECK=0

RUN_ROOT=""
SUMMARY_FINALIZED=0
ACTIVE_MONITOR_PID=""
ACTIVE_MONITOR_FIFO=""
ACTIVE_TRAIN_PID=""
ACTIVE_TEE_PID=""
ACTIVE_LOG_FIFO=""

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Uses /mnt/data2/xmy/venv read-only plus vendor PYTHONPATH from
torch-moe-qwen-current. Does not install or pin packages in xmy.

  backend                 PyTorch TORCH owner-sharded CPU experts
  server                  8 GPUs, global batch 8
  finetuning type         full

Options:
  --seq-lengths LIST      Comma-separated sequence lengths
  --seq-length N          Run exactly one sequence length
  --steps N               Optimizer steps per sequence (default: 15)
  --warmup-steps N        Steps excluded from stable TPS (default: 5)
  --gas N                 Gradient accumulation steps (default: 1)
  --learning-rate VALUE   Default: 1.0e-5
  --cpu-threads N         OMP/MKL threads on every rank (default: 64)
  --devices LIST          Physical GPU ids; uses the first 8
  --model-path PATH       Default: /mnt/data2/models/GLM-4.5-Air
  --dataset-dir PATH      LLaMA-Factory dataset directory
  --dataset-name NAME     Registered dataset name (default: fft_real_100)
  --log-base PATH         Result directory base
  --continue-on-error     Continue after a failed sequence
  --keep-model-output     Keep generated final model output
  --skip-dataset-check    Skip model/tokenizer/dataset length validation
  --dry-run               Generate configs and print commands only
  -h, --help              Show this help
EOF
}

need_value() {
    local flag="$1" count="$2"
    (( count >= 2 )) || { printf 'Missing value for %s\n' "${flag}" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seq-lengths) need_value "$1" "$#"; SEQUENCE_LENGTHS_CSV="$2"; SEQUENCE_LENGTHS_OVERRIDE_SET=1; shift ;;
        --seq-length) need_value "$1" "$#"; SEQUENCE_LENGTH="$2"; SEQUENCE_LENGTH_SET=1; shift ;;
        --steps) need_value "$1" "$#"; STEPS="$2"; shift ;;
        --warmup-steps) need_value "$1" "$#"; WARMUP_STEPS="$2"; shift ;;
        --gas) need_value "$1" "$#"; GRAD_ACCUM_STEPS="$2"; shift ;;
        --learning-rate) need_value "$1" "$#"; LEARNING_RATE="$2"; shift ;;
        --cpu-threads) need_value "$1" "$#"; CPU_THREADS_OVERRIDE="$2"; shift ;;
        --devices) need_value "$1" "$#"; DEVICES_OVERRIDE="$2"; shift ;;
        --model-path) need_value "$1" "$#"; MODEL_PATH="$2"; shift ;;
        --dataset-dir) need_value "$1" "$#"; DATASET_DIR="$2"; shift ;;
        --dataset-name) need_value "$1" "$#"; DATASET_NAME="$2"; shift ;;
        --log-base) need_value "$1" "$#"; LOG_BASE="$2"; shift ;;
        --continue-on-error) CONTINUE_ON_ERROR=1 ;;
        --keep-model-output) KEEP_MODEL_OUTPUT=1 ;;
        --skip-dataset-check) SKIP_DATASET_CHECK=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
die() { printf '[%s] ERROR: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; exit 1; }

require_positive_int() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer, got ${value}"
}
require_nonnegative_int() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got ${value}"
}

require_positive_int "--steps" "${STEPS}"
require_nonnegative_int "--warmup-steps" "${WARMUP_STEPS}"
require_positive_int "--gas" "${GRAD_ACCUM_STEPS}"
require_positive_int "--cpu-threads" "${CPU_THREADS_OVERRIDE}"
(( WARMUP_STEPS < STEPS )) || die "--warmup-steps must be smaller than --steps"

if [[ "${SEQUENCE_LENGTH_SET}" -eq 1 ]]; then
    [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 0 ]] || die "--seq-length and --seq-lengths are mutually exclusive"
    require_positive_int "--seq-length" "${SEQUENCE_LENGTH}"
    SEQUENCE_LENGTHS_CSV="${SEQUENCE_LENGTH}"
    SEQUENCE_LENGTHS_OVERRIDE_SET=1
fi

declare -a SEQUENCE_LENGTHS_OVERRIDE=()
if [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 1 ]]; then
    IFS=',' read -r -a SEQUENCE_LENGTHS_OVERRIDE <<< "${SEQUENCE_LENGTHS_CSV// /}"
    (( ${#SEQUENCE_LENGTHS_OVERRIDE[@]} > 0 )) || die "sequence length list is empty"
fi
MAX_SEQUENCE_LENGTH=32
if [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 1 ]]; then
    MAX_SEQUENCE_LENGTH=0
    for seq in "${SEQUENCE_LENGTHS_OVERRIDE[@]}"; do
        require_positive_int "sequence length" "${seq}"
        (( seq > MAX_SEQUENCE_LENGTH )) && MAX_SEQUENCE_LENGTH="${seq}"
    done
fi

PYTHON="$(fft_torch_python)"
[[ -x "${PYTHON}" ]] || die "xmy python not found: ${PYTHON}"
MONITOR_PYTHON="/mnt/data2/wbw/conda/envs/Deepspeed/bin/python3"
[[ -x "${MONITOR_PYTHON}" ]] || die "Deepspeed python is required for memory monitoring"

NUM_GPUS=8
GLOBAL_BATCH_SIZE=8
PER_DEVICE_BATCH_SIZE=1
CPU_THREADS_PER_RANK="${CPU_THREADS_OVERRIDE}"

set_yaml_value() {
    local file="$1" key="$2" value="$3"
    if grep -q "^${key}:" "${file}"; then
        sed -i "s|^${key}:.*|${key}: ${value}|" "${file}"
    else
        printf '%s: %s\n' "${key}" "${value}" >> "${file}"
    fi
}

delete_yaml_key() {
    local file="$1" key="$2"
    sed -i "/^${key}:/d" "${file}"
}

check_files_and_environment() {
    [[ -d "${MODEL_PATH}" ]] || die "model directory not found: ${MODEL_PATH}"
    [[ -d "${DATASET_DIR}" ]] || die "dataset directory not found: ${DATASET_DIR}"
    [[ -d "${FFT_TORCH_MOE_ROOT}" ]] || die "vendor root not found: ${FFT_TORCH_MOE_ROOT}"
    [[ -d "${FFT_TORCH_MOE_V56_DEPS}" ]] || die "transformers 5.6 deps not found: ${FFT_TORCH_MOE_V56_DEPS}"
    local required
    for required in \
        "${TRAIN_CONFIG_BASE}" "${ACCEL_TEMPLATE}" "${VALIDATOR}" \
        "${AGGREGATOR}" "${TIMING_VALIDATOR}" "${RESOURCE_EXEC}" \
        "${MONITOR_SCRIPT}" "${MEMORY_ANALYZER}" \
        "${FFT_ROOT}/pytorch_torch/install_torch_moe_full_ft.py" \
        "${FFT_TORCH_MOE_TRANSFORMERS}/transformers/__init__.py" \
        "${FFT_TORCH_MOE_ACCELERATE}/accelerate/utils/torch_moe.py" \
        "${FFT_TORCH_MOE_LLAMAFACTORY}/llamafactory/__init__.py"; do
        [[ -f "${required}" ]] || die "required file not found: ${required}"
    done
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
    command -v systemd-run >/dev/null || die "systemd-run is required"
    command -v numactl >/dev/null || die "numactl is required"
    env MPLCONFIGDIR=/tmp/fft_glm45_air_matplotlib \
        "${MONITOR_PYTHON}" -c 'import matplotlib, psutil, pynvml' || \
        die "memory monitoring dependencies are unavailable"
    FFT_VENDOR_PYTHONPATH="$(fft_torch_pythonpath "${SCRIPT_DIR}:${SHARED_FLOW_DIR}:${FFT_ROOT}/pytorch_torch")"
    PYTHONPATH="${FFT_VENDOR_PYTHONPATH}" "${PYTHON}" - <<'PY'
import inspect
import accelerate
import huggingface_hub
import torch
import transformers

if "torch-moe-qwen-current" not in inspect.getfile(accelerate):
    raise SystemExit(f"accelerate is not the vendor copy: {inspect.getfile(accelerate)}")
if "transformers-v56" not in inspect.getfile(transformers):
    raise SystemExit(f"transformers is not vendor v56: {inspect.getfile(transformers)}")
from accelerate.utils import torch_moe
if "torch-moe-qwen-current" not in inspect.getfile(torch_moe):
    raise SystemExit(f"torch_moe is not the vendor copy: {inspect.getfile(torch_moe)}")
print({
    "python": __import__("sys").executable,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "transformers_import": inspect.getfile(transformers),
    "accelerate_import": inspect.getfile(accelerate),
    "torch_moe_import": inspect.getfile(torch_moe),
})
PY
}

resolve_devices() {
    local source="${DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-}}"
    [[ -n "${source}" ]] || source="0,1,2,3,4,5,6,7"
    source="${source// /}"
    local -a candidates
    IFS=',' read -r -a candidates <<< "${source}"
    (( ${#candidates[@]} >= NUM_GPUS )) || die "GPU list '${source}' has fewer than ${NUM_GPUS} entries"
    local -a selected=("${candidates[@]:0:NUM_GPUS}")
    local device
    declare -A seen=()
    for device in "${selected[@]}"; do
        [[ "${device}" =~ ^[0-9]+$ ]] || die "invalid GPU id: ${device}"
        [[ -z "${seen[${device}]:-}" ]] || die "duplicate GPU id: ${device}"
        seen["${device}"]=1
    done
    (IFS=','; printf '%s' "${selected[*]}")
}

check_cluster() {
    [[ "${DRY_RUN}" -eq 1 ]] && return
    local actual
    actual="$(nvidia-smi -L | wc -l)"
    (( actual >= NUM_GPUS )) || die "requested ${NUM_GPUS} GPUs but only ${actual} were detected"
    local devices gpu
    devices="$(resolve_devices)"
    IFS=',' read -r -a gpu_list <<< "${devices}"
    for gpu in "${gpu_list[@]}"; do
        if nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
            die "GPU ${gpu} is occupied; refusing to start pytorch_torch"
        fi
    done
    local available_gib
    available_gib="$(awk '/^MemAvailable:/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)"
    (( available_gib >= MIN_AVAILABLE_GIB )) || \
        die "MemAvailable=${available_gib}GiB, need ${MIN_AVAILABLE_GIB}GiB"
}

make_train_config() {
    local run_dir="$1" seq="$2"
    local config="${run_dir}/train_config.yaml"
    cp "${TRAIN_CONFIG_BASE}" "${config}"
    set_yaml_value "${config}" model_name_or_path "${MODEL_PATH}"
    set_yaml_value "${config}" dataset "${DATASET_NAME}"
    set_yaml_value "${config}" dataset_dir "${DATASET_DIR}"
    set_yaml_value "${config}" template "glm4_moe"
    set_yaml_value "${config}" cutoff_len "${seq}"
    set_yaml_value "${config}" output_dir "${run_dir}/model_output"
    set_yaml_value "${config}" per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
    set_yaml_value "${config}" gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    set_yaml_value "${config}" learning_rate "${LEARNING_RATE}"
    set_yaml_value "${config}" max_steps "${STEPS}"
    set_yaml_value "${config}" finetuning_type "full"
    set_yaml_value "${config}" bf16 "true"
    set_yaml_value "${config}" fp16 "false"
    set_yaml_value "${config}" tf32 "false"
    set_yaml_value "${config}" pure_bf16 "true"
    set_yaml_value "${config}" gradient_checkpointing "false"
    set_yaml_value "${config}" disable_gradient_checkpointing "true"
    set_yaml_value "${config}" use_kt "true"
    set_yaml_value "${config}" kt_backend "TORCH"
    set_yaml_value "${config}" kt_expert_checkpoint_path "${MODEL_PATH}"
    set_yaml_value "${config}" kt_num_threads "${CPU_THREADS_PER_RANK}"
    set_yaml_value "${config}" kt_tp_enabled "false"
    set_yaml_value "${config}" kt_threadpool_count "1"
    set_yaml_value "${config}" kt_num_gpu_experts "0"
    set_yaml_value "${config}" kt_use_lora_experts "false"
    delete_yaml_key "${config}" kt_weight_path
    delete_yaml_key "${config}" gradient_checkpointing_kwargs
    printf '%s\n' "${config}"
}

make_accel_config() {
    local run_dir="$1"
    local config="${run_dir}/accelerate_config.yaml"
    cp "${ACCEL_TEMPLATE}" "${config}"
    sed -i "s|^  kt_num_threads:.*|  kt_num_threads: ${CPU_THREADS_PER_RANK}|" "${config}" || true
    if ! grep -q "^  kt_num_threads:" "${config}"; then
        printf '  kt_num_threads: %s\n' "${CPU_THREADS_PER_RANK}" >> "${config}"
    fi
    printf '%s\n' "${config}"
}

write_run_config() {
    local path="$1" seq="$2" devices="$3" tokens_per_step="$4"
    PYTHONPATH="${FFT_VENDOR_PYTHONPATH}" "${PYTHON}" - \
        "${path}" "${seq}" "${devices}" "${tokens_per_step}" \
        "${STEPS}" "${WARMUP_STEPS}" "${GRAD_ACCUM_STEPS}" \
        "${LEARNING_RATE}" "${MODEL_PATH}" "${DATASET_NAME}" \
        "${CPU_THREADS_PER_RANK}" "${DRY_RUN}" <<'PY'
import json, sys
from pathlib import Path
out, seq, devices, tokens, steps, warmup, gas, lr, model, dataset, threads, dry = sys.argv[1:]
Path(out).write_text(json.dumps({
    "backend": "pytorch_torch",
    "profile": "server",
    "benchmark_class": "exact_model_full_finetune",
    "precision": "bf16",
    "finetuning_type": "full",
    "kt_backend": "TORCH",
    "python": "/mnt/data2/xmy/venv/bin/python",
    "venv_mutation": False,
    "vendor_root": "/mnt/data3/qujing/torch-moe-qwen-current",
    "sequence_length": int(seq),
    "num_gpus": 8,
    "global_batch_size": 8,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": int(gas),
    "tokens_per_step": int(tokens),
    "steps": int(steps),
    "warmup_steps": int(warmup),
    "learning_rate": lr,
    "devices": devices,
    "model_path": model,
    "dataset_name": dataset,
    "cpu_threads_per_rank": int(threads),
    "dry_run": bool(int(dry)),
    "timing_mode": "coarse_host_wall_no_cuda_sync",
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

stop_active_monitor() {
    if [[ -n "${ACTIVE_MONITOR_PID}" ]] && kill -0 "${ACTIVE_MONITOR_PID}" 2>/dev/null; then
        kill -TERM "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
        wait "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    fi
    ACTIVE_MONITOR_PID=""
    [[ -n "${ACTIVE_MONITOR_FIFO}" ]] && rm -f "${ACTIVE_MONITOR_FIFO}"
    ACTIVE_MONITOR_FIFO=""
}

stop_active_training() {
    if [[ -n "${ACTIVE_TRAIN_PID}" ]] && kill -0 "${ACTIVE_TRAIN_PID}" 2>/dev/null; then
        kill -TERM "${ACTIVE_TRAIN_PID}" 2>/dev/null || true
        wait "${ACTIVE_TRAIN_PID}" 2>/dev/null || true
    fi
    ACTIVE_TRAIN_PID=""
    if [[ -n "${ACTIVE_TEE_PID}" ]] && kill -0 "${ACTIVE_TEE_PID}" 2>/dev/null; then
        kill -TERM "${ACTIVE_TEE_PID}" 2>/dev/null || true
        wait "${ACTIVE_TEE_PID}" 2>/dev/null || true
    fi
    ACTIVE_TEE_PID=""
    [[ -n "${ACTIVE_LOG_FIFO}" ]] && rm -f "${ACTIVE_LOG_FIFO}"
    ACTIVE_LOG_FIFO=""
}

generate_sweep_summary() {
    [[ -n "${RUN_ROOT}" && -d "${RUN_ROOT}" ]] || return 0
    [[ "${SUMMARY_FINALIZED}" -eq 0 ]] || return 0
    compgen -G "${RUN_ROOT}/*/seq_*/run_config.json" >/dev/null || return 0
    SUMMARY_FINALIZED=1
    if PYTHONPATH="${FFT_VENDOR_PYTHONPATH}" "${PYTHON}" "${AGGREGATOR}" --root "${RUN_ROOT}"; then
        log "Sweep summary: ${RUN_ROOT}/summary.md"
        return 0
    fi
    warn "Sweep aggregation failed for ${RUN_ROOT}"
    return 98
}

finalize_sweep_on_exit() {
    local original_status="${1:-0}" summary_status=0
    trap - EXIT
    set +e
    stop_active_training
    stop_active_monitor
    generate_sweep_summary
    summary_status=$?
    if [[ "${original_status}" -eq 0 && "${summary_status}" -ne 0 ]]; then
        original_status="${summary_status}"
    fi
    exit "${original_status}"
}

start_memory_monitor() {
    local run_dir="$1"
    mkdir -p "${run_dir}/.mplconfig"
    ACTIVE_MONITOR_FIFO="${run_dir}/monitor_events.fifo"
    rm -f "${ACTIVE_MONITOR_FIFO}"
    env MPLCONFIGDIR="${run_dir}/.mplconfig" \
        "${MONITOR_PYTHON}" "${MONITOR_SCRIPT}" \
        --out "${run_dir}/monitor.csv" \
        --fifo "${ACTIVE_MONITOR_FIFO}" \
        --interval 1 \
        --disk-mount /mnt/data2 \
        --pid "$$" \
        --resource-contract "${run_dir}/resource_contract.json" \
        >> "${run_dir}/monitor.log" 2>&1 &
    ACTIVE_MONITOR_PID=$!
    local attempt
    for attempt in {1..30}; do
        [[ -f "${run_dir}/monitor.csv" ]] && break
        kill -0 "${ACTIVE_MONITOR_PID}" 2>/dev/null || die "memory monitor failed; see ${run_dir}/monitor.log"
        sleep 0.1
    done
    [[ -f "${run_dir}/monitor.csv" ]] || die "memory monitor did not create monitor.csv"
}

analyze_memory_usage() {
    local run_dir="$1"
    env MPLCONFIGDIR="${run_dir}/.mplconfig" \
        "${MONITOR_PYTHON}" "${MEMORY_ANALYZER}" \
        --log-dir "${run_dir}" \
        --require-cgroup-memory \
        >> "${run_dir}/memory_analysis.log" 2>&1 || \
        warn "Memory analysis failed; see ${run_dir}/memory_analysis.log"
}

run_one_sequence() {
    local seq="$1" devices="$2"
    local profile_dir="${RUN_ROOT}/server_${NUM_GPUS}gpu_batch${GLOBAL_BATCH_SIZE}"
    local run_dir="${profile_dir}/seq_${seq}"
    local timing_dir="${run_dir}/step_timing"
    local train_log="${run_dir}/train.log"
    local case_unit="fft-glm45-air-pytorch-torch-seq${seq}-$$"
    local tokens_per_step=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE * seq * GRAD_ACCUM_STEPS))
    mkdir -p "${run_dir}"
    local train_config accel_config
    train_config="$(make_train_config "${run_dir}" "${seq}")"
    accel_config="$(make_accel_config "${run_dir}")"
    write_run_config "${run_dir}/run_config.json" "${seq}" "${devices}" "${tokens_per_step}"

    local -a command=(
        env
        PYTHONPATH="${FFT_VENDOR_PYTHONPATH}"
        USE_KT=1
        ACCELERATE_USE_KT=true
        ACCELERATE_KT_BACKEND=TORCH
        ACCELERATE_KT_TRAIN_MODE=full
        ACCELERATE_KT_LORA_RANK=0
        ACCELERATE_KT_LORA_ALPHA=0
        ACCELERATE_KT_EXPERT_CHECKPOINT_PATH="${MODEL_PATH}"
        ACCELERATE_KT_SKIP_EXPERT_LOADING=true
        ACCELERATE_KT_TP_ENABLED=false
        ACCELERATE_KT_NUM_GPU_EXPERTS=0
        ACCELERATE_KT_USE_LORA_EXPERTS=false
        ACCELERATE_KT_MODEL_MAX_LENGTH="${seq}"
        ACCELERATE_KT_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        KT_FINETUNE_MODE=full
        FFT_TRAINING_BACKEND=pytorch_torch
        FFT_PRECISION=bf16
        FFT_FINETUNING_TYPE=full
        FFT_SKIP_FINAL_SAVE="$((1 - KEEP_MODEL_OUTPUT))"
        FFT_STEP_TIMING_OUT_DIR="${timing_dir}"
        FFT_STEP_TIMING_WARMUP_STEPS="${WARMUP_STEPS}"
        FFT_STEP_TIMING_TOKENS_PER_STEP="${tokens_per_step}"
        FFT_DISABLE_PERF_PROBES=1
        FFT_CPU_THREADS="${CPU_THREADS_PER_RANK}"
        TORCH_MOE_BACKWARD_GRAD_SCALE=1
        TORCH_MOE_BACKWARD_DTYPE=bfloat16
        DISABLE_VERSION_CHECK=1
        HF_HUB_OFFLINE=1
        HF_DATASETS_OFFLINE=1
        TRANSFORMERS_OFFLINE=1
        TOKENIZERS_PARALLELISM=false
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        TORCH_NCCL_ENABLE_MONITORING=0
        TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
        NCCL_DEBUG=WARN
        MALLOC_ARENA_MAX=4
        OMP_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        MKL_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        BLIS_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        OMP_DYNAMIC=FALSE
        MKL_DYNAMIC=FALSE
        CUDA_VISIBLE_DEVICES="${devices}"
        CUDA_DEVICE_ORDER=PCI_BUS_ID
        "${PYTHON}" -m accelerate.commands.accelerate_cli launch
        --main_process_port "${MAIN_PORT}"
        --config_file "${accel_config}"
        -m "${TRAIN_ENTRY_MODULE}" train "${train_config}"
    )
    local -a execution_command=(
        numactl --interleave=all
        "${PYTHON}" "${RESOURCE_EXEC}"
        --profile server
        --output-dir "${run_dir}"
        --expected-cgroup-suffix "${case_unit}.scope"
        -- "${command[@]}"
    )
    execution_command=(
        systemd-run --user --scope --quiet
        --unit "${case_unit}"
        --property MemoryAccounting=yes
        -- "${execution_command[@]}"
    )

    log "pytorch_torch/server: seq=${seq}, GPUs=${NUM_GPUS}, global_batch=${GLOBAL_BATCH_SIZE}, tokens/step=${tokens_per_step}, threads=${CPU_THREADS_PER_RANK}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[DRY-RUN]'
        printf ' %q' "${execution_command[@]}"
        printf '\n'
        printf 'DRY_RUN\n' > "${run_dir}/exit_code.txt"
        return 0
    fi

    start_memory_monitor "${run_dir}"
    ACTIVE_LOG_FIFO="${run_dir}/.train-log.fifo"
    rm -f "${ACTIVE_LOG_FIFO}"
    mkfifo "${ACTIVE_LOG_FIFO}"
    tee "${train_log}" < "${ACTIVE_LOG_FIFO}" &
    ACTIVE_TEE_PID=$!

    local exit_code=0
    set +e
    (
        cd "${SCRIPT_DIR}"
        exec "${execution_command[@]}"
    ) > "${ACTIVE_LOG_FIFO}" 2>&1 &
    ACTIVE_TRAIN_PID=$!
    wait "${ACTIVE_TRAIN_PID}"
    exit_code=$?
    ACTIVE_TRAIN_PID=""
    wait "${ACTIVE_TEE_PID}"
    ACTIVE_TEE_PID=""
    set -e
    rm -f "${ACTIVE_LOG_FIFO}"
    ACTIVE_LOG_FIFO=""
    stop_active_monitor
    analyze_memory_usage "${run_dir}"

    if [[ "${exit_code}" -eq 0 ]]; then
        if [[ ! -f "${timing_dir}/step_timing.json" ]]; then
            warn "Training succeeded but canonical rank-0 timing is missing"
            exit_code=90
        elif ! PYTHONPATH="${FFT_VENDOR_PYTHONPATH}" "${PYTHON}" "${TIMING_VALIDATOR}" \
            --path "${timing_dir}/step_timing.json" \
            --expected-steps "${STEPS}" \
            --warmup-steps "${WARMUP_STEPS}" \
            --backend pytorch_torch; then
            warn "Timing output violates the probe-free phase contract"
            exit_code=92
        fi
    fi
    printf '%s\n' "${exit_code}" > "${run_dir}/exit_code.txt"
    if [[ "${KEEP_MODEL_OUTPUT}" -eq 0 && -d "${run_dir}/model_output" ]]; then
        rm -rf "${run_dir}/model_output"
    fi
    if [[ "${exit_code}" -ne 0 ]]; then
        warn "pytorch_torch/seq_${seq} failed with ${exit_code}"
        return "${exit_code}"
    fi
}

trap 'finalize_sweep_on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

check_files_and_environment
FFT_VENDOR_PYTHONPATH="$(fft_torch_pythonpath "${SCRIPT_DIR}:${SHARED_FLOW_DIR}:${FFT_ROOT}/pytorch_torch")"
check_cluster
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ROOT="${LOG_BASE}/${RUN_TIMESTAMP}_PYTORCH_TORCH_BF16_FULL_SWEEP"
mkdir -p "${RUN_ROOT}"

if [[ "${SKIP_DATASET_CHECK}" -eq 0 ]]; then
    log "Validating GLM-4.5-Air BF16 checkpoint and dataset lengths"
    PYTHONPATH="${FFT_VENDOR_PYTHONPATH}" "${PYTHON}" "${VALIDATOR}" \
        --model-path "${MODEL_PATH}" \
        --dataset-dir "${DATASET_DIR}" \
        --dataset-name "${DATASET_NAME}" \
        --required-length "${MAX_SEQUENCE_LENGTH}" \
        --output-json "${RUN_ROOT}/dataset_validation.json"
fi

log "GLM-4.5-Air PyTorch TORCH full-FT; xmy venv is read-only"
log "Result root: ${RUN_ROOT}"
devices="$(resolve_devices)"
declare -a sequences=()
if [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 1 ]]; then
    sequences=("${SEQUENCE_LENGTHS_OVERRIDE[@]}")
else
    sequences=("${SERVER_SEQUENCE_LENGTHS[@]}")
fi

overall_status=0
for seq in "${sequences[@]}"; do
    if ! run_one_sequence "${seq}" "${devices}"; then
        overall_status=1
        (( CONTINUE_ON_ERROR )) || break
    fi
done
exit "${overall_status}"
