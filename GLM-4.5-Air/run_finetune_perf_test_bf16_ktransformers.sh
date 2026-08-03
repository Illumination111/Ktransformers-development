#!/usr/bin/env bash
# GLM-4.5-Air native-BF16 full-finetuning sequence sweep.
# Scope is intentionally fixed to KTransformers server/consumer profiles.

set -Eeuo pipefail

export TZ="${FFT_TIMEZONE:-Asia/Shanghai}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
LOG_BASE="${FFT_LOG_BASE:-${SCRIPT_DIR}/test_log}"
LLAMA_FACTORY_DIR="${FFT_LLAMA_FACTORY_DIR:-/mnt/data2/wbw/LLaMA-Factory}"
MODEL_PATH="${FFT_MODEL_PATH:-/mnt/data3/models/GLM-4.5-Air}"
DATASET_DIR="${FFT_DATASET_DIR:-${FFT_ROOT}/dataset}"
DATASET_NAME="${FFT_DATASET_NAME:-fft_real_100}"
SHARED_FLOW_DIR="${FFT_SHARED_FLOW_DIR:-${FFT_ROOT}/Qwen3.5-35B-A3B}"

TRAIN_ENTRY_MODULE="finetune_train_with_timing"
TRAIN_CONFIG_BASE="${CONFIGS_DIR}/train_full_bf16_glm45_air.yaml"
VALIDATOR="${SCRIPT_DIR}/validate_benchmark_dataset.py"
AGGREGATOR="${SCRIPT_DIR}/aggregate_sweep_results.py"
TIMING_VALIDATOR="${SHARED_FLOW_DIR}/validate_step_timing.py"
RESOURCE_EXEC="${SHARED_FLOW_DIR}/resource_scope_exec.py"
MONITOR_SCRIPT="${SHARED_FLOW_DIR}/monitor.py"
MEMORY_ANALYZER="${SHARED_FLOW_DIR}/analyze_memory_usage.py"
STEP_PHASE_TIMER="${SHARED_FLOW_DIR}/step_phase_timer.py"

readonly -a SERVER_SEQUENCE_LENGTHS=(32 64 128 256 512 1024 2048 4096)
readonly -a CONSUMER_SEQUENCE_LENGTHS=(16 32 64 128 256 512 1024 2048)

PROFILE="server"
SEQUENCE_LENGTHS_CSV=""
SEQUENCE_LENGTHS_OVERRIDE_SET=0
SERVER_SEQUENCE_LENGTH=""
SERVER_SEQUENCE_LENGTH_SET=0
STEPS=15
WARMUP_STEPS=5
GRAD_ACCUM_STEPS=1
LEARNING_RATE="1.0e-5"
DEVICES_OVERRIDE=""
CPU_THREADS_OVERRIDE="${FFT_CPU_THREADS:-}"
KT_OWNER_THREADS_OVERRIDE="${FFT_KT_OWNER_THREADS:-}"
KT_DISTRIBUTED_CHECKPOINT_REUSE="${FFT_KT_DISTRIBUTED_CHECKPOINT_REUSE:-on}"
CONSUMER_NUMA_NODES="${FFT_CONSUMER_NUMA_NODES:-0,1}"
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

Fixed benchmark contract:
  backend                 KTransformers AMXBF16
  server                  8 GPUs, global batch 8, seq 32..4096
  consumer                2 GPUs, global batch 2, seq 16..2048
  finetuning type         full

Options:
  --profile MODE          server, consumer or both (default: server)
  --seq-lengths LIST      Comma-separated sequence lengths; order is preserved
  --server-seq-length N   Run exactly one canonical server sequence length
  --steps N               Optimizer steps per sequence (default: 15)
  --warmup-steps N        Steps excluded from stable TPS (default: 5)
  --gas N                 Gradient accumulation steps (default: 1)
  --learning-rate VALUE   Default: 1.0e-5
  --cpu-threads N         Threads for each non-owner rank (default: 2)
  --kt-owner-threads N    Threads for the rank-0 KT CPU owner
  --devices LIST          Physical GPU ids; each profile uses its first N ids
  --model-path PATH       Default: /mnt/data3/models/GLM-4.5-Air
  --dataset-dir PATH      LLaMA-Factory dataset directory
  --dataset-name NAME     Registered dataset name (default: fft_real_100)
  --log-base PATH         Result directory base
  --kt-distributed-checkpoint-reuse on|off
                           Reuse the first KT forward during distributed
                           non-reentrant checkpoint recomputation
  --consumer-numa-nodes LIST
                           Equal-interleave nodes (default: 0,1)
  --continue-on-error     Continue after a failed sequence
  --keep-model-output     Keep generated final model output
  --skip-dataset-check    Skip model/tokenizer/dataset length validation
  --dry-run               Generate configs and print commands only
  -h, --help              Show this help

Every sequence length runs in an independent process, matching the current
Qwen3.5 KTransformers flow. Timing records only coarse forward, backward,
optimizer and total host-wall phases without forced CUDA synchronization.
CPU/GPU sampling runs outside the measured phase path.
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
        --profile) need_value "$1" "$#"; PROFILE="$2"; shift ;;
        --seq-lengths)
            need_value "$1" "$#"; SEQUENCE_LENGTHS_CSV="$2"
            SEQUENCE_LENGTHS_OVERRIDE_SET=1
            shift
            ;;
        --server-seq-length)
            need_value "$1" "$#"; SERVER_SEQUENCE_LENGTH="$2"
            SERVER_SEQUENCE_LENGTH_SET=1
            shift
            ;;
        --steps) need_value "$1" "$#"; STEPS="$2"; shift ;;
        --warmup-steps) need_value "$1" "$#"; WARMUP_STEPS="$2"; shift ;;
        --gas) need_value "$1" "$#"; GRAD_ACCUM_STEPS="$2"; shift ;;
        --learning-rate) need_value "$1" "$#"; LEARNING_RATE="$2"; shift ;;
        --cpu-threads) need_value "$1" "$#"; CPU_THREADS_OVERRIDE="$2"; shift ;;
        --kt-owner-threads) need_value "$1" "$#"; KT_OWNER_THREADS_OVERRIDE="$2"; shift ;;
        --devices) need_value "$1" "$#"; DEVICES_OVERRIDE="$2"; shift ;;
        --model-path) need_value "$1" "$#"; MODEL_PATH="$2"; shift ;;
        --dataset-dir) need_value "$1" "$#"; DATASET_DIR="$2"; shift ;;
        --dataset-name) need_value "$1" "$#"; DATASET_NAME="$2"; shift ;;
        --log-base) need_value "$1" "$#"; LOG_BASE="$2"; shift ;;
        --kt-distributed-checkpoint-reuse)
            need_value "$1" "$#"; KT_DISTRIBUTED_CHECKPOINT_REUSE="$2"; shift ;;
        --consumer-numa-nodes)
            need_value "$1" "$#"; CONSUMER_NUMA_NODES="$2"; shift ;;
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
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || \
        die "${name} must be a positive integer, got ${value}"
}

require_nonnegative_int() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || \
        die "${name} must be a non-negative integer, got ${value}"
}

require_positive_number() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || \
        die "${name} must be a positive number, got ${value}"
    awk -v value="${value}" 'BEGIN { exit !(value > 0) }' || \
        die "${name} must be greater than zero, got ${value}"
}

require_positive_int "--steps" "${STEPS}"
require_nonnegative_int "--warmup-steps" "${WARMUP_STEPS}"
require_positive_int "--gas" "${GRAD_ACCUM_STEPS}"
require_positive_number "--learning-rate" "${LEARNING_RATE}"
(( WARMUP_STEPS < STEPS )) || \
    die "--warmup-steps must be smaller than --steps"
[[ "${PROFILE}" =~ ^(server|consumer|both)$ ]] || \
    die "--profile must be server, consumer or both"
[[ "${KT_DISTRIBUTED_CHECKPOINT_REUSE}" =~ ^(on|off)$ ]] || \
    die "--kt-distributed-checkpoint-reuse must be on or off"
if [[ -n "${CPU_THREADS_OVERRIDE}" ]]; then
    require_positive_int "--cpu-threads" "${CPU_THREADS_OVERRIDE}"
fi
if [[ -n "${KT_OWNER_THREADS_OVERRIDE}" ]]; then
    require_positive_int "--kt-owner-threads" "${KT_OWNER_THREADS_OVERRIDE}"
fi

if [[ "${SERVER_SEQUENCE_LENGTH_SET}" -eq 1 ]]; then
    [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 0 ]] || \
        die "--server-seq-length and --seq-lengths are mutually exclusive"
    [[ "${PROFILE}" == "server" ]] || \
        die "--server-seq-length requires --profile server"
    require_positive_int "--server-seq-length" "${SERVER_SEQUENCE_LENGTH}"
    server_sequence_allowed=0
    for seq in "${SERVER_SEQUENCE_LENGTHS[@]}"; do
        if [[ "${seq}" == "${SERVER_SEQUENCE_LENGTH}" ]]; then
            server_sequence_allowed=1
            break
        fi
    done
    [[ "${server_sequence_allowed}" -eq 1 ]] || \
        die "--server-seq-length must be one of: ${SERVER_SEQUENCE_LENGTHS[*]}"
    SEQUENCE_LENGTHS_CSV="${SERVER_SEQUENCE_LENGTH}"
    SEQUENCE_LENGTHS_OVERRIDE_SET=1
fi

declare -a SEQUENCE_LENGTHS_OVERRIDE=()
if [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 1 ]]; then
    IFS=',' read -r -a SEQUENCE_LENGTHS_OVERRIDE <<< "${SEQUENCE_LENGTHS_CSV// /}"
fi
if [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 1 ]]; then
    (( ${#SEQUENCE_LENGTHS_OVERRIDE[@]} > 0 )) || \
        die "sequence length list is empty"
fi
declare -A SEEN_SEQUENCE=()
MAX_SEQUENCE_LENGTH=0
for seq in "${SEQUENCE_LENGTHS_OVERRIDE[@]}"; do
    require_positive_int "sequence length" "${seq}"
    (( seq >= 16 && seq <= 4096 )) || \
        die "sequence length must be in [16, 4096], got ${seq}"
    if [[ "${seq}" == "16" && "${PROFILE}" =~ ^(server|both)$ ]]; then
        die "sequence length 16 is consumer-only"
    fi
    if [[ "${seq}" == "4096" && "${PROFILE}" =~ ^(consumer|both)$ ]]; then
        die "sequence length 4096 is server-only"
    fi
    [[ -z "${SEEN_SEQUENCE[${seq}]:-}" ]] || \
        die "duplicate sequence length: ${seq}"
    SEEN_SEQUENCE["${seq}"]=1
    (( seq > MAX_SEQUENCE_LENGTH )) && MAX_SEQUENCE_LENGTH="${seq}"
done
if [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 0 ]]; then
    case "${PROFILE}" in
        server|both) MAX_SEQUENCE_LENGTH=4096 ;;
        consumer) MAX_SEQUENCE_LENGTH=2048 ;;
    esac
fi

_find_conda_python() {
    local env_name="$1"
    local candidates=(
        "/mnt/data2/wbw/conda/envs/${env_name}/bin/python3"
        "/mnt/data2/wbw/miniconda3/envs/${env_name}/bin/python3"
        "/opt/conda/envs/${env_name}/bin/python3"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        [[ -x "${candidate}" ]] && {
            printf '%s\n' "${candidate}"
            return 0
        }
    done
    return 1
}

CONDA_ENV="${FFT_CONDA_ENV:-Kllama}"
PYTHON="$(_find_conda_python "${CONDA_ENV}" || true)"
[[ -n "${PYTHON}" ]] || die "Python for ${CONDA_ENV} was not found"
CONDA_BIN_DIR="$(dirname "${PYTHON}")"
MONITOR_PYTHON="$(_find_conda_python Deepspeed || true)"
[[ -n "${MONITOR_PYTHON}" ]] || \
    die "Deepspeed Python is required for the existing monitor dependencies"

detect_physical_cores() {
    "${PYTHON}" - <<'PY'
import os
from pathlib import Path

try:
    cpu_ids = set(os.sched_getaffinity(0))
except (AttributeError, OSError):
    cpu_ids = set(range(os.cpu_count() or 1))
cores = set()
for cpu_id in cpu_ids:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
    try:
        cores.add(
            (
                (topology / "physical_package_id").read_text(),
                (topology / "core_id").read_text(),
            )
        )
    except OSError:
        pass
print(max(1, len(cores) if cores else len(cpu_ids)))
PY
}

PHYSICAL_CORES="$(detect_physical_cores)"

profile_parameters() {
    local profile_name="$1"
    case "${profile_name}" in
        server)
            NUM_GPUS=8
            GLOBAL_BATCH_SIZE=8
            ;;
        consumer)
            NUM_GPUS=2
            GLOBAL_BATCH_SIZE=2
            ;;
        *) die "internal profile: ${profile_name}" ;;
    esac
    PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))
    CPU_THREADS_PER_RANK="${CPU_THREADS_OVERRIDE:-2}"
    if [[ -n "${KT_OWNER_THREADS_OVERRIDE}" ]]; then
        KT_OWNER_THREADS="${KT_OWNER_THREADS_OVERRIDE}"
    else
        KT_OWNER_THREADS=$((PHYSICAL_CORES - 2 - CPU_THREADS_PER_RANK * (NUM_GPUS - 1)))
        (( KT_OWNER_THREADS > 0 )) || \
            die "No physical cores remain for the rank-0 KT owner"
    fi
    CPU_THREAD_BUDGET_TOTAL=$((KT_OWNER_THREADS + CPU_THREADS_PER_RANK * (NUM_GPUS - 1)))
    (( KT_OWNER_THREADS <= PHYSICAL_CORES )) || \
        die "KT owner threads exceed visible physical cores"
    if (( CPU_THREAD_BUDGET_TOTAL > PHYSICAL_CORES )); then
        warn "CPU thread budget ${CPU_THREAD_BUDGET_TOTAL} exceeds ${PHYSICAL_CORES} physical cores"
    fi
}

check_files_and_environment() {
    [[ -d "${MODEL_PATH}" ]] || die "model directory not found: ${MODEL_PATH}"
    [[ -d "${DATASET_DIR}" ]] || die "dataset directory not found: ${DATASET_DIR}"
    [[ -d "${LLAMA_FACTORY_DIR}/src/llamafactory" ]] || \
        die "LLaMA-Factory source not found: ${LLAMA_FACTORY_DIR}"
    local required
    for required in \
        "${TRAIN_CONFIG_BASE}" \
        "${CONFIGS_DIR}/accelerate_ktransformers_bf16_2gpu.yaml" \
        "${CONFIGS_DIR}/accelerate_ktransformers_bf16_8gpu.yaml" \
        "${VALIDATOR}" \
        "${AGGREGATOR}" "${TIMING_VALIDATOR}" "${RESOURCE_EXEC}" \
        "${MONITOR_SCRIPT}" "${MEMORY_ANALYZER}" "${STEP_PHASE_TIMER}"; do
        [[ -f "${required}" ]] || die "required file not found: ${required}"
    done
    "${PYTHON}" -c \
        'import accelerate, ktransformers, kt_kernel, transformers' || \
        die "KTransformers dependencies are unavailable in ${CONDA_ENV}"
    env MPLCONFIGDIR=/tmp/fft_glm45_matplotlib \
        "${MONITOR_PYTHON}" -c 'import matplotlib, psutil, pynvml' || \
        die "memory monitoring dependencies are unavailable"
}

resolve_devices() {
    local requested="$1"
    local source="${DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-}}"
    [[ -n "${source}" ]] || source="0,1,2,3,4,5,6,7"
    source="${source// /}"
    local -a candidates
    IFS=',' read -r -a candidates <<< "${source}"
    (( ${#candidates[@]} >= requested )) || \
        die "GPU list '${source}' has fewer than ${requested} entries"
    local -a selected=("${candidates[@]:0:requested}")
    local device
    declare -A seen=()
    for device in "${selected[@]}"; do
        [[ "${device}" =~ ^[0-9]+$ ]] || die "invalid GPU id: ${device}"
        [[ -z "${seen[${device}]:-}" ]] || die "duplicate GPU id: ${device}"
        seen["${device}"]=1
    done
    local joined
    joined="$(IFS=','; printf '%s' "${selected[*]}")"
    printf '%s\n' "${joined}"
}

check_visible_gpu_capacity() {
    local requested="$1"
    [[ "${DRY_RUN}" -eq 1 ]] && return
    command -v nvidia-smi >/dev/null || \
        die "nvidia-smi is required for a real run"
    local actual
    actual="$(nvidia-smi -L | wc -l)"
    (( actual >= requested )) || \
        die "requested ${requested} GPUs but only ${actual} were detected"
}

check_numa_capacity() {
    command -v numactl >/dev/null || \
        die "numactl is required for the consumer NUMA policy"
    local node total_kib
    local -a nodes
    IFS=',' read -r -a nodes <<< "${CONSUMER_NUMA_NODES// /}"
    (( ${#nodes[@]} == 2 )) || \
        die "consumer requires exactly two NUMA nodes"
    for node in "${nodes[@]}"; do
        [[ "${node}" =~ ^[0-9]+$ ]] || die "invalid NUMA node: ${node}"
        [[ -r "/sys/devices/system/node/node${node}/meminfo" ]] || \
            die "NUMA node ${node} is unavailable"
        total_kib="$(awk '/MemTotal/ {print $4}' \
            "/sys/devices/system/node/node${node}/meminfo")"
        (( total_kib >= 536870912 )) || \
            die "NUMA node ${node} has less than 512 GiB total memory"
    done
}

declare -a RESOURCE_PREFIX=()
MEMORY_LIMIT_LABEL=""
NUMA_POLICY_LABEL=""

build_resource_policy() {
    local profile_name="$1"
    RESOURCE_PREFIX=()
    if [[ "${profile_name}" == "server" ]]; then
        MEMORY_LIMIT_LABEL="host-unlimited (~2T visible)"
        NUMA_POLICY_LABEL="host/default nodes 0,1"
        return
    fi
    check_numa_capacity
    RESOURCE_PREFIX=(numactl "--interleave=${CONSUMER_NUMA_NODES}")
    MEMORY_LIMIT_LABEL="uncapped; sampled for manual 1TiB review"
    NUMA_POLICY_LABEL="equal interleave nodes ${CONSUMER_NUMA_NODES}"
}

set_yaml_value() {
    local file="$1" key="$2" value="$3"
    if grep -q "^${key}:" "${file}"; then
        sed -i "s|^${key}:.*|${key}: ${value}|" "${file}"
    else
        printf '%s: %s\n' "${key}" "${value}" >> "${file}"
    fi
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
    set_yaml_value "${config}" gradient_checkpointing "true"
    set_yaml_value "${config}" gradient_checkpointing_kwargs "{use_reentrant: false}"
    set_yaml_value "${config}" use_kt "true"
    set_yaml_value "${config}" kt_weight_path "${MODEL_PATH}"
    printf '%s\n' "${config}"
}

make_accel_config() {
    local run_dir="$1"
    local config="${run_dir}/accelerate_config.yaml"
    local template="${CONFIGS_DIR}/accelerate_ktransformers_bf16_${NUM_GPUS}gpu.yaml"
    [[ -f "${template}" ]] || \
        die "accelerate config not found: ${template}"
    cp "${template}" "${config}"
    sed -i \
        "s|^  kt_num_threads:.*|  kt_num_threads: ${KT_OWNER_THREADS}|" \
        "${config}"
    printf '%s\n' "${config}"
}

write_run_config() {
    local path="$1" profile_name="$2" seq="$3" devices="$4"
    local tokens_per_step="$5"
    "${PYTHON}" - \
        "${path}" "${profile_name}" "${seq}" "${devices}" \
        "${tokens_per_step}" "${NUM_GPUS}" "${GLOBAL_BATCH_SIZE}" \
        "${PER_DEVICE_BATCH_SIZE}" "${MEMORY_LIMIT_LABEL}" \
        "${NUMA_POLICY_LABEL}" "${STEPS}" \
        "${WARMUP_STEPS}" "${GRAD_ACCUM_STEPS}" "${LEARNING_RATE}" \
        "${MODEL_PATH}" "${DATASET_NAME}" "${CPU_THREADS_PER_RANK}" \
        "${KT_OWNER_THREADS}" "${CPU_THREAD_BUDGET_TOTAL}" "${DRY_RUN}" \
        "${KT_DISTRIBUTED_CHECKPOINT_REUSE}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
obj = {
    "backend": "ktransformers",
    "profile": sys.argv[2],
    "benchmark_class": "exact_model_full_finetune",
    "result_validity": "exact_model",
    "precision": "bf16",
    "modality": "text_only",
    "finetuning_type": "full",
    "source_architecture": "Glm4MoeForCausalLM",
    "model_load_architecture": "Glm4MoeForCausalLM",
    "sequence_length": int(sys.argv[3]),
    "num_gpus": int(sys.argv[6]),
    "global_batch_size": int(sys.argv[7]),
    "per_device_batch_size": int(sys.argv[8]),
    "gradient_accumulation_steps": int(sys.argv[13]),
    "tokens_per_step": int(sys.argv[5]),
    "steps": int(sys.argv[11]),
    "warmup_steps": int(sys.argv[12]),
    "learning_rate": sys.argv[14],
    "devices": sys.argv[4],
    "model_path": sys.argv[15],
    "dataset_name": sys.argv[16],
    "memory_limit": sys.argv[9],
    "memory_enforcement": "none_by_benchmark",
    "memory_monitoring": True,
    "automatic_memory_termination": False,
    "automatic_oom_classification": False,
    "gpu_allocation_lifetime": "sequence_process_lifetime",
    "persistent_profile_process": False,
    "artificial_gpu_reservation": False,
    "numa_policy": sys.argv[10],
    "cpu_threads_per_rank": int(sys.argv[17]),
    "kt_owner_rank": 0,
    "kt_owner_threads": int(sys.argv[18]),
    "cpu_thread_budget_total": int(sys.argv[19]),
    "dry_run": bool(int(sys.argv[20])),
    "kt_distributed_checkpoint_forward_reuse": sys.argv[21] == "on",
    "timing_mode": "coarse_host_wall_no_cuda_sync",
    "result_scope": "end-to-end GLM-4.5-Air full-finetune throughput",
}
out.write_text(
    json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
}

print_command() {
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
}

stop_active_monitor() {
    if [[ -n "${ACTIVE_MONITOR_PID}" ]] && \
       kill -0 "${ACTIVE_MONITOR_PID}" 2>/dev/null; then
        kill -TERM "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
        wait "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    fi
    ACTIVE_MONITOR_PID=""
    if [[ -n "${ACTIVE_MONITOR_FIFO}" ]]; then
        rm -f "${ACTIVE_MONITOR_FIFO}"
    fi
    ACTIVE_MONITOR_FIFO=""
}

stop_active_training() {
    if [[ -n "${ACTIVE_TRAIN_PID}" ]] && \
       kill -0 "${ACTIVE_TRAIN_PID}" 2>/dev/null; then
        kill -TERM "${ACTIVE_TRAIN_PID}" 2>/dev/null || true
        wait "${ACTIVE_TRAIN_PID}" 2>/dev/null || true
    fi
    ACTIVE_TRAIN_PID=""
    if [[ -n "${ACTIVE_TEE_PID}" ]] && \
       kill -0 "${ACTIVE_TEE_PID}" 2>/dev/null; then
        kill -TERM "${ACTIVE_TEE_PID}" 2>/dev/null || true
        wait "${ACTIVE_TEE_PID}" 2>/dev/null || true
    fi
    ACTIVE_TEE_PID=""
    if [[ -n "${ACTIVE_LOG_FIFO}" ]]; then
        rm -f "${ACTIVE_LOG_FIFO}"
    fi
    ACTIVE_LOG_FIFO=""
}

cleanup_active_processes() {
    stop_active_training
    stop_active_monitor
}

generate_sweep_summary() {
    [[ -n "${RUN_ROOT}" && -d "${RUN_ROOT}" ]] || return 0
    [[ "${SUMMARY_FINALIZED}" -eq 0 ]] || return 0
    compgen -G "${RUN_ROOT}/*/seq_*/run_config.json" \
        >/dev/null || return 0
    SUMMARY_FINALIZED=1
    if "${PYTHON}" "${AGGREGATOR}" --root "${RUN_ROOT}"; then
        log "Sweep summary: ${RUN_ROOT}/summary.md"
        log "Machine-readable results: ${RUN_ROOT}/sweep_results.csv"
        return 0
    fi
    warn "Sweep aggregation failed for ${RUN_ROOT}"
    return 98
}

finalize_sweep_on_exit() {
    local original_status="${1:-0}"
    local summary_status=0
    trap - EXIT
    set +e
    cleanup_active_processes
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
        --interval 2 \
        --disk-mount /mnt/data2 \
        --pid "$$" \
        >> "${run_dir}/monitor.log" 2>&1 &
    ACTIVE_MONITOR_PID=$!
    local attempt
    for attempt in {1..30}; do
        [[ -f "${run_dir}/monitor.csv" ]] && break
        kill -0 "${ACTIVE_MONITOR_PID}" 2>/dev/null || \
            die "memory monitor failed; see ${run_dir}/monitor.log"
        sleep 0.1
    done
    [[ -f "${run_dir}/monitor.csv" ]] || \
        die "memory monitor did not create monitor.csv"
}

analyze_memory_usage() {
    local run_dir="$1"
    env MPLCONFIGDIR="${run_dir}/.mplconfig" \
        "${MONITOR_PYTHON}" "${MEMORY_ANALYZER}" \
        --log-dir "${run_dir}" \
        >> "${run_dir}/memory_analysis.log" 2>&1 || \
        warn "Memory analysis failed; see ${run_dir}/memory_analysis.log"
}

run_one_sequence() {
    local profile_name="$1" profile_dir="$2" seq="$3" devices="$4"
    local run_dir="${profile_dir}/seq_${seq}"
    local timing_dir="${run_dir}/step_timing"
    local train_log="${run_dir}/train.log"
    local tokens_per_step=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE * seq * GRAD_ACCUM_STEPS))
    local empty_cache_after_prepare=0
    if [[ "${profile_name}" == "consumer" ]]; then
        empty_cache_after_prepare=1
    fi
    mkdir -p "${run_dir}"

    local train_config accel_config
    train_config="$(make_train_config "${run_dir}" "${seq}")"
    accel_config="$(make_accel_config "${run_dir}")"
    write_run_config \
        "${run_dir}/run_config.json" "${profile_name}" "${seq}" "${devices}" \
        "${tokens_per_step}"

    local reuse_enabled=0
    [[ "${KT_DISTRIBUTED_CHECKPOINT_REUSE}" == "on" ]] && reuse_enabled=1
    local accelerate_bin="${CONDA_BIN_DIR}/accelerate"
    [[ -x "${accelerate_bin}" ]] || accelerate_bin="accelerate"
    local -a command=(
        env
        USE_KT=1
        ACCELERATE_USE_KT=true
        ACCELERATE_KT_TRAIN_MODE=full
        ACCELERATE_KT_LORA_RANK=0
        ACCELERATE_KT_LORA_ALPHA=0
        KT_FINETUNE_MODE=full
        FFT_TRAINING_BACKEND=kt
        FFT_PRECISION=bf16
        FFT_FINETUNING_TYPE=full
        FFT_SKIP_FINAL_SAVE="$((1 - KEEP_MODEL_OUTPUT))"
        FFT_STEP_TIMING_OUT_DIR="${timing_dir}"
        FFT_STEP_TIMING_WARMUP_STEPS="${WARMUP_STEPS}"
        FFT_STEP_TIMING_TOKENS_PER_STEP="${tokens_per_step}"
        FFT_DISABLE_PERF_PROBES=1
        FFT_CUDA_EMPTY_CACHE_AFTER_PREPARE="${empty_cache_after_prepare}"
        FFT_CPU_THREADS="${CPU_THREADS_PER_RANK}"
        FFT_KT_OWNER_THREADS="${KT_OWNER_THREADS}"
        FFT_KT_NON_OWNER_THREADS="${CPU_THREADS_PER_RANK}"
        KT_BACKWARD_TIMING=off
        KT_SFT_PROFILE=0
        KT_REUSE_CHECKPOINT_FORWARD="${reuse_enabled}"
        KT_REUSE_CHECKPOINT_FORWARD_DISTRIBUTED="${reuse_enabled}"
        DS_PROBE_MODE=off
        ACCELERATE_KT_MODEL_MAX_LENGTH="${seq}"
        OMP_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        MKL_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        BLIS_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        OMP_DYNAMIC=FALSE
        MKL_DYNAMIC=FALSE
        ACCELERATE_KT_OMP_NUM_THREADS="${CPU_THREADS_PER_RANK}"
        TOKENIZERS_PARALLELISM=false
        HF_DATASETS_OFFLINE=1
        TRANSFORMERS_OFFLINE=1
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        CUDA_VISIBLE_DEVICES="${devices}"
        PYTHONPATH="${SCRIPT_DIR}:${SHARED_FLOW_DIR}:${LLAMA_FACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
        "${accelerate_bin}" launch
        --config_file "${accel_config}"
        -m "${TRAIN_ENTRY_MODULE}" train "${train_config}"
    )
    local -a execution_command=(
        "${RESOURCE_PREFIX[@]}"
        "${PYTHON}" "${RESOURCE_EXEC}"
        --profile "${profile_name}"
        --numa-nodes "${CONSUMER_NUMA_NODES}"
        --output-dir "${run_dir}"
        -- "${command[@]}"
    )

    log "ktransformers/${profile_name}: seq=${seq}, GPUs=${NUM_GPUS}, global_batch=${GLOBAL_BATCH_SIZE}, tokens/step=${tokens_per_step}, full, BF16, memory=${MEMORY_LIMIT_LABEL}, NUMA=${NUMA_POLICY_LABEL}, KT owner threads=${KT_OWNER_THREADS}, non-owner threads=${CPU_THREADS_PER_RANK}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        print_command "${execution_command[@]}"
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
        cd "${LLAMA_FACTORY_DIR}"
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

    local monitor_failed=0
    if [[ -z "${ACTIVE_MONITOR_PID}" ]] || \
       ! kill -0 "${ACTIVE_MONITOR_PID}" 2>/dev/null; then
        monitor_failed=1
    fi
    stop_active_monitor
    analyze_memory_usage "${run_dir}"
    if [[ "${exit_code}" -eq 0 && "${monitor_failed}" -ne 0 ]]; then
        warn "Training succeeded but the memory monitor exited early"
        exit_code=89
    elif [[ "${exit_code}" -eq 0 && \
            ! -f "${run_dir}/memory_summary.json" ]]; then
        warn "Training succeeded but memory_summary.json is missing"
        exit_code=89
    fi

    if [[ "${exit_code}" -eq 0 ]]; then
        if [[ ! -f "${timing_dir}/step_timing.json" ]]; then
            warn "Training succeeded but canonical rank-0 timing is missing"
            exit_code=90
        elif ! "${PYTHON}" "${TIMING_VALIDATOR}" \
            --path "${timing_dir}/step_timing.json" \
            --expected-steps "${STEPS}" \
            --warmup-steps "${WARMUP_STEPS}" \
            --backend ktransformers; then
            warn "Timing output violates the probe-free phase contract"
            exit_code=92
        fi
    fi
    printf '%s\n' "${exit_code}" > "${run_dir}/exit_code.txt"

    if [[ "${KEEP_MODEL_OUTPUT}" -eq 0 && \
          -d "${run_dir}/model_output" ]]; then
        log "Removing generated model output for seq=${seq}"
        rm -rf "${run_dir}/model_output"
    fi
    if [[ "${exit_code}" -ne 0 ]]; then
        warn "ktransformers/${profile_name}/seq_${seq} failed with ${exit_code}"
        return "${exit_code}"
    fi
}

trap 'finalize_sweep_on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

check_files_and_environment
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ROOT="${LOG_BASE}/${RUN_TIMESTAMP}_KTRANSFORMERS_BF16_FULL_SWEEP"
mkdir -p "${RUN_ROOT}"

if [[ "${SKIP_DATASET_CHECK}" -eq 0 ]]; then
    log "Validating GLM-4.5-Air BF16 checkpoint and dataset lengths"
    "${PYTHON}" "${VALIDATOR}" \
        --model-path "${MODEL_PATH}" \
        --dataset-dir "${DATASET_DIR}" \
        --dataset-name "${DATASET_NAME}" \
        --required-length "${MAX_SEQUENCE_LENGTH}" \
        --output-json "${RUN_ROOT}/dataset_validation.json"
else
    warn "Dataset tokenizer validation skipped by request"
fi

log "GLM-4.5-Air full-finetuning sweep: backend=ktransformers, profile=${PROFILE}, precision=BF16"
log "KTransformers checkpoint forward reuse: ${KT_DISTRIBUTED_CHECKPOINT_REUSE}"
log "Result root: ${RUN_ROOT}"

run_profile() {
    local profile_name="$1"
    profile_parameters "${profile_name}"
    build_resource_policy "${profile_name}"
    check_visible_gpu_capacity "${NUM_GPUS}"
    local devices
    devices="$(resolve_devices "${NUM_GPUS}")"
    local profile_dir="${RUN_ROOT}/${profile_name}_${NUM_GPUS}gpu_batch${GLOBAL_BATCH_SIZE}"
    mkdir -p "${profile_dir}"

    local -a profile_sequences=()
    if [[ "${SEQUENCE_LENGTHS_OVERRIDE_SET}" -eq 1 ]]; then
        profile_sequences=("${SEQUENCE_LENGTHS_OVERRIDE[@]}")
    elif [[ "${profile_name}" == "server" ]]; then
        profile_sequences=("${SERVER_SEQUENCE_LENGTHS[@]}")
    else
        profile_sequences=("${CONSUMER_SEQUENCE_LENGTHS[@]}")
    fi

    log "Profile ${profile_name}: GPUs=${NUM_GPUS}, global_batch=${GLOBAL_BATCH_SIZE}, devices=${devices}, sequences=${profile_sequences[*]}"
    local profile_status=0
    local seq
    for seq in "${profile_sequences[@]}"; do
        if ! run_one_sequence \
            "${profile_name}" "${profile_dir}" "${seq}" "${devices}"; then
            profile_status=1
            if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
                break
            fi
        fi
    done
    return "${profile_status}"
}

declare -a PROFILES=()
case "${PROFILE}" in
    server) PROFILES=(server) ;;
    consumer) PROFILES=(consumer) ;;
    both) PROFILES=(server consumer) ;;
esac

overall_status=0
for selected_profile in "${PROFILES[@]}"; do
    if ! run_profile "${selected_profile}"; then
        overall_status=1
        if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
            break
        fi
    fi
done
generate_sweep_summary || overall_status=$?
exit "${overall_status}"
