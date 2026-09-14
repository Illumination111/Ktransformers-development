#!/usr/bin/env bash
# GLM-4.5-Air APTMoE component-isomorphic deployment-proxy server sweep.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHARED_TOOLS="${FFT_SHARED_TOOLS:-${FFT_ROOT}/Qwen3.5-35B-A3B}"
MODEL_PATH="${FFT_MODEL_PATH:-/mnt/data2/models/GLM-4.5-Air}"
DATASET_DIR="${FFT_DATASET_DIR:-${FFT_ROOT}/dataset}"
DATASET_NAME="${FFT_DATASET_NAME:-fft_real_100}"
APTMOE_ROOT="${FFT_APTMOE_ROOT:-/mnt/data2/wbw/APTMoE-baseline}"
SIMULATION_ROOT="${FFT_APTMOE_SIMULATION_ROOT:-${FFT_ROOT}/APTMoE-simulate}"
ROUTE_ROOT="${FFT_APTMOE_ROUTE_ROOT:-${SIMULATION_ROOT}/routes/glm45_air/server}"
LOOKUP_TABLE="${FFT_APTMOE_LOOKUP_TABLE:-${SIMULATION_ROOT}/lookups/glm45_air/server.json}"
LOG_BASE="${FFT_LOG_BASE:-${SCRIPT_DIR}/test_log}"
ENTRYPOINT="${SCRIPT_DIR}/aptmoe_glm45_air_proxy_train.py"
AGGREGATOR="${SCRIPT_DIR}/aggregate_sweep_results.py"
RESOURCE_EXEC="${SHARED_TOOLS}/resource_scope_exec.py"
MONITOR="${SHARED_TOOLS}/monitor.py"
MEMORY_ANALYZER="${SHARED_TOOLS}/analyze_memory_usage.py"
TIMING_VALIDATOR="${SHARED_TOOLS}/validate_step_timing.py"

SEQUENCES="32,64,128,256,512,1024,2048,4096"
STEPS=15
WARMUP_STEPS=5
GAS=1
LEARNING_RATE="1.0e-5"
DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CPU_THREADS="${FFT_CPU_THREADS:-1}"
APTMOE_PYTHON="${FFT_APTMOE_PYTHON:-}"
ALLOW_SYNTHETIC=0
ALLOW_UNPROFILED=0
DRY_RUN=0
CONTINUE_ON_ERROR=0

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]
  --seq-lengths LIST       Subset/order from 32,64,...,4096
  --steps N                Default: 15
  --warmup-steps N         Default: 5
  --gas N                  Default: 1
  --learning-rate VALUE    Default: 1e-5
  --devices LIST           Exactly 8 visible GPU ids
  --cpu-threads N
  --model-path PATH        Default: /mnt/data2/models/GLM-4.5-Air
  --dataset-dir PATH
  --dataset-name NAME
  --aptmoe-root PATH
  --aptmoe-python PATH
  --route-root PATH
  --lookup-table PATH
  --simulation-root PATH
  --log-base PATH
  --allow-synthetic-routing
  --allow-unprofiled-placement
                            Either fallback makes every result SMOKE_ONLY
  --continue-on-error
  --dry-run                 Generate contracts; do not launch torchrun
EOF
}

need_value() {
    (( $# >= 2 )) || { echo "missing value for $1" >&2; exit 2; }
}
while (( $# )); do
    case "$1" in
        --seq-lengths) need_value "$@"; SEQUENCES="$2"; shift ;;
        --steps) need_value "$@"; STEPS="$2"; shift ;;
        --warmup-steps) need_value "$@"; WARMUP_STEPS="$2"; shift ;;
        --gas) need_value "$@"; GAS="$2"; shift ;;
        --learning-rate) need_value "$@"; LEARNING_RATE="$2"; shift ;;
        --devices) need_value "$@"; DEVICES="$2"; shift ;;
        --cpu-threads) need_value "$@"; CPU_THREADS="$2"; shift ;;
        --model-path) need_value "$@"; MODEL_PATH="$2"; shift ;;
        --dataset-dir) need_value "$@"; DATASET_DIR="$2"; shift ;;
        --dataset-name) need_value "$@"; DATASET_NAME="$2"; shift ;;
        --aptmoe-root) need_value "$@"; APTMOE_ROOT="$2"; shift ;;
        --aptmoe-python) need_value "$@"; APTMOE_PYTHON="$2"; shift ;;
        --route-root) need_value "$@"; ROUTE_ROOT="$2"; shift ;;
        --lookup-table) need_value "$@"; LOOKUP_TABLE="$2"; shift ;;
        --simulation-root) need_value "$@"; SIMULATION_ROOT="$2"; shift ;;
        --log-base) need_value "$@"; LOG_BASE="$2"; shift ;;
        --allow-synthetic-routing) ALLOW_SYNTHETIC=1 ;;
        --allow-unprofiled-placement) ALLOW_UNPROFILED=1 ;;
        --continue-on-error) CONTINUE_ON_ERROR=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid steps" >&2; exit 2; }
[[ "${WARMUP_STEPS}" =~ ^[0-9]+$ ]] || { echo "invalid warmup" >&2; exit 2; }
[[ "${GAS}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid gas" >&2; exit 2; }
(( WARMUP_STEPS < STEPS )) || { echo "warmup must be less than steps" >&2; exit 2; }

IFS=',' read -r -a SEQUENCE_LIST <<< "${SEQUENCES// /}"
declare -A SEEN=()
for seq in "${SEQUENCE_LIST[@]}"; do
    [[ "${seq}" =~ ^(32|64|128|256|512|1024|2048|4096)$ ]] || {
        echo "unsupported server sequence: ${seq}" >&2
        exit 2
    }
    [[ -z "${SEEN[${seq}]:-}" ]] || { echo "duplicate sequence: ${seq}" >&2; exit 2; }
    SEEN["${seq}"]=1
done

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES// /}"
(( ${#DEVICE_LIST[@]} == 8 )) || {
    echo "server protocol requires exactly 8 GPU ids" >&2
    exit 2
}

# Hold one advisory lock for the complete sweep. GLM and Qwen APTMoE sweeps
# use the same lock naming scheme, so two commands cannot benchmark the same
# ordered GPU set at the same time.
command -v flock >/dev/null || { echo "flock is required for sweep serialization" >&2; exit 1; }
DEVICE_LOCK_KEY="${DEVICES// /}"
DEVICE_LOCK_KEY="${DEVICE_LOCK_KEY//,/_}"
SWEEP_LOCK_PATH="${FFT_APTMOE_SWEEP_LOCK_PATH:-/tmp/fft-aptmoe-gpus-${DEVICE_LOCK_KEY}.lock}"
exec {SWEEP_LOCK_FD}>"${SWEEP_LOCK_PATH}"
flock -n "${SWEEP_LOCK_FD}" || {
    echo "another APTMoE sweep is already using devices ${DEVICES} (lock: ${SWEEP_LOCK_PATH})" >&2
    exit 75
}

find_python() {
    local candidate
    for candidate in \
        "${APTMOE_PYTHON}" \
        /mnt/data2/wbw/conda/envs/Aptmoe/bin/python3 \
        /mnt/data2/wbw/conda/envs/Kllama/bin/python3; do
        [[ -n "${candidate}" && -x "${candidate}" ]] && {
            printf '%s\n' "${candidate}"
            return
        }
    done
    return 1
}
PYTHON="$(find_python)" || { echo "no suitable Python found" >&2; exit 1; }
TORCHRUN="$(dirname "${PYTHON}")/torchrun"
[[ -x "${TORCHRUN}" ]] || TORCHRUN="torchrun"

for required in \
    "${ENTRYPOINT}" "${AGGREGATOR}" "${RESOURCE_EXEC}" "${MONITOR}" \
    "${MEMORY_ANALYZER}" "${TIMING_VALIDATOR}" "${MODEL_PATH}/config.json"; do
    [[ -e "${required}" ]] || { echo "required path missing: ${required}" >&2; exit 1; }
done
if (( DRY_RUN == 0 )); then
    [[ -d "${DATASET_DIR}" ]] || { echo "dataset missing: ${DATASET_DIR}" >&2; exit 1; }
    [[ -d "${APTMOE_ROOT}" ]] || { echo "APTMoE missing: ${APTMOE_ROOT}" >&2; exit 1; }
fi

SMOKE=0
(( ALLOW_SYNTHETIC || ALLOW_UNPROFILED )) && SMOKE=1
STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ROOT="${LOG_BASE}/${STAMP}_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY"
PROFILE_DIR="${RUN_ROOT}/server_8gpu_batch8"
mkdir -p "${PROFILE_DIR}"

write_run_config() {
    local path="$1" seq="$2" route="$3"
    "${PYTHON}" - "${path}" "${seq}" "${route}" "${LOOKUP_TABLE}" \
        "${STEPS}" "${WARMUP_STEPS}" "${GAS}" "${LEARNING_RATE}" \
        "${DEVICES}" "${MODEL_PATH}" "${DATASET_NAME}" "${SMOKE}" \
        "${ALLOW_SYNTHETIC}" "${ALLOW_UNPROFILED}" "${APTMOE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

(
    output, seq, route, lookup, steps, warmup, gas, lr, devices,
    model, dataset, smoke, synthetic, unprofiled, aptmoe,
) = sys.argv[1:]
sequence = int(seq)
obj = {
    "backend": "aptmoe",
    "profile": "server",
    "benchmark_class": "deployment_proxy",
    "result_validity": "SMOKE_ONLY" if int(smoke) else "formal_deployment_proxy",
    "weight_source": "deterministic_random_bf16_initialization",
    "checkpoint_compatible": False,
    "exact_model_claim_allowed": False,
    "llamafactory_backend": False,
    "real_forward_backward_optimizer_update": True,
    "precision": "bf16",
    "finetuning_type": "full",
    "model_load_architecture": "Glm45AirComponentIsomorphicAPTMoEProxy",
    "proxy_target_architecture": "Glm4MoeForCausalLM",
    "sequence_length": sequence,
    "num_gpus": 8,
    "global_batch_size": 8,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": int(gas),
    "tokens_per_step": 8 * sequence * int(gas),
    "steps": int(steps),
    "warmup_steps": int(warmup),
    "learning_rate": lr,
    "devices": devices,
    "model_path": model,
    "dataset_name": dataset,
    "route_trace": route or None,
    "lookup_table": lookup or None,
    "allow_synthetic_routing": bool(int(synthetic)),
    "allow_unprofiled_placement": bool(int(unprofiled)),
    "aptmoe_root": aptmoe,
    "result_scope": "GLM-4.5-Air component-isomorphic APTMoE deployment proxy only",
}
Path(output).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

overall=0
for seq in "${SEQUENCE_LIST[@]}"; do
    echo "starting sequence length ${seq}"
    run_dir="${PROFILE_DIR}/seq_${seq}"
    timing_dir="${run_dir}/step_timing"
    route="${ROUTE_ROOT}/seq_${seq}.npz"
    mkdir -p "${timing_dir}"
    if [[ ! -f "${route}" ]]; then
        if (( ALLOW_SYNTHETIC )); then route=""; else
            echo "formal route trace missing: ${route}" >&2
            exit 1
        fi
    fi
    if [[ ! -f "${LOOKUP_TABLE}" ]]; then
        if (( ALLOW_UNPROFILED )); then lookup_arg=""; else
            echo "formal lookup table missing: ${LOOKUP_TABLE}" >&2
            exit 1
        fi
    else
        lookup_arg="${LOOKUP_TABLE}"
    fi
    original_lookup="${LOOKUP_TABLE}"
    LOOKUP_TABLE="${lookup_arg}"
    write_run_config "${run_dir}/run_config.json" "${seq}" "${route}"
    LOOKUP_TABLE="${original_lookup}"
    if (( DRY_RUN )); then
        printf 'DRY_RUN\n' > "${run_dir}/exit_code.txt"
        continue
    fi
    optional=()
    [[ -z "${route}" ]] || optional+=(--route-trace "${route}")
    [[ -z "${lookup_arg}" ]] || optional+=(--lookup-table "${lookup_arg}")
    (( ALLOW_SYNTHETIC == 0 )) || optional+=(--allow-synthetic-routing)
    (( ALLOW_UNPROFILED == 0 )) || optional+=(--allow-unprofiled-placement)

    mkdir -p "${run_dir}/.mplconfig"
    fifo="${run_dir}/monitor.fifo"
    rm -f "${fifo}"
    env MPLCONFIGDIR="${run_dir}/.mplconfig" \
        "${PYTHON}" "${MONITOR}" --out "${run_dir}/monitor.csv" \
        --fifo "${fifo}" --interval 2 --disk-mount /mnt/data2 --pid "$$" \
        > "${run_dir}/monitor.log" 2>&1 &
    monitor_pid=$!
    set +e
    env CUDA_VISIBLE_DEVICES="${DEVICES}" FFT_CPU_THREADS="${CPU_THREADS}" \
        OMP_NUM_THREADS="${CPU_THREADS}" TRANSFORMERS_OFFLINE=1 \
        HF_DATASETS_OFFLINE=1 \
        PYTHONPATH="${SCRIPT_DIR}:${APTMOE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON}" "${RESOURCE_EXEC}" --profile server \
        --output-dir "${run_dir}" -- \
        "${TORCHRUN}" --standalone --nproc_per_node=8 "${ENTRYPOINT}" \
        --aptmoe-root "${APTMOE_ROOT}" --model-path "${MODEL_PATH}" \
        --dataset-dir "${DATASET_DIR}" --dataset-name "${DATASET_NAME}" \
        --step-timing-output-dir "${timing_dir}" \
        --deployment-profile server --sequence-length "${seq}" \
        --num-gpus 8 --global-batch-size 8 --per-device-batch-size 1 \
        --gradient-accumulation-steps "${GAS}" --steps "${STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" --learning-rate "${LEARNING_RATE}" \
        "${optional[@]}" > "${run_dir}/train.log" 2>&1
    code=$?
    kill -TERM "${monitor_pid}" 2>/dev/null
    wait "${monitor_pid}" 2>/dev/null
    env MPLCONFIGDIR="${run_dir}/.mplconfig" \
        "${PYTHON}" "${MEMORY_ANALYZER}" --log-dir "${run_dir}" \
        > "${run_dir}/memory_analysis.log" 2>&1
    if (( code == 0 )); then
        for artifact in step_timing/step_timing.json memory_summary.json \
            proxy_manifest.json full_update_verification.json; do
            [[ -f "${run_dir}/${artifact}" ]] || code=90
        done
    fi
    if (( code == 0 )); then
        "${PYTHON}" "${TIMING_VALIDATOR}" \
            --path "${timing_dir}/step_timing.json" \
            --expected-steps "${STEPS}" --warmup-steps "${WARMUP_STEPS}" \
            --backend aptmoe || code=92
    fi
    set -e
    printf '%s\n' "${code}" > "${run_dir}/exit_code.txt"
    echo "finished sequence length ${seq} with exit code ${code}"
    if (( code != 0 )); then
        overall=1
        (( CONTINUE_ON_ERROR )) || break
    fi
done

"${PYTHON}" "${AGGREGATOR}" --root "${RUN_ROOT}" || overall=1
echo "result root: ${RUN_ROOT}"
exit "${overall}"
