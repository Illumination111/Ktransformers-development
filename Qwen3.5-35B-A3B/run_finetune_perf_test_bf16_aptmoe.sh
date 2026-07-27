#!/usr/bin/env bash
# APTMoE synthetic benchmark defined by FFTtest/docs/aptmoe.
#
# Important: despite this script living beside the Qwen3.5-35B-A3B tests, the
# authoritative APTMoE artifact models Qwen3.5-397B-A17B as a ~391B synthetic
# TransformerLM. It is not an end-to-end or component-isomorphic 35B benchmark.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC_DIR="$(cd "${SCRIPT_DIR}/../docs/aptmoe" && pwd)"
APTMOE_ROOT="${FFT_APTMOE_ROOT:-/mnt/data2/wbw/APTMoE-baseline}"
APTMOE_PYTHON="${FFT_APTMOE_PYTHON:-}"
LOG_BASE="${FFT_LOG_BASE:-${SCRIPT_DIR}/test_log}"

PROFILE="server"
FINETUNING_TYPE="full"
SEQUENCE_LENGTHS_CSV="512,1024"
MEASURE_STEPS=3
WARMUP_STEPS=1
TOP_K=2
LORA_RANK=8
LORA_ALPHA=16
DEVICES="0,1,2,3,4,5,6,7"
MASTER_PORT=29501
CONTINUE_ON_ERROR=0
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

This runs the synthetic QWEN35_397B APTMoE benchmark documented in:
  ${DOC_DIR}/README.md

Options:
  --profile server              Only the documented 8-GPU server profile is valid
  --finetuning-type full|lora   Default: full
  --seq-lengths LIST            Default: 512,1024
  --top-k N                     Experts/token; default: 2
                                The real 397B config is top-10, but documented
                                runnable sweeps use top-1/2/4 because of OOM
  --steps N                     Measured steps after warmup; default: 3
  --warmup-steps N              Excluded warmup steps; default: 1
  --gas 1                       Accepted for compatibility; APTMoE does not
                                support gradient accumulation
  --lora-rank N                 Default: 8
  --lora-alpha N                Default: 16
  --devices LIST                Exactly eight physical GPU IDs
  --master-port N               First torchrun port; default: 29501
  --aptmoe-root PATH            Default: ${APTMOE_ROOT}
  --aptmoe-python PATH          Python in the APTMoE environment
  --log-base PATH               Result directory base
  --continue-on-error           Continue the sequence sweep after a failed run
  --dry-run                     Audit the implementation and print commands only
  -h, --help                    Show this help

Fixed authoritative parameters:
  model_config=QWEN35_397B, batch_size=1, num_chunks=1,
  pipeline=APTMoE, gini=0.3, topo=C1+G2, BF16.

TPS is reported by APTMoE as batch_size * sequence_length / step_time.
The eight ranks form one pipeline; TPS must not be multiplied by GPU count.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

need_value() {
    [[ "$2" -ge 2 ]] || die "missing value for $1"
}

require_positive_int() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer, got: $2"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            need_value "$1" "$#"; PROFILE="$2"; shift
            ;;
        --finetuning-type)
            need_value "$1" "$#"; FINETUNING_TYPE="$2"; shift
            ;;
        --seq-lengths)
            need_value "$1" "$#"; SEQUENCE_LENGTHS_CSV="${2// /}"; shift
            ;;
        --top-k)
            need_value "$1" "$#"; TOP_K="$2"; shift
            ;;
        --steps)
            need_value "$1" "$#"; MEASURE_STEPS="$2"; shift
            ;;
        --warmup-steps)
            need_value "$1" "$#"; WARMUP_STEPS="$2"; shift
            ;;
        --gas)
            need_value "$1" "$#"
            [[ "$2" == "1" ]] || die "APTMoE pipeline does not support gradient accumulation; --gas must be 1"
            shift
            ;;
        --lora-rank)
            need_value "$1" "$#"; LORA_RANK="$2"; shift
            ;;
        --lora-alpha)
            need_value "$1" "$#"; LORA_ALPHA="$2"; shift
            ;;
        --devices)
            need_value "$1" "$#"; DEVICES="${2// /}"; shift
            ;;
        --master-port)
            need_value "$1" "$#"; MASTER_PORT="$2"; shift
            ;;
        --aptmoe-root)
            need_value "$1" "$#"; APTMOE_ROOT="$2"; shift
            ;;
        --aptmoe-python)
            need_value "$1" "$#"; APTMOE_PYTHON="$2"; shift
            ;;
        --log-base)
            need_value "$1" "$#"; LOG_BASE="$2"; shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

[[ "${PROFILE}" == "server" ]] || \
    die "docs/aptmoe defines this QWEN35_397B benchmark on 8 GPUs only; profile '${PROFILE}' is unsupported"
[[ "${FINETUNING_TYPE}" =~ ^(full|lora)$ ]] || \
    die "--finetuning-type must be full or lora"
require_positive_int "--steps" "${MEASURE_STEPS}"
require_positive_int "--warmup-steps" "${WARMUP_STEPS}"
require_positive_int "--top-k" "${TOP_K}"
require_positive_int "--lora-rank" "${LORA_RANK}"
require_positive_int "--lora-alpha" "${LORA_ALPHA}"
require_positive_int "--master-port" "${MASTER_PORT}"
(( TOP_K <= 512 )) || die "--top-k cannot exceed the configured 512 experts"
(( MASTER_PORT <= 65535 )) || die "--master-port must not exceed 65535"

IFS=',' read -r -a DEVICE_IDS <<< "${DEVICES}"
(( ${#DEVICE_IDS[@]} == 8 )) || \
    die "the documented benchmark requires exactly 8 GPU IDs, got: ${DEVICES}"
declare -A SEEN_DEVICES=()
for device_id in "${DEVICE_IDS[@]}"; do
    [[ "${device_id}" =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${device_id}"
    [[ -z "${SEEN_DEVICES[${device_id}]:-}" ]] || die "duplicate GPU ID: ${device_id}"
    SEEN_DEVICES["${device_id}"]=1
done

IFS=',' read -r -a SEQUENCE_LENGTHS <<< "${SEQUENCE_LENGTHS_CSV}"
(( ${#SEQUENCE_LENGTHS[@]} > 0 )) || die "--seq-lengths cannot be empty"
declare -A SEEN_SEQUENCES=()
for seq in "${SEQUENCE_LENGTHS[@]}"; do
    [[ "${seq}" =~ ^(512|1024|2048|4096|8192|16384)$ ]] || \
        die "unsupported sequence length ${seq}; docs/aptmoe uses 512..16384 powers of two"
    [[ -z "${SEEN_SEQUENCES[${seq}]:-}" ]] || die "duplicate sequence length: ${seq}"
    SEEN_SEQUENCES["${seq}"]=1
done
if [[ "${FINETUNING_TYPE}" == "full" ]]; then
    if (( TOP_K >= 4 )); then
        warn "docs/aptmoe reports QWEN35_397B full training top-k >= 4 is killed by CPU RAM pressure"
    fi
    for seq in "${SEQUENCE_LENGTHS[@]}"; do
        if (( seq >= 2048 )); then
            warn "docs/aptmoe reports QWEN35_397B full training seq >= 2048 can be killed by CPU RAM pressure"
            break
        fi
    done
elif (( TOP_K >= 2 )); then
    for seq in "${SEQUENCE_LENGTHS[@]}"; do
        if (( seq >= 8192 )); then
            warn "docs/aptmoe reports LoRA top-k >= 2 at seq >= 8192 can hit GPU OOM"
            break
        fi
    done
fi

if [[ -z "${APTMOE_PYTHON}" ]]; then
    for candidate in \
        /mnt/data2/wbw/conda/envs/Aptmoe/bin/python3 \
        /mnt/data2/wbw/miniconda3/envs/Aptmoe/bin/python3 \
        /mnt/data2/hxx/mini/envs/sft/bin/python3
    do
        if [[ -x "${candidate}" ]]; then
            APTMOE_PYTHON="${candidate}"
            break
        fi
    done
fi

[[ -x "${APTMOE_PYTHON}" ]] || \
    die "APTMoE Python was not found; pass --aptmoe-python"
[[ -d "${APTMOE_ROOT}" ]] || die "APTMoE root not found: ${APTMOE_ROOT}"
[[ -f "${APTMOE_ROOT}/main.py" ]] || die "APTMoE main.py not found under ${APTMOE_ROOT}"
TORCHRUN="$(dirname "${APTMOE_PYTHON}")/torchrun"
[[ -x "${TORCHRUN}" ]] || die "torchrun not found beside ${APTMOE_PYTHON}"

verify_aptmoe_contract() {
    local required_source_patterns=(
        "main.py:QWEN35_397B"
        "main.py:num_experts_per_tok"
        "utils.py:QWEN35_397B"
        "model/transformer_lm.py:gate_proj"
        "model/transformer_lm.py:up_proj"
        "model/transformer_lm.py:down_proj"
        "model/top2gate.py:num_experts_per_tok"
        "model/moe_layer.py:_expand_for_topk"
        "model/moe_layer.py:_reduce_from_topk"
        "model/lora.py:gate_proj"
        "Static/lookup_table.py:LookupTable_QWEN35_397B"
        "Runtime/OffloadRuntime/R_solver.py:QWEN35_397B"
        "Runtime/OffloadRuntime/offload.py:prefetch_portion = 0.1"
    )
    local item relative_path pattern
    for item in "${required_source_patterns[@]}"; do
        relative_path="${item%%:*}"
        pattern="${item#*:}"
        grep -Fq "${pattern}" "${APTMOE_ROOT}/${relative_path}" || \
            die "APTMoE checkout does not match docs/aptmoe: ${relative_path} lacks ${pattern}. Apply ${DOC_DIR}/aptmoe_changes.diff"
    done

    env PYTHONPATH="${APTMOE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${APTMOE_PYTHON}" - "${TOP_K}" <<'PY'
from argparse import Namespace
import sys

import torch

from model.top2gate import random_gating
from utils import model_config

top_k = int(sys.argv[1])
args = Namespace(batch_size=64, num_chunks=4, seq_length=128)
args = model_config(args, "QWEN35_397B")
expected = {
    "embedding_dim": 4096,
    "hidden_dim": 1024,
    "num_heads": 32,
    "num_layers": 60,
    "num_stages": 60,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "batch_size": 1,
    "num_chunks": 1,
    "seq_length": 512,
}
for key, expected_value in expected.items():
    actual = getattr(args, key)
    if actual != expected_value:
        raise SystemExit(
            f"QWEN35_397B contract mismatch: {key}={actual}, expected {expected_value}"
        )

audit_tokens = args.num_experts if top_k == 1 else 17
gate_result = random_gating(
    torch.randn(audit_tokens, args.num_experts),
    layer_id=0,
    topk=top_k,
)
expected_assignments = audit_tokens * top_k
if top_k == 1:
    counts = gate_result
    if sum(counts) != expected_assignments:
        raise SystemExit("top-1 routing does not execute one expert assignment/token")
else:
    counts, gather_indices = gate_result
    if (
        sum(counts) != expected_assignments
        or len(gather_indices) != expected_assignments
    ):
        raise SystemExit(
            "top-k routing does not execute num_tokens * top_k expert assignments"
        )

expert_parameters = (
    args.num_layers
    * args.num_experts
    * 3
    * args.embedding_dim
    * args.hidden_dim
)
active_expert_parameters_per_token = (
    args.num_layers
    * top_k
    * 3
    * args.embedding_dim
    * args.hidden_dim
)
dense_attention_parameters = (
    args.num_layers * 4 * args.embedding_dim * args.embedding_dim
)
synthetic_parameters = expert_parameters + dense_attention_parameters
print(
    "APTMoE contract OK: "
    f"synthetic≈{synthetic_parameters / 1e9:.3f}B params "
    f"(experts={expert_parameters / 1e9:.3f}B), "
    f"top-k={top_k}, "
    "expert-linear weights activated/token="
    f"{active_expert_parameters_per_token / 1e9:.3f}B"
)
PY
}

preflight_machine() {
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
    nvidia-smi -L >/dev/null || die "NVIDIA driver/GPU access is unavailable"

    local visible_count
    visible_count="$(
        env CUDA_VISIBLE_DEVICES="${DEVICES}" "${APTMOE_PYTHON}" -c \
            'import torch; print(torch.cuda.device_count())'
    )"
    [[ "${visible_count}" == "8" ]] || \
        die "expected 8 visible GPUs through CUDA_VISIBLE_DEVICES=${DEVICES}, got ${visible_count}"

    local active_pids
    active_pids="$(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | sed '/^[[:space:]]*$/d' || true
    )"
    [[ -z "${active_pids}" ]] || \
        die "GPU compute processes are still active (${active_pids//$'\n'/,}); clean them before running"

    local available_gb
    available_gb="$(free -g | awk '/^Mem:/ {print $7}')"
    [[ "${available_gb}" =~ ^[0-9]+$ ]] || die "could not read available host RAM"
    (( available_gb > 1800 )) || \
        die "docs/aptmoe requires available host RAM > 1800 GiB; found ${available_gb} GiB"
}

print_command() {
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
}

verify_aptmoe_contract
if [[ "${DRY_RUN}" -eq 0 ]]; then
    preflight_machine
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ROOT="${LOG_BASE}/${RUN_TIMESTAMP}_APTMOE_QWEN35_397B_BF16_${FINETUNING_TYPE^^}_TOP${TOP_K}"
mkdir -p "${RUN_ROOT}"

log "Benchmark: QWEN35_397B synthetic TransformerLM (~391B), not Qwen3.5-35B-A3B"
log "Pipeline ranks=8, batch_size=1, top-k=${TOP_K}, gini=0.3, topo=C1+G2"
log "Measured steps=${MEASURE_STEPS}, excluded warmup steps=${WARMUP_STEPS}"
log "Sequences: ${SEQUENCE_LENGTHS[*]}"
log "Results: ${RUN_ROOT}"

overall_status=0
run_index=0
for seq in "${SEQUENCE_LENGTHS[@]}"; do
    port=$((MASTER_PORT + run_index))
    (( port <= 65535 )) || die "derived torchrun port exceeds 65535"
    run_dir="${RUN_ROOT}/top${TOP_K}_seq${seq}"
    train_log="${run_dir}/train.log"
    mkdir -p "${run_dir}"

    command=(
        env
        CUDA_VISIBLE_DEVICES="${DEVICES}"
        "${TORCHRUN}"
        --nproc_per_node 8
        --master_port "${port}"
        ./main.py
        --is_moe=True
        --num_training_steps="${MEASURE_STEPS}"
        --num_warmup_steps="${WARMUP_STEPS}"
        --model_config=QWEN35_397B
        --seq_length="${seq}"
        --num_experts_per_tok="${TOP_K}"
        --gini=0.3
        --topo=C1+G2
        --pipeline=APTMoE
    )
    if [[ "${FINETUNING_TYPE}" == "lora" ]]; then
        command+=(
            --lora
            --lora_rank="${LORA_RANK}"
            --lora_alpha="${LORA_ALPHA}"
            --lora_target=all
        )
    fi
    log "seq=${seq}: one pipeline processes ${seq} tokens/step (do not multiply by 8 GPUs)"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        print_command bash -c "cd $(printf '%q' "${APTMOE_ROOT}") && $(printf '%q ' "${command[@]}")>$(printf '%q' "${train_log}") 2>&1"
        run_index=$((run_index + 1))
        continue
    fi

    set +e
    (
        cd "${APTMOE_ROOT}"
        "${command[@]}"
    ) >"${train_log}" 2>&1
    exit_code=$?
    set -e
    printf '%s\n' "${exit_code}" > "${run_dir}/exit_code.txt"

    if (( exit_code == 0 )); then
        grep -E 'model #params|training elapsed time per step|training throughput' \
            "${train_log}" | tail -n 3 || true
    else
        warn "seq=${seq} failed with exit code ${exit_code}; see ${train_log}"
        tail -n 30 "${train_log}" >&2 || true
        overall_status=1
        if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
            break
        fi
    fi
    run_index=$((run_index + 1))
done

exit "${overall_status}"
