#!/usr/bin/env bash
# Detach Qwen3.5-122B then GLM-4.5-Air PyTorch TORCH full-FT sweeps.
# Survives terminal hangup. Does not touch /mnt/data2/xmy/venv.

set -Eeuo pipefail

export TZ="${FFT_TIMEZONE:-Asia/Shanghai}"

FFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="${FFT_DETACH_LOG_BASE:-${FFT_ROOT}/test_log}"
SEQ_LENGTHS="${FFT_TORCH_SEQ_LENGTHS:-1024,512,256,128,64,32}"
STEPS="${FFT_TORCH_STEPS:-15}"
WARMUP_STEPS="${FFT_TORCH_WARMUP_STEPS:-5}"
DEVICES="${FFT_TORCH_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_STAMP="$(date '+%Y%m%d_%H%M%S')"
MASTER_DIR="${LOG_BASE}/${MASTER_STAMP}_PYTORCH_TORCH_QWEN_THEN_GLM"
MASTER_LOG="${MASTER_DIR}/master.log"
PID_FILE="${MASTER_DIR}/master.pid"

mkdir -p "${MASTER_DIR}"

run_chain() {
    echo "[$(date '+%F %T')] start Qwen3.5-122B-A10B seq=${SEQ_LENGTHS}"
    bash "${FFT_ROOT}/Qwen3.5-122B-A10B/run_finetune_perf_test_pytorch_torch.sh" \
        --seq-lengths "${SEQ_LENGTHS}" \
        --steps "${STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --devices "${DEVICES}" \
        --continue-on-error
    echo "[$(date '+%F %T')] finished Qwen3.5-122B-A10B"

    echo "[$(date '+%F %T')] start GLM-4.5-Air seq=${SEQ_LENGTHS}"
    bash "${FFT_ROOT}/GLM-4.5-Air/run_finetune_perf_test_pytorch_torch.sh" \
        --seq-lengths "${SEQ_LENGTHS}" \
        --steps "${STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --devices "${DEVICES}" \
        --continue-on-error
    echo "[$(date '+%F %T')] finished GLM-4.5-Air"
    echo "[$(date '+%F %T')] all done"
}

if [[ "${FFT_TORCH_CHAIN_INNER:-0}" == "1" ]]; then
    run_chain
    exit 0
fi

# First invocation: detach into a new session so closing the terminal
# cannot deliver SIGHUP to the sweep.
setsid env \
    FFT_TORCH_CHAIN_INNER=1 \
    FFT_DETACH_LOG_BASE="${LOG_BASE}" \
    FFT_TORCH_SEQ_LENGTHS="${SEQ_LENGTHS}" \
    FFT_TORCH_STEPS="${STEPS}" \
    FFT_TORCH_WARMUP_STEPS="${WARMUP_STEPS}" \
    FFT_TORCH_DEVICES="${DEVICES}" \
    bash "${BASH_SOURCE[0]}" \
    </dev/null >"${MASTER_LOG}" 2>&1 &
echo $! > "${PID_FILE}"

cat <<EOF
detached pid=$(cat "${PID_FILE}")
master log: ${MASTER_LOG}
pid file:   ${PID_FILE}

follow:
  tail -f ${MASTER_LOG}

stop:
  kill \$(cat ${PID_FILE})
EOF
