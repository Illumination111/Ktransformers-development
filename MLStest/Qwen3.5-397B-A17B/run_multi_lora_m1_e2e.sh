#!/usr/bin/env bash
# End-to-end M1 multi-LoRA test for Qwen3.5-397B-A17B:
# start server -> wait ready -> client smoke -> stop server.

set -Eeuo pipefail

export TZ="${MLS_TIMEZONE:-Asia/Shanghai}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/configs/default_env.sh"

SERVE_SCRIPT="${SCRIPT_DIR}/run_multi_lora_m1_serve.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/run_multi_lora_m1_client.sh"

DRY_RUN=0
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"
SERVER_PID=""
RUN_DIR=""
PASS_ARGS=()

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Orchestrates:
  1) background M1 server (run_multi_lora_m1_serve.sh)
  2) wait until HTTP ready
  3) client smoke (run_multi_lora_m1_client.sh)
  4) terminate server

All unknown flags are forwarded to the serve script. Client-specific flags:
  --adapters, --rounds, --max-tokens, --prompt, --timeout-sec

Required (same as serve):
  --kt-weight-path PATH
  --lora-paths L0=/path/L0,L1=/path/L1

Extra options:
  --ready-timeout-sec N     Default: ${READY_TIMEOUT_SEC}
  --dry-run                 Forward dry-run to serve+client; do not start process
  -h, --help

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

CLIENT_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ready-timeout-sec) need_value "$1" "$#"; READY_TIMEOUT_SEC="$2"; shift ;;
        --adapters|--rounds|--max-tokens|--temperature|--prompt|--timeout-sec)
            need_value "$1" "$#"
            CLIENT_ARGS+=("$1" "$2")
            # also keep --lora-paths for serve if present via PASS_ARGS
            if [[ "$1" == "--adapters" ]]; then
                :
            fi
            shift
            ;;
        --lora-paths)
            need_value "$1" "$#"
            PASS_ARGS+=("$1" "$2")
            CLIENT_ARGS+=("$1" "$2")
            shift
            ;;
        --host|--port|--served-model-name)
            need_value "$1" "$#"
            PASS_ARGS+=("$1" "$2")
            CLIENT_ARGS+=("$1" "$2")
            # keep local copies for readiness probe
            case "$1" in
                --host) HOST="$2" ;;
                --port) PORT="$2" ;;
                --served-model-name) SERVED_MODEL_NAME="$2" ;;
            esac
            shift
            ;;
        --dry-run) DRY_RUN=1; PASS_ARGS+=(--dry-run); CLIENT_ARGS+=(--dry-run) ;;
        -h|--help) usage; exit 0 ;;
        *)
            PASS_ARGS+=("$1")
            ;;
    esac
    shift
done

cleanup() {
    local ec=$?
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        printf 'Stopping server pid=%s\n' "${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${RUN_DIR}" ]]; then
        printf '%s\n' "${ec}" > "${RUN_DIR}/exit_code.txt"
    fi
    exit "${ec}"
}
trap cleanup EXIT INT TERM

RUN_ID="$(date +%Y%m%d_%H%M%S)_m1_e2e"
RUN_DIR="${LOG_BASE}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

if (( DRY_RUN )); then
    printf '[dry-run] serve:\n'
    bash "${SERVE_SCRIPT}" "${PASS_ARGS[@]}" --log-base "${RUN_DIR}" --dry-run
    printf '[dry-run] client:\n'
    bash "${CLIENT_SCRIPT}" "${CLIENT_ARGS[@]}" --log-dir "${RUN_DIR}/client" --dry-run
    exit 0
fi

printf 'Starting server in background; e2e dir=%s\n' "${RUN_DIR}"
bash "${SERVE_SCRIPT}" "${PASS_ARGS[@]}" --log-base "${RUN_DIR}" --skip-ready-wait \
    > "${RUN_DIR}/e2e_serve_stdout.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" > "${RUN_DIR}/server.pid"

BASE_URL="http://${HOST}:${PORT}"
printf 'Waiting for server ready at %s (timeout %ss)\n' "${BASE_URL}" "${READY_TIMEOUT_SEC}"
deadline=$((SECONDS + READY_TIMEOUT_SEC))
ready=0
while (( SECONDS < deadline )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        printf 'ERROR: server exited early; see %s/e2e_serve_stdout.log and */server.log\n' \
            "${RUN_DIR}" >&2
        exit 1
    fi
    if curl -sS --max-time 2 "${BASE_URL}/v1/models" >/dev/null 2>&1 \
        || curl -sS --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 5
done
if (( ! ready )); then
    printf 'ERROR: server not ready within %ss\n' "${READY_TIMEOUT_SEC}" >&2
    exit 1
fi
printf 'Server ready.\n'

bash "${CLIENT_SCRIPT}" "${CLIENT_ARGS[@]}" --log-dir "${RUN_DIR}/client"
printf 'E2E PASSED\n' | tee "${RUN_DIR}/summary.md"
