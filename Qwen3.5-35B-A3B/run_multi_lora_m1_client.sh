#!/usr/bin/env bash
# Client smoke tests for Qwen3.5-35B-A3B multi-LoRA serving (M1).
# Exercises base-only naming and alternating adapters (one adapter per request).

set -Eeuo pipefail

export TZ="${MLS_TIMEZONE:-Asia/Shanghai}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/configs/default_env.sh"

ADAPTERS_CSV="${ADAPTERS_CSV:-}"
ROUNDS="${ROUNDS:-2}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
PROMPT="${PROMPT:-用一句话解释什么是 LoRA。}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
LOG_DIR=""
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Hit a running M1 multi-LoRA server and verify:
  1) base model name responds
  2) each registered adapter name responds (model=served:adapter)
  3) adapters can be alternated across sequential requests (M1 batch boundary switch)

Options:
  --host ADDR               Default: ${HOST}
  --port N                  Default: ${PORT}
  --served-model-name NAME  Default: ${SERVED_MODEL_NAME}
  --adapters LIST           Comma-separated adapter names (e.g. L0,L1)
                            If omitted, derived from --lora-paths / LORA_PATHS env
  --lora-paths LIST         name=path,... used only to derive adapter names
  --rounds N                Alternation rounds (default: ${ROUNDS})
  --max-tokens N            Default: ${MAX_TOKENS}
  --temperature F           Default: ${TEMPERATURE}
  --prompt TEXT             User prompt
  --timeout-sec N           Per-request curl max-time (default: ${TIMEOUT_SEC})
  --log-dir PATH            Write request/response JSON under this dir
  --dry-run                 Print planned requests only
  -h, --help

See: ${MLS_ROOT}/docs/task_bash_Qwen3.5-35B-A3B.md
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
        --host) need_value "$1" "$#"; HOST="$2"; shift ;;
        --port) need_value "$1" "$#"; PORT="$2"; shift ;;
        --served-model-name) need_value "$1" "$#"; SERVED_MODEL_NAME="$2"; shift ;;
        --adapters) need_value "$1" "$#"; ADAPTERS_CSV="$2"; shift ;;
        --lora-paths) need_value "$1" "$#"; LORA_PATHS="$2"; shift ;;
        --rounds) need_value "$1" "$#"; ROUNDS="$2"; shift ;;
        --max-tokens) need_value "$1" "$#"; MAX_TOKENS="$2"; shift ;;
        --temperature) need_value "$1" "$#"; TEMPERATURE="$2"; shift ;;
        --prompt) need_value "$1" "$#"; PROMPT="$2"; shift ;;
        --timeout-sec) need_value "$1" "$#"; TIMEOUT_SEC="$2"; shift ;;
        --log-dir) need_value "$1" "$#"; LOG_DIR="$2"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ -z "${ADAPTERS_CSV}" ]]; then
    if [[ -z "${LORA_PATHS}" ]]; then
        printf 'ERROR: provide --adapters L0,L1 or --lora-paths L0=/p0,L1=/p1\n' >&2
        exit 2
    fi
    names=()
    IFS=',' read -r -a pairs <<< "${LORA_PATHS}"
    for pair in "${pairs[@]}"; do
        pair="${pair#"${pair%%[![:space:]]*}"}"
        [[ -n "${pair}" ]] || continue
        names+=("${pair%%=*}")
    done
    ADAPTERS_CSV="$(IFS=','; echo "${names[*]}")"
fi

IFS=',' read -r -a ADAPTERS <<< "${ADAPTERS_CSV}"
if ((${#ADAPTERS[@]} < 1)); then
    printf 'ERROR: empty adapter list\n' >&2
    exit 2
fi

BASE_URL="http://${HOST}:${PORT}"
if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="${LOG_BASE}/$(date +%Y%m%d_%H%M%S)_m1_client"
fi
mkdir -p "${LOG_DIR}"

PYTHON_BIN="${MLS_CONDA_PYTHON}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

chat_once() {
    local model="$1"
    local tag="$2"
    local body out http_code
    body="$(
        MODEL_NAME="${model}" PROMPT_TEXT="${PROMPT}" MAX_TOKENS="${MAX_TOKENS}" TEMPERATURE="${TEMPERATURE}" \
        "${PYTHON_BIN}" - <<'PY'
import json, os
print(json.dumps({
    "model": os.environ["MODEL_NAME"],
    "messages": [{"role": "user", "content": os.environ["PROMPT_TEXT"]}],
    "temperature": float(os.environ["TEMPERATURE"]),
    "max_tokens": int(os.environ["MAX_TOKENS"]),
}, ensure_ascii=False))
PY
    )"
    out="${LOG_DIR}/${tag}.json"
    if (( DRY_RUN )); then
        printf '[dry-run] POST %s/v1/chat/completions model=%s -> %s\n' \
            "${BASE_URL}" "${model}" "${out}"
        printf '%s\n' "${body}" > "${out}.request.json"
        return 0
    fi
    printf 'REQUEST model=%s tag=%s\n' "${model}" "${tag}"
    http_code="$(
        curl -sS -o "${out}" -w '%{http_code}' \
            --max-time "${TIMEOUT_SEC}" \
            -H 'Content-Type: application/json' \
            -d "${body}" \
            "${BASE_URL}/v1/chat/completions"
    )"
    printf '%s\n' "${body}" > "${out}.request.json"
    if [[ "${http_code}" != "200" ]]; then
        printf 'FAIL tag=%s http=%s\n' "${tag}" "${http_code}" >&2
        printf 'response saved at %s\n' "${out}" >&2
        return 1
    fi
    "${PYTHON_BIN}" - "${out}" <<'PY'
import json, sys
path = sys.argv[1]
obj = json.load(open(path, encoding="utf-8"))
choices = obj.get("choices") or []
if not choices:
    raise SystemExit(f"no choices in {path}")
msg = choices[0].get("message") or {}
content = msg.get("content") or ""
if not str(content).strip():
    raise SystemExit(f"empty content in {path}")
print("OK content_chars=", len(content))
print(content[:240].replace("\n", " "))
PY
}

# 1) Health (optional soft check)
if (( ! DRY_RUN )); then
    if ! curl -sS --max-time 5 "${BASE_URL}/health" >/dev/null 2>&1 \
        && ! curl -sS --max-time 5 "${BASE_URL}/v1/models" >/dev/null 2>&1; then
        printf 'WARN: health/models probe failed; continuing anyway\n' >&2
    fi
fi

FAIL=0

# 2) Base-only model name
if ! chat_once "${SERVED_MODEL_NAME}" "01_base"; then
    FAIL=1
fi

# 3) Each adapter once
idx=2
for adapter in "${ADAPTERS[@]}"; do
    adapter="${adapter#"${adapter%%[![:space:]]*}"}"
    adapter="${adapter%"${adapter##*[![:space:]]}"}"
    [[ -n "${adapter}" ]] || continue
    tag="$(printf '%02d_adapter_%s' "${idx}" "${adapter}")"
    if ! chat_once "${SERVED_MODEL_NAME}:${adapter}" "${tag}"; then
        FAIL=1
    fi
    idx=$((idx + 1))
done

# 4) Alternation rounds (M1: sequential switch across requests)
for ((r = 1; r <= ROUNDS; r++)); do
    for adapter in "${ADAPTERS[@]}"; do
        adapter="${adapter#"${adapter%%[![:space:]]*}"}"
        adapter="${adapter%"${adapter##*[![:space:]]}"}"
        [[ -n "${adapter}" ]] || continue
        tag="$(printf 'alt_r%02d_%s' "${r}" "${adapter}")"
        if ! chat_once "${SERVED_MODEL_NAME}:${adapter}" "${tag}"; then
            FAIL=1
        fi
    done
done

SUMMARY="${LOG_DIR}/summary.md"
{
    echo "# M1 client smoke summary"
    echo
    echo "- base_url: \`${BASE_URL}\`"
    echo "- served_model_name: \`${SERVED_MODEL_NAME}\`"
    echo "- adapters: \`${ADAPTERS_CSV}\`"
    echo "- rounds: ${ROUNDS}"
    echo "- result: $([[ ${FAIL} -eq 0 ]] && echo PASS || echo FAIL)"
} > "${SUMMARY}"

if (( FAIL )); then
    printf 'CLIENT SMOKE FAILED; see %s\n' "${LOG_DIR}" >&2
    exit 1
fi
printf 'CLIENT SMOKE PASSED; logs at %s\n' "${LOG_DIR}"
