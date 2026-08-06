#!/usr/bin/env bash
# Concurrent multi-adapter client smoke for M2 (N-way sub-agent style).
# Fires one request per adapter in parallel; each keeps its own response.
# Does NOT merge adapter outputs into a single user answer.

set -Eeuo pipefail

export TZ="${MLS_TIMEZONE:-Asia/Shanghai}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/configs/default_env.sh"

ADAPTERS_CSV="${ADAPTERS_CSV:-}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
PROMPT="${PROMPT:-用一句话解释什么是 LoRA。}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
LOG_DIR=""
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Concurrent N-adapter smoke against a running server (prefer
--kt-lora-dispatch grouped so distinct adapters can share a forward batch).

Options:
  --host ADDR
  --port N
  --served-model-name NAME
  --adapters LIST           Comma-separated adapter names (e.g. cuda,swe,cpp)
  --lora-paths LIST         name=path,... used only to derive adapter names
  --max-tokens N
  --temperature F
  --prompt TEXT
  --timeout-sec N
  --log-dir PATH
  --dry-run
  -h, --help
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
        printf 'ERROR: provide --adapters or --lora-paths\n' >&2
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
if ((${#ADAPTERS[@]} < 2)); then
    printf 'ERROR: concurrent smoke needs >=2 adapters, got %d\n' "${#ADAPTERS[@]}" >&2
    exit 2
fi

BASE_URL="http://${HOST}:${PORT}"
if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="${LOG_BASE}/$(date +%Y%m%d_%H%M%S)_m2_concurrent_client"
fi
mkdir -p "${LOG_DIR}"

PYTHON_BIN="${MLS_CONDA_PYTHON}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

launch_one() {
    local adapter="$1"
    local tag="$2"
    local model="${SERVED_MODEL_NAME}:${adapter}"
    local body out
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
    out="${LOG_DIR}/${tag}"
    printf '%s\n' "${body}" > "${out}.request.json"
    if (( DRY_RUN )); then
        printf '[dry-run] POST model=%s -> %s.json\n' "${model}" "${out}"
        return 0
    fi
    (
        http_code="$(
            curl -sS -o "${out}.json" -w '%{http_code}' \
                --max-time "${TIMEOUT_SEC}" \
                -H 'Content-Type: application/json' \
                -d "${body}" \
                "${BASE_URL}/v1/chat/completions"
        )"
        printf '%s\n' "${http_code}" > "${out}.http"
    ) &
    echo $!
}

PIDS=()
TAGS=()
idx=1
for adapter in "${ADAPTERS[@]}"; do
    adapter="${adapter#"${adapter%%[![:space:]]*}"}"
    adapter="${adapter%"${adapter##*[![:space:]]}"}"
    [[ -n "${adapter}" ]] || continue
    tag="$(printf 'concurrent_%02d_%s' "${idx}" "${adapter}")"
    TAGS+=("${tag}")
    if (( DRY_RUN )); then
        launch_one "${adapter}" "${tag}" || true
    else
        pid="$(launch_one "${adapter}" "${tag}")"
        PIDS+=("${pid}")
        printf 'LAUNCHED adapter=%s pid=%s tag=%s\n' "${adapter}" "${pid}" "${tag}"
    fi
    idx=$((idx + 1))
done

FAIL=0
if (( ! DRY_RUN )); then
    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            FAIL=1
        fi
    done
    for tag in "${TAGS[@]}"; do
        http_code="$(cat "${LOG_DIR}/${tag}.http" 2>/dev/null || echo missing)"
        if [[ "${http_code}" != "200" ]]; then
            printf 'FAIL tag=%s http=%s\n' "${tag}" "${http_code}" >&2
            FAIL=1
            continue
        fi
        if ! "${PYTHON_BIN}" - "${LOG_DIR}/${tag}.json" <<'PY'
import json, sys
path = sys.argv[1]
obj = json.load(open(path, encoding="utf-8"))
choices = obj.get("choices") or []
if not choices:
    raise SystemExit(f"no choices in {path}")
content = ((choices[0].get("message") or {}).get("content")) or ""
if not str(content).strip():
    raise SystemExit(f"empty content in {path}")
print("OK", path, "chars=", len(content))
PY
        then
            FAIL=1
        fi
    done
fi

SUMMARY="${LOG_DIR}/summary.md"
{
    echo "# M2 concurrent client smoke summary"
    echo
    echo "- base_url: \`${BASE_URL}\`"
    echo "- adapters: \`${ADAPTERS_CSV}\`"
    echo "- n_requests: ${#TAGS[@]}"
    echo "- result: $([[ ${FAIL} -eq 0 ]] && echo PASS || echo FAIL)"
    echo
    echo "Each request binds one LoRA; responses are not merged by serving."
} > "${SUMMARY}"

if (( FAIL )); then
    printf 'M2 CONCURRENT CLIENT FAILED; see %s\n' "${LOG_DIR}" >&2
    exit 1
fi
printf 'M2 CONCURRENT CLIENT PASSED; logs at %s\n' "${LOG_DIR}"
