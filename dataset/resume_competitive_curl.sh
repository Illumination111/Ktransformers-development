#!/usr/bin/env bash
# Resume Competitive C++ jsonl via curl (useful when hf through proxy is flaky).
#
# Env:
#   DATASET_ROOT, HTTP_PROXY / HTTPS_PROXY (optional)
#   HF_TOKEN (optional, for gated / rate-limited downloads)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}}"
DIR="${DATASET_ROOT}/Nemotron-SFT-Competitive-Programming-v2"
DATA="${DIR}/data"
LOG="${DATASET_ROOT}/competitive_curl.log"
PROXY="${HTTPS_PROXY:-${HTTP_PROXY:-${https_proxy:-${http_proxy:-}}}}"

mkdir -p "${DATA}"

while read -r p; do
  [[ -n "$p" && "$p" != "$$" ]] && kill "$p" 2>/dev/null || true
done < <(pgrep -f '/hf download nvidia/Nemotron-SFT-Competitive-Programming-v2' || true)
sleep 1

BASE_URL="https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2/resolve/main/data"
AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
fi
CURL_PROXY=()
if [[ -n "${PROXY}" ]]; then
  CURL_PROXY=(-x "${PROXY}")
fi

download_one() {
  local name="$1"
  local dest="${DATA}/${name}"
  local url="${BASE_URL}/${name}"
  echo "$(date -Is) curl -C - ${name}" | tee -a "${LOG}"
  curl -L --fail --retry 20 --retry-delay 5 -C - \
    "${CURL_PROXY[@]}" \
    "${AUTH[@]}" \
    -o "${dest}" \
    "${url}" 2>&1 | tee -a "${LOG}"
  ls -lh "${dest}" | tee -a "${LOG}"
}

download_one competitive_programming_cpp_00.jsonl
download_one competitive_programming_cpp_01.jsonl
echo "$(date -Is) competitive curl done" | tee -a "${LOG}"
