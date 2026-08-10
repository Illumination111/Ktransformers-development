#!/usr/bin/env bash
# Resume Competitive C++ jsonl via aria2c multi-connection download.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}}"
DATA="${DATASET_ROOT}/Nemotron-SFT-Competitive-Programming-v2/data"
LOG="${DATASET_ROOT}/competitive_aria2.log"
PROXY="${HTTPS_PROXY:-${HTTP_PROXY:-${https_proxy:-${http_proxy:-}}}}"

mkdir -p "${DATA}"
cd "${DATA}"

expect00=23691055037
expect01=23690693444

download_one() {
  local name="$1" expect="$2"
  local url="https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2/resolve/main/data/${name}"
  local attempt=1
  local aria_proxy=()
  if [[ -n "${PROXY}" ]]; then
    aria_proxy=(--all-proxy="${PROXY}")
  fi
  while true; do
    local cur=0
    [[ -f "${name}" ]] && cur=$(stat -c%s "${name}")
    if (( cur >= expect )); then
      echo "$(date -Is) ${name} DONE ${cur}"
      return 0
    fi
    echo "$(date -Is) ${name} attempt=${attempt} have=${cur} expect=${expect}"
    aria2c -c \
      "${aria_proxy[@]}" \
      -x 8 -s 8 -k 1M \
      --max-tries=0 --retry-wait=5 \
      --timeout=60 --connect-timeout=30 \
      --lowest-speed-limit=10K \
      --file-allocation=none \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      -o "${name}" \
      "${url}" >>"${LOG}" 2>&1 || true
    cur=$(stat -c%s "${name}" 2>/dev/null || echo 0)
    if (( cur >= expect )); then
      echo "$(date -Is) ${name} DONE ${cur}"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 3
  done
}

download_one competitive_programming_cpp_00.jsonl "${expect00}"
download_one competitive_programming_cpp_01.jsonl "${expect01}"
