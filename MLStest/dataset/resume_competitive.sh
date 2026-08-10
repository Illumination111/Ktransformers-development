#!/usr/bin/env bash
# Resume Competitive Programming C++ download via Hugging Face CLI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}}"
HF="${HF:-$(command -v hf || true)}"
if [[ -z "${HF}" ]]; then
  echo "[error] hf not found; set HF=/path/to/hf" >&2
  exit 1
fi

if [[ -n "${HTTP_PROXY:-}" && -z "${HTTPS_PROXY:-}" ]]; then
  export HTTPS_PROXY="${HTTP_PROXY}"
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

DIR="${DATASET_ROOT}/Nemotron-SFT-Competitive-Programming-v2"
LOG="${DATASET_ROOT}/competitive_download.log"

mapfile -t PIDS < <(pgrep -f '/hf download nvidia/Nemotron-SFT-Competitive-Programming-v2' || true)
for p in "${PIDS[@]:-}"; do
  if [[ -n "$p" && "$p" != "$$" ]]; then
    echo "kill $p"
    kill "$p" 2>/dev/null || true
  fi
done
sleep 2

{
  echo "==== $(date -Is) competitive resume via hf ===="
  "${HF}" download nvidia/Nemotron-SFT-Competitive-Programming-v2 --repo-type dataset \
    --include "data/competitive_programming_cpp_*.jsonl" "README.md" ".gitattributes" \
    --local-dir "${DIR}"
  echo "exit:$?"
  du -sh "${DIR}"
  ls -lh "${DIR}/data" || true
} 2>&1 | tee -a "${LOG}"
