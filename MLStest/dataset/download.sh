#!/usr/bin/env bash
# Download Nemotron SFT datasets into MLStest/dataset/.
#
# Usage:
#   bash dataset/download.sh              # all
#   bash dataset/download.sh cuda swe
#   HTTP_PROXY=http://host:port bash dataset/download.sh cpp
#
# Env:
#   DATASET_ROOT  default: <repo>/dataset
#   HF            path to hf CLI (default: hf on PATH)
#   HTTP_PROXY / HTTPS_PROXY / ALL_PROXY  optional

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}}"
HF="${HF:-$(command -v hf || true)}"
if [[ -z "${HF}" ]]; then
  HF="$(command -v huggingface-cli || true)"
fi
if [[ -z "${HF}" ]]; then
  echo "[error] hf / huggingface-cli not found on PATH" >&2
  exit 1
fi

if [[ -n "${HTTP_PROXY:-}" && -z "${HTTPS_PROXY:-}" ]]; then
  export HTTPS_PROXY="${HTTP_PROXY}"
fi
if [[ -n "${HTTPS_PROXY:-}" && -z "${HTTP_PROXY:-}" ]]; then
  export HTTP_PROXY="${HTTPS_PROXY}"
fi

TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
  TASKS=(all)
fi

download_cuda() {
  echo "[download] Nemotron-SFT-CUDA-v1 -> ${DATASET_ROOT}/Nemotron-SFT-CUDA-v1"
  "${HF}" download nvidia/Nemotron-SFT-CUDA-v1 --repo-type dataset \
    --local-dir "${DATASET_ROOT}/Nemotron-SFT-CUDA-v1"
}

download_swe() {
  echo "[download] Nemotron-SFT-SWE-v3 -> ${DATASET_ROOT}/Nemotron-SFT-SWE-v3"
  "${HF}" download nvidia/Nemotron-SFT-SWE-v3 --repo-type dataset \
    --local-dir "${DATASET_ROOT}/Nemotron-SFT-SWE-v3"
}

download_cpp() {
  echo "[download] Competitive Programming C++ subset -> ${DATASET_ROOT}/Nemotron-SFT-Competitive-Programming-v2"
  "${HF}" download nvidia/Nemotron-SFT-Competitive-Programming-v2 --repo-type dataset \
    --include "data/competitive_programming_cpp_*.jsonl" "README.md" ".gitattributes" \
    --local-dir "${DATASET_ROOT}/Nemotron-SFT-Competitive-Programming-v2"
}

run_one() {
  case "$1" in
    cuda) download_cuda ;;
    swe) download_swe ;;
    cpp) download_cpp ;;
    all)
      download_cuda
      download_swe
      download_cpp
      ;;
    *)
      echo "Usage: $0 [cuda|swe|cpp|all]..." >&2
      exit 2
      ;;
  esac
}

for t in "${TASKS[@]}"; do
  run_one "${t}"
done

echo "[download] done"
du -sh "${DATASET_ROOT}"/Nemotron-* 2>/dev/null || true
