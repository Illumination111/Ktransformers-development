#!/usr/bin/env bash
# Resume SWE + Competitive C++ downloads into DATASET_ROOT.
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

LOG="${DATASET_ROOT}/download.log"
exec > >(tee -a "${LOG}") 2>&1
echo "==== $(date -Is) resume start ===="

echo "=== SWE-v3 resume ==="
"${HF}" download nvidia/Nemotron-SFT-SWE-v3 --repo-type dataset \
  --local-dir "${DATASET_ROOT}/Nemotron-SFT-SWE-v3"
echo "SWE exit:$?"
du -sh "${DATASET_ROOT}/Nemotron-SFT-SWE-v3"

echo "=== Competitive C++ subset ==="
"${HF}" download nvidia/Nemotron-SFT-Competitive-Programming-v2 --repo-type dataset \
  --include "data/competitive_programming_cpp_*.jsonl" "README.md" ".gitattributes" \
  --local-dir "${DATASET_ROOT}/Nemotron-SFT-Competitive-Programming-v2"
echo "Competitive exit:$?"
du -sh "${DATASET_ROOT}/Nemotron-SFT-Competitive-Programming-v2"
ls -lh "${DATASET_ROOT}/Nemotron-SFT-Competitive-Programming-v2/data" || true

echo "==== $(date -Is) ALL DONE ===="
du -sh "${DATASET_ROOT}"/*
