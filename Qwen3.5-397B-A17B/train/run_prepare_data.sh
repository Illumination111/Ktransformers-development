#!/usr/bin/env bash
# Prepare Nemotron datasets into LLaMA-Factory openai jsonl.
#
# Usage:
#   bash run_prepare_data.sh              # cuda + swe + cpp
#   bash run_prepare_data.sh cuda         # one task
#   bash run_prepare_data.sh swe cpp
#   MAX_SAMPLES=1000 bash run_prepare_data.sh cuda

set -euo pipefail

# Log / filename timestamps use China local time (script-only; host TZ unchanged).
export MLS_TIMEZONE="${MLS_TIMEZONE:-Asia/Shanghai}"
export TZ="${MLS_TIMEZONE}"

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${TRAIN_DIR}/configs/default_env.sh"

PREPARE_PY="${TRAIN_DIR}/scripts/prepare_nemotron_datasets.py"
TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
  TASKS=(all)
fi

PYTHON_BIN="${MLS_CONDA_PYTHON}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "${DATASET_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/prepare_$(date +%Y%m%d_%H%M%S).log"

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "[prepare] dataset_root=${DATASET_ROOT}"
echo "[prepare] output_dir=${DATASET_DIR}"
echo "[prepare] tasks=${TASKS[*]}"
echo "[prepare] log=${LOG_FILE}"

set -x
"${PYTHON_BIN}" "${PREPARE_PY}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${DATASET_DIR}" \
  --tasks "${TASKS[@]}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
set +x

echo "[prepare] done"
