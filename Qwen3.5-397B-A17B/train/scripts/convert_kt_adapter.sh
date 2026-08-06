#!/usr/bin/env bash
# Convert a KT LoRA run directory into a sglang-kt merged composite adapter.
#
# Usage:
#   bash scripts/convert_kt_adapter.sh <run_dir> <adapter_name>
#   bash scripts/convert_kt_adapter.sh \
#     /mnt/data2/wbw/MLStest/Qwen3.5-397B-A17B/train/runs/nemotron_cuda \
#     cuda

set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${TRAIN_DIR}/configs/default_env.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <kt_run_dir> <adapter_name>" >&2
  exit 2
fi

RUN_DIR="$(realpath "$1")"
ADAPTER_NAME="$2"
OUT_DIR="${ADAPTER_ROOT}/${ADAPTER_NAME}"

if [[ ! -f "${RUN_DIR}/fused_expert_lora.safetensors" ]]; then
  echo "[error] missing fused_expert_lora.safetensors under ${RUN_DIR}" >&2
  echo "        (KT LoRA SFT should write this next to adapter_model.safetensors)" >&2
  exit 1
fi

mkdir -p "${ADAPTER_ROOT}"
PYTHON_BIN="${MLS_CONDA_PYTHON}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

echo "[convert] input=${RUN_DIR}"
echo "[convert] output=${OUT_DIR}"
echo "[convert] base=${MODEL_PATH}"

"${PYTHON_BIN}" "${CONVERT_KT_SCRIPT}" \
  "${RUN_DIR}" \
  "${OUT_DIR}" \
  --base-model-name-or-path "${MODEL_PATH}" \
  --overwrite

echo "[convert] done -> ${OUT_DIR}"
ls -lh "${OUT_DIR}" || true
