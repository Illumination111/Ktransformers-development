#!/usr/bin/env bash
# Multi-step, full-modality Qwen3.5-397B-A17B VLM LoRA stability test.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$(cd "${SCRIPT_DIR}/../Qwen3.5-122B-A10B" && pwd)"

export VLM_MODEL_PATH="${VLM_MODEL_PATH:-/mnt/data2/models/Qwen3.5-397B-A17B}"
export VLM_LLAMA_FACTORY_DIR="${VLM_LLAMA_FACTORY_DIR:-/mnt/data2/wbw/LlamaFactory-vlm-pr}"
export VLM_KT_SOURCE_DIR="${VLM_KT_SOURCE_DIR:-/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel}"
export VLM_FORMAL_LOG_BASE="${VLM_FORMAL_LOG_BASE:-${SCRIPT_DIR}/formal_test_log}"
export VLM_LORA_SCOPE="${VLM_LORA_SCOPE:-all}"
export VLM_EXPECTED_LAYERS=60
export VLM_EXPECTED_EXPERTS=512
export VLM_EXPECTED_TOP_K=10
export VLM_EXPECTED_KT_WRAPPERS=60

exec bash "${SHARED_DIR}/run_vlm_lora_formal.sh" "$@"
