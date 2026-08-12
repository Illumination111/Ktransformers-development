#!/usr/bin/env bash
# Twenty-step Qwen3-VL LoRA stability run using the same per-step contracts.

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_vlm_lora_smoke.sh" \
    --max-steps 20 \
    --log-base "${SCRIPT_DIR}/formal_test_log" \
    "$@"
