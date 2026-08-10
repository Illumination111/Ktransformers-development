#!/usr/bin/env bash
# Default Qwen3.5-122B-A10B entrypoint: KTransformers AMXBF16, text-only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash \
    "${SCRIPT_DIR}/run_finetune_perf_test_bf16_ktransformers.sh" "$@"
