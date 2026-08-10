#!/usr/bin/env bash
# Compatibility name retained for existing callers. The old AMXINT4 phase
# stress test has been superseded by the native-BF16 sequence sweep.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_finetune_perf_test_bf16_ktransformers.sh" "$@"
