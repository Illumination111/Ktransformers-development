#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FFT_CANONICAL_LAUNCHER="$(basename "${BASH_SOURCE[0]}")"
exec bash "${SCRIPT_DIR}/run_finetune_perf_sweep_bf16_external_common.sh" deepspeed "$@"
