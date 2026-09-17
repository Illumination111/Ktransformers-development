#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$(cd "${SCRIPT_DIR}/../Qwen3.5-35B-A3B" && pwd)"

export FFT_MODEL_PATH="${FFT_MODEL_PATH:-/mnt/data2/models/Qwen3.5-122B-A10B}"
export FFT_MODEL_DISPLAY_NAME="Qwen3.5-122B-A10B"
export FFT_LOG_BASE="${FFT_LOG_BASE:-${SCRIPT_DIR}/test_log}"
export FFT_RUNTIME_ROOT="${FFT_RUNTIME_ROOT:-/mnt/data2/wbw/fft_runtime/qwen35_122b/aptmoe}"
export FFT_AGGREGATOR="${SCRIPT_DIR}/aggregate_sweep_results.py"
export FFT_APTMOE_ENTRYPOINT="${SCRIPT_DIR}/aptmoe_qwen35_122b_proxy_train.py"
export FFT_APTMOE_SWEEP_ENTRYPOINT="${SCRIPT_DIR}/aptmoe_qwen35_122b_sweep.py"
export FFT_APTMOE_PROXY_TAG="qwen35_122b"
export FFT_APTMOE_PROXY_ARCHITECTURE="qwen35_122b_component_isomorphic"
export FFT_APTMOE_MODEL_LOAD_ARCHITECTURE="Qwen35_122BComponentIsomorphicAPTMoEProxy"
# Keep formal 122B artifacts separate from the 35B route/lookup set.  The
# shared runner already defaults all smoke fallbacks to disabled; these paths
# make a formal run fail early until 122B-specific artifacts are profiled.
export FFT_APTMOE_ROUTE_ROOT="${FFT_APTMOE_ROUTE_ROOT:-${SCRIPT_DIR}/../APTMoE-simulate/routes/qwen35_122b}"
export FFT_APTMOE_LOOKUP_ROOT="${FFT_APTMOE_LOOKUP_ROOT:-${SCRIPT_DIR}/../APTMoE-simulate/lookups/qwen35_122b}"

exec bash "${SHARED_DIR}/run_finetune_perf_sweep_bf16_common.sh" aptmoe "$@"
