# Shared defaults for Qwen3.5-397B-A17B multi-LoRA serving (M1/M2) tests.
# Sourced by run_*.sh; override via CLI flags or environment variables.

# Display / log timestamps in China local time (does NOT change host timezone).
export MLS_TIMEZONE="${MLS_TIMEZONE:-Asia/Shanghai}"
export TZ="${MLS_TIMEZONE}"

MLS_ROOT="${MLS_ROOT:-/mnt/data2/wbw/MLStest}"
MODEL_CASE_DIR="${MODEL_CASE_DIR:-${MLS_ROOT}/Qwen3.5-397B-A17B}"

# Conda / code roots
MLS_CONDA_ENV="${MLS_CONDA_ENV:-kt-kernel}"
MLS_CONDA_SH="${MLS_CONDA_SH:-/mnt/data2/wbw/miniconda3/etc/profile.d/conda.sh}"
MLS_CONDA_PYTHON="${MLS_CONDA_PYTHON:-/mnt/data2/wbw/conda/envs/kt-kernel/bin/python}"
KTRANSFORMERS_ROOT="${KTRANSFORMERS_ROOT:-/mnt/data2/wbw/ktransformers}"
SGLANG_KT_PYTHON="${SGLANG_KT_PYTHON:-${KTRANSFORMERS_ROOT}/third_party/sglang/python}"
KT_KERNEL_PYTHON="${KT_KERNEL_PYTHON:-${KTRANSFORMERS_ROOT}/kt-kernel/python}"

# Model paths
MODEL_PATH="${MODEL_PATH:-/mnt/data2/models/Qwen3.5-397B-A17B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
# Must point to a verified KT CPU expert pack for this base; do not invent a path.
# If KT weights live alongside the HF checkpoint, set KT_WEIGHT_PATH to the same path.
KT_WEIGHT_PATH="${KT_WEIGHT_PATH:-}"
KT_METHOD="${KT_METHOD:-AMXBF16}"

# Serving identity
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31006}"

# Hardware / KT
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-8}"
KT_CPUINFER="${KT_CPUINFER:-96}"
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-2}"
KT_NUMA_NODES="${KT_NUMA_NODES:-0 1}"
KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS:-0}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"

# LoRA contract (M1 default = single; set KT_LORA_DISPATCH=grouped for M2)
LORA_BACKEND="${LORA_BACKEND:-triton}"
MAX_LORA_RANK="${MAX_LORA_RANK:-8}"
MAX_LOADED_LORAS="${MAX_LOADED_LORAS:-4}"
KT_LORA_DISPATCH="${KT_LORA_DISPATCH:-single}"
if [[ "${KT_LORA_DISPATCH}" == "grouped" ]]; then
  MAX_LORAS_PER_BATCH="${MAX_LORAS_PER_BATCH:-4}"
  KT_MAX_LORAS_PER_BATCH="${KT_MAX_LORAS_PER_BATCH:-4}"
else
  MAX_LORAS_PER_BATCH="${MAX_LORAS_PER_BATCH:-1}"
  KT_MAX_LORAS_PER_BATCH="${KT_MAX_LORAS_PER_BATCH:-1}"
fi
KT_MAX_LOADED_LORAS="${KT_MAX_LOADED_LORAS:-${MAX_LOADED_LORAS}}"

# Adapter dirs: merged KT composite adapters (each with adapter_model.safetensors).
# Comma-separated name=path pairs, e.g. L0=/path/L0,L1=/path/L1
LORA_PATHS="${LORA_PATHS:-}"

# Caches / logs
LOG_BASE="${LOG_BASE:-${MODEL_CASE_DIR}/test_log}"
SGLANG_KT_LORA_CACHE_DIR="${SGLANG_KT_LORA_CACHE_DIR:-${MODEL_CASE_DIR}/cache/sglang_kt_lora}"
TMPDIR="${TMPDIR:-${MODEL_CASE_DIR}/tmp}"
