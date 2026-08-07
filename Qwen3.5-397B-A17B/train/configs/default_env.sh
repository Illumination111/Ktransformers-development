# Shared defaults for Qwen3.5-397B-A17B LoRA SFT (KT) training.
# Sourced by run_*.sh; override via environment variables.

# Display / log timestamps in China local time (does NOT change host timezone).
export MLS_TIMEZONE="${MLS_TIMEZONE:-Asia/Shanghai}"
export TZ="${MLS_TIMEZONE}"

MLS_ROOT="${MLS_ROOT:-/mnt/data2/wbw/MLStest}"
MODEL_CASE_DIR="${MODEL_CASE_DIR:-${MLS_ROOT}/Qwen3.5-397B-A17B}"
TRAIN_DIR="${TRAIN_DIR:-${MODEL_CASE_DIR}/train}"

# Conda: LLaMA-Factory editable install lives in Kllama (not kt-kernel).
MLS_CONDA_ENV="${MLS_CONDA_ENV:-Kllama}"
MLS_CONDA_SH="${MLS_CONDA_SH:-/mnt/data2/wbw/miniconda3/etc/profile.d/conda.sh}"
MLS_CONDA_PYTHON="${MLS_CONDA_PYTHON:-/mnt/data2/wbw/conda/envs/${MLS_CONDA_ENV}/bin/python}"
MLS_CONDA_BIN="${MLS_CONDA_BIN:-/mnt/data2/wbw/conda/envs/${MLS_CONDA_ENV}/bin}"

LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-/mnt/data2/wbw/LLaMA-Factory}"
KTRANSFORMERS_ROOT="${KTRANSFORMERS_ROOT:-/mnt/data2/wbw/ktransformers}"
CONVERT_KT_SCRIPT="${CONVERT_KT_SCRIPT:-${KTRANSFORMERS_ROOT}/kt-kernel/scripts/convert_kt_to_sglang_adapter.py}"

# Base model (must match serving MODEL_PATH)
MODEL_PATH="${MODEL_PATH:-/mnt/data2/models/Qwen3.5-397B-A17B}"

# Raw HF downloads
DATASET_ROOT="${DATASET_ROOT:-${MLS_ROOT}/dataset}"
# Prepared openai jsonl + dataset_info.json
DATASET_DIR="${DATASET_DIR:-${TRAIN_DIR}/data}"

# Checkpoints (raw KT LoRA save) and serving composites
RUN_ROOT="${RUN_ROOT:-${TRAIN_DIR}/runs}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${MLS_ROOT}/lora-adapter/Qwen3.5-397B-A17B}"

# Accelerate / hardware
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${TRAIN_DIR}/configs/accelerate_fsdp2_kt_bf16_8gpu.yaml}"
# Optional: path to pre-converted AMXINT8 expert pack; if set, prefer int8 accelerate config.
KT_WEIGHT_PATH="${KT_WEIGHT_PATH:-}"

# Text-only Qwen3.5: text_config → Qwen3_5MoeForCausalLM (no visual/Conv3d).
# Same contract as FFTtest/Qwen3.5-35B-A3B/qwen35_text_only.py.
MLS_TEXT_ONLY="${MLS_TEXT_ONLY:-1}"

# Triton / DeepSpeed autotune cache (avoid broken ~/.triton symlink).
# Prefer the permanent user offload on data disk; MLStest train falls back here too.
MLS_CACHE_ROOT="${MLS_CACHE_ROOT:-/mnt/data2/wbw/.home-offload}"
if [[ -f /mnt/data2/wbw/.home-offload/env.sh ]]; then
  # shellcheck source=/dev/null
  source /mnt/data2/wbw/.home-offload/env.sh
fi
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${MLS_CACHE_ROOT}/triton}"
# gcc / triton / torch JIT temps — root /tmp is often full; keep on data disk (FFTtest-style).
MLS_TMPDIR="${MLS_TMPDIR:-${MLS_CACHE_ROOT}/tmp}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${MLS_CACHE_ROOT}/cuda}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${MLS_CACHE_ROOT}/torch_extensions}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${MLS_CACHE_ROOT}/torchinductor}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${MLS_CACHE_ROOT}/cache/matplotlib}"

LOG_DIR="${LOG_DIR:-${TRAIN_DIR}/logs}"
