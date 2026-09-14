#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FFT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LLAMA_FACTORY_DIR="${FFT_LLAMA_FACTORY_DIR:-/mnt/data2/wbw/LLaMA-Factory}"
SHARED_FLOW_DIR="${FFT_SHARED_FLOW_DIR:-${FFT_ROOT}/Qwen3.5-35B-A3B}"
MODEL_KIND=""
MODEL_PATH=""
MODEL_DIR=""
BASE_CONFIG=""
TEMPLATE=""
LAYER_CLASS=""
DATASET_DIR="${FFT_DATASET_DIR:-${FFT_ROOT}/dataset}"
DATASET_NAME="${FFT_DATASET_NAME:-fft_real_100}"
SEQ_LENGTHS="1024,512,256,64,32"
STEPS=15
WARMUP_STEPS=5
GAS=1
THREADS="${FFT_PYTORCH_CUDA_THREADS:-12}"
DEVICES="${FFT_PYTORCH_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
CONDA_ENV="${FFT_PYTORCH_CUDA_CONDA_ENV:-Onednn}"
OUTPUT_ROOT=""
DRY_RUN=0
CONTINUE_ON_ERROR=0

usage() {
  cat <<'EOF'
Usage: run_pytorch_cuda_sweep.sh --model-kind qwen35_122b|glm45_air [options]
  --model-path PATH       Local model checkpoint
  --seq-lengths LIST      Comma-separated lengths (default: 1024,512,256,64,32)
  --steps N               Optimizer steps per case (default: 15)
  --warmup-steps N        Warm-up steps excluded from TPS (default: 5)
  --gas N                 Gradient accumulation steps (default: 1)
  --threads N             CPU threads per rank (default: 12)
  --devices LIST          Exactly 8 GPU ids (default: 0,1,2,3,4,5,6,7)
  --conda-env NAME        Runtime environment (default: Onednn)
  --output-root PATH      Sweep output directory
  --continue-on-error     Continue remaining sequence lengths after failure
  --dry-run               Generate configs and print commands only
EOF
}

while (($#)); do
  case "$1" in
    --model-kind) MODEL_KIND="$2"; shift 2;;
    --model-path) MODEL_PATH="$2"; shift 2;;
    --seq-lengths) SEQ_LENGTHS="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2;;
    --gas|--gradient-accumulation-steps) GAS="$2"; shift 2;;
    --threads|--cpu-threads) THREADS="$2"; shift 2;;
    --devices) DEVICES="$2"; shift 2;;
    --conda-env) CONDA_ENV="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

[[ "$MODEL_KIND" == qwen35_122b || "$MODEL_KIND" == glm45_air ]] || { echo "--model-kind is required" >&2; exit 2; }
if [[ "$MODEL_KIND" == qwen35_122b ]]; then
  MODEL_PATH="${MODEL_PATH:-/mnt/data2/models/Qwen3.5-122B-A10B}"
  MODEL_DIR="${FFT_ROOT}/Qwen3.5-122B-A10B"
  BASE_CONFIG="${MODEL_DIR}/configs/train_full_bf16_qwen35_122b.yaml"
  TEMPLATE="qwen3"
  LAYER_CLASS="Qwen3_5MoeDecoderLayer"
else
  MODEL_PATH="${MODEL_PATH:-/mnt/data2/models/GLM-4.5-Air}"
  MODEL_DIR="${FFT_ROOT}/GLM-4.5-Air"
  BASE_CONFIG="${MODEL_DIR}/configs/train_full_bf16_glm45_air.yaml"
  TEMPLATE="glm4_moe"
  LAYER_CLASS="Glm4MoeDecoderLayer"
fi
PYTHON="/mnt/data2/wbw/conda/envs/${CONDA_ENV}/bin/python3"
TORCHRUN="/mnt/data2/wbw/conda/envs/${CONDA_ENV}/bin/torchrun"
[[ -x "$PYTHON" ]] || { echo "missing environment ${CONDA_ENV}; run setup_onednn_env.sh" >&2; exit 2; }
[[ -x "$TORCHRUN" ]] || { echo "missing torchrun in ${CONDA_ENV}" >&2; exit 2; }
[[ "$DEVICES" =~ ^[0-9]+(,[0-9]+){7}$ ]] || { echo "--devices must contain exactly 8 ids" >&2; exit 2; }

ACTIVE="$(ps -eo pid=,args= | awk '$2 != "awk" && $0 ~ /aptmoe_(qwen|glm)|run_finetune_perf.*aptmoe|proxy_train\.py|pytorch_onednn_full_ft\.py/ {print}')"
if (( ! DRY_RUN )) && [[ -n "$ACTIVE" ]]; then
  echo "refusing to start PyTorch CUDA while another benchmark is active:" >&2
  echo "$ACTIVE" >&2
  exit 3
fi
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="${MODEL_DIR}/test_log/$(date -u +%Y%m%d_%H%M%S)_PYTORCH_CUDA_8GPU_FULL"
mkdir -p "$OUTPUT_ROOT"

make_config() {
  local output="$1" seq="$2" outdir="$3"
  "$PYTHON" - "$BASE_CONFIG" "$output" "$MODEL_PATH" "$DATASET_NAME" "$DATASET_DIR" "$seq" "$STEPS" "$GAS" "$outdir" "$TEMPLATE" "$LAYER_CLASS" <<'PY'
import sys
from pathlib import Path
import yaml

source, output, model, dataset, dataset_dir, seq, steps, gas, outdir, template, layer = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
config.update({
    "model_name_or_path": model, "dataset": dataset, "dataset_dir": dataset_dir,
    "template": template, "cutoff_len": int(seq), "max_steps": int(steps),
    "gradient_accumulation_steps": int(gas), "per_device_train_batch_size": 1,
    # pure_bf16 sets model compute dtype without asking Accelerate to enable
    # its mixed-precision FSDP path, which would create an extra FP32 flat copy.
    "output_dir": outdir, "finetuning_type": "full", "bf16": False,
    "fp16": False, "tf32": False, "pure_bf16": True, "use_kt": False,
    # Let FSDP exclusively own activation checkpointing. LLaMA-Factory also
    # enables model-native checkpointing unless this model argument is set,
    # even when Trainer gradient_checkpointing is false.
    "disable_gradient_checkpointing": True,
    "gradient_checkpointing": False,
    "gradient_checkpointing_kwargs": None,
    "kt_weight_path": None, "deepspeed": None, "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_transformer_layer_cls_to_wrap": layer,
        # Accelerate FSDP1 expects a ShardingStrategy name here; a boolean
        # becomes the invalid enum key ``TRUE`` in this runtime.
        "fsdp_reshard_after_forward": "FULL_SHARD",
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        # Keep non-active FSDP shards on host memory. A 122B full-FT shard
        # leaves too little headroom on 48 GiB GPUs for backward all-gathers.
        "fsdp_offload_params": True, "fsdp_use_orig_params": True,
        "cpu_ram_efficient_loading": True, "sync_module_states": True,
        "activation_checkpointing": True,
    },
})
Path(output).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

IFS=',' read -r -a lengths <<< "$SEQ_LENGTHS"
status=0
for seq in "${lengths[@]}"; do
  [[ "$seq" =~ ^[0-9]+$ && "$seq" -gt 0 ]] || { echo "invalid sequence: $seq" >&2; exit 2; }
  run_dir="${OUTPUT_ROOT}/seq_${seq}"
  timing_dir="${run_dir}/step_timing"
  config="${run_dir}/train_config.yaml"
  mkdir -p "$timing_dir"
  make_config "$config" "$seq" "${run_dir}/model_output"
  "$PYTHON" - "$run_dir/run_config.json" "$seq" "$MODEL_KIND" "$MODEL_PATH" "$DEVICES" "$STEPS" "$WARMUP_STEPS" "$GAS" "$CONDA_ENV" <<'PY'
import json, sys
from pathlib import Path
output, seq, kind, model, devices, steps, warmup, gas, env = sys.argv[1:]
Path(output).write_text(json.dumps({
    "backend": "pytorch_cuda_fsdp", "model_kind": kind, "model_path": model,
    "device": "cuda", "num_gpus": 8, "devices": devices.split(","),
    "precision": "bf16", "finetuning_type": "full", "sequence_length": int(seq),
    "steps": int(steps), "warmup_steps": int(warmup), "gradient_accumulation_steps": int(gas),
    "conda_env": env, "fsdp": True,
}, indent=2) + "\n", encoding="utf-8")
PY
  cmd=("$TORCHRUN" --standalone --nproc_per_node=8 -m finetune_train_with_timing train "$config")
  echo "[pytorch_cuda] ${cmd[*]}"
  if (( DRY_RUN )); then continue; fi
  set +e
  env CUDA_VISIBLE_DEVICES="$DEVICES" DISABLE_VERSION_CHECK=1 ACCELERATE_USE_FSDP=true \
    FSDP_CPU_RAM_EFFICIENT_LOADING=true FSDP_SYNC_MODULE_STATES=true \
    FFT_TRAINING_BACKEND=pytorch_cuda FFT_PRECISION=bf16 \
    FFT_FINETUNING_TYPE=full FFT_TEXT_ONLY=1 FFT_DISABLE_PERF_PROBES=1 FFT_SKIP_FINAL_SAVE=1 \
    FFT_STEP_TIMING_OUT_DIR="$timing_dir" FFT_STEP_TIMING_WARMUP_STEPS="$WARMUP_STEPS" \
    FFT_STEP_TIMING_TOKENS_PER_STEP="$((8 * seq * GAS))" FFT_CPU_THREADS="$THREADS" \
    OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="${MODEL_DIR}:${SHARED_FLOW_DIR}:${LLAMA_FACTORY_DIR}/src:${PYTHONPATH:-}" \
    "${cmd[@]}" 2>&1 | tee "${run_dir}/train.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "${run_dir}/exit_code.txt"
  if (( rc != 0 )); then status="$rc"; (( CONTINUE_ON_ERROR )) || exit "$rc"; fi
done
exit "$status"
