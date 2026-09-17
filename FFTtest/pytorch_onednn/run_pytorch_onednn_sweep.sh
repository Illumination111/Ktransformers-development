#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FFT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_KIND=""
MODEL_PATH=""
DATASET_DIR="${FFT_ROOT}/dataset"
DATASET_NAME="fft_real_100"
SEQ_LENGTHS="1024,512,256,64,32"
STEPS="15"
WARMUP_STEPS="5"
BATCH_SIZE="1"
GAS="1"
GLOBAL_BATCH_SIZE="8"
THREADS="${FFT_ONEDNN_THREADS:-0}"
OWNER_THREADS="${FFT_ONEDNN_OWNER_THREADS:-80}"
NON_OWNER_THREADS="${FFT_ONEDNN_NON_OWNER_THREADS:-2}"
NUM_GPUS="${FFT_ONEDNN_NUM_GPUS:-8}"
PIPELINE_PARALLEL_SIZE="${FFT_ONEDNN_PP_SIZE:-8}"
DATA_PARALLEL_SIZE="${FFT_ONEDNN_DP_SIZE:-1}"
TENSOR_PARALLEL_SIZE="${FFT_ONEDNN_TP_SIZE:-1}"
DEVICES="${FFT_ONEDNN_DEVICES:-0,1,2,3,4,5,6,7}"
CONDA_ENV="${FFT_PYTORCH_CONDA_ENV:-Onednn}"
OUTPUT_ROOT=""
DRY_RUN=0
ALLOW_CONCURRENT=0
CONTINUE_ON_ERROR=0
DNNL_VERBOSE_VALUE="${DNNL_VERBOSE:-0}"

usage() {
  cat <<'EOF'
Usage: run_pytorch_onednn_sweep.sh --model-kind qwen35_122b|glm45_air [options]
  --model-path PATH       Local model directory (otherwise the model default)
  --dataset-dir PATH      Dataset directory (default: FFTtest/dataset)
  --dataset-name NAME     Dataset registered in dataset_info.json
  --seq-lengths LIST      Comma-separated lengths (default: 1024,512,256,64,32)
  --steps N                Optimizer steps per case (default: 15)
  --warmup-steps N        Warm-up steps excluded from TPS (default: 5)
  --batch-size N          Per-stage microbatch size (default: 1)
  --global-batch-size N   Global batch split into pipeline microbatches (default: 8)
  --gas N                  Gradient accumulation steps (default: 1)
  --threads N              Override CPU threads on every rank (default: KT role split)
  --owner-threads N        Rank 0 CPU expert threads (default: 80)
  --non-owner-threads N    Rank 1..N CPU threads (default: 2)
  --num-gpus N             Pipeline ranks (default: 8)
  --pipeline-parallel-size N  Pipeline ranks per DP replica (default: 8)
  --data-parallel-size N  Replicated pipeline groups (default: 1)
  --tensor-parallel-size N Tensor parallel degree (default: 1)
  --devices LIST           CUDA devices exposed to torchrun (default: 0,1,2,3,4,5,6,7)
  --conda-env NAME         Conda environment under /mnt/data2/wbw/conda/envs
  --output-root PATH       Sweep output directory
  --dnnl-verbose 0|1       Set DNNL_VERBOSE for kernel verification
  --allow-concurrent       Override the APTMoE safety guard
  --continue-on-error      Run remaining lengths after a failed case
  --dry-run                Print commands without loading a model
EOF
}

while (($#)); do
  case "$1" in
    --model-kind) MODEL_KIND="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --seq-lengths) SEQ_LENGTHS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
    --gas|--gradient-accumulation-steps) GAS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --owner-threads) OWNER_THREADS="$2"; shift 2 ;;
    --non-owner-threads) NON_OWNER_THREADS="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --pipeline-parallel-size) PIPELINE_PARALLEL_SIZE="$2"; shift 2 ;;
    --data-parallel-size) DATA_PARALLEL_SIZE="$2"; shift 2 ;;
    --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --device) shift 2 ;; # legacy single-device option; torchrun owns device binding
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dnnl-verbose) DNNL_VERBOSE_VALUE="$2"; shift 2 ;;
    --allow-concurrent) ALLOW_CONCURRENT=1; shift ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$MODEL_KIND" == "qwen35_122b" || "$MODEL_KIND" == "glm45_air" ]] || {
  echo "--model-kind is required and must be qwen35_122b or glm45_air" >&2; exit 2;
}
if [[ -z "$MODEL_PATH" ]]; then
  if [[ "$MODEL_KIND" == "qwen35_122b" ]]; then MODEL_PATH="/mnt/data2/models/Qwen3.5-122B-A10B"; fi
  if [[ "$MODEL_KIND" == "glm45_air" ]]; then MODEL_PATH="/mnt/data2/models/GLM-4.5-Air"; fi
fi
PYTHON="/mnt/data2/wbw/conda/envs/${CONDA_ENV}/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -x "$PYTHON" ]] || { echo "cannot find Python for conda env ${CONDA_ENV}" >&2; exit 2; }
TORCHRUN="/mnt/data2/wbw/conda/envs/${CONDA_ENV}/bin/torchrun"
[[ -x "$TORCHRUN" ]] || TORCHRUN="$(command -v torchrun)"
[[ -x "$TORCHRUN" ]] || { echo "cannot find torchrun for conda env ${CONDA_ENV}" >&2; exit 2; }

# Ignore awk itself: its command line contains the detection expression.
APTMOE_PROCS="$(ps -eo pid=,args= | awk '$2 != "awk" && $0 ~ /aptmoe_(qwen|glm)|run_finetune_perf.*aptmoe|proxy_train\.py|pytorch_cuda.*finetune_train_with_timing|pytorch_onednn_full_ft/ {print}')"
if (( ! DRY_RUN && ! ALLOW_CONCURRENT )) && [[ -n "$APTMOE_PROCS" ]]; then
  echo "refusing to start PyTorch oneDNN while APTMoE is active:" >&2
  echo "$APTMOE_PROCS" >&2
  echo "wait for APTMoE or pass --allow-concurrent explicitly" >&2
  exit 3
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  model_dir="${FFT_ROOT}/Qwen3.5-122B-A10B"
  [[ "$MODEL_KIND" == "glm45_air" ]] && model_dir="${FFT_ROOT}/GLM-4.5-Air"
  OUTPUT_ROOT="${model_dir}/test_log/$(date -u +%Y%m%d_%H%M%S)_PYTORCH_ONEDNN_FULL"
fi
mkdir -p "$OUTPUT_ROOT"

IFS=',' read -r -a lengths <<< "$SEQ_LENGTHS"
overall_status=0
for seq in "${lengths[@]}"; do
  [[ "$seq" =~ ^[0-9]+$ && "$seq" -gt 0 ]] || { echo "invalid sequence length: $seq" >&2; exit 2; }
  case_dir="${OUTPUT_ROOT}/seq_${seq}"
  cmd=("$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" "${SCRIPT_DIR}/pytorch_onednn_full_ft.py"
    --model-kind "$MODEL_KIND" --model-path "$MODEL_PATH"
    --dataset-dir "$DATASET_DIR" --dataset-name "$DATASET_NAME"
    --output-dir "$case_dir" --sequence-length "$seq" --steps "$STEPS"
    --warmup-steps "$WARMUP_STEPS" --batch-size "$BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GAS" --threads "$THREADS"
    --owner-threads "$OWNER_THREADS" --non-owner-threads "$NON_OWNER_THREADS"
    --num-gpus "$NUM_GPUS" --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
    --data-parallel-size "$DATA_PARALLEL_SIZE" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
  (( DRY_RUN )) && cmd+=(--dry-run)
  echo "[pytorch_onednn] ${cmd[*]}"
  if (( DRY_RUN )); then continue; fi
  set +e
  launcher_threads="$THREADS"
  [[ "$launcher_threads" == "0" ]] && launcher_threads="1"
  env CUDA_VISIBLE_DEVICES="$DEVICES" \
    OMP_NUM_THREADS="$launcher_threads" MKL_NUM_THREADS="$launcher_threads" \
    OPENBLAS_NUM_THREADS="$launcher_threads" DNNL_VERBOSE="$DNNL_VERBOSE_VALUE" "${cmd[@]}"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "${case_dir}/exit_code.txt"
  if (( rc != 0 )); then
    overall_status="$rc"
    (( CONTINUE_ON_ERROR )) || exit "$rc"
  fi
done
exit "$overall_status"
