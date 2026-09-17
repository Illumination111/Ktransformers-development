#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../pytorch_cuda/run_pytorch_cuda_sweep.sh" --model-kind qwen35_122b "$@"
