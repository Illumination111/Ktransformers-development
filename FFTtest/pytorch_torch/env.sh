# Shared, read-only environment for the accepted sap4 Torch MoE stack.
# This file must never pip/conda-install into /mnt/data2/xmy/venv.
# Extra/vendor trees are prepended via PYTHONPATH only.

FFT_XMY_VENV="${FFT_XMY_VENV:-/mnt/data2/xmy/venv}"
FFT_TORCH_MOE_ROOT="${FFT_TORCH_MOE_ROOT:-/mnt/data3/qujing/torch-moe-qwen-current}"
FFT_TORCH_MOE_V56_DEPS="${FFT_TORCH_MOE_V56_DEPS:-/mnt/data3/qujing/qwen-v56-deps}"
FFT_TORCH_MOE_COMPAT_DEPS="${FFT_TORCH_MOE_COMPAT_DEPS:-/mnt/data/djw/torch-moe-e2e-dsv31-20260803-full/python-deps}"
FFT_TORCH_MOE_TRANSFORMERS="${FFT_TORCH_MOE_TRANSFORMERS:-${FFT_TORCH_MOE_ROOT}/sources/transformers-v56}"
FFT_TORCH_MOE_ACCELERATE="${FFT_TORCH_MOE_ACCELERATE:-${FFT_TORCH_MOE_ROOT}/sources/accelerate}"
FFT_TORCH_MOE_LLAMAFACTORY="${FFT_TORCH_MOE_LLAMAFACTORY:-${FFT_TORCH_MOE_ROOT}/sources/llamafactory-vendor}"

fft_torch_python() {
  printf '%s\n' "${FFT_XMY_VENV}/bin/python"
}

fft_torch_pythonpath() {
  local extra="$1"
  local path="${FFT_TORCH_MOE_V56_DEPS}:${FFT_TORCH_MOE_COMPAT_DEPS}:${FFT_TORCH_MOE_LLAMAFACTORY}:${FFT_TORCH_MOE_TRANSFORMERS}:${FFT_TORCH_MOE_ACCELERATE}"
  if [[ -n "${extra}" ]]; then
    path="${extra}:${path}"
  fi
  if [[ -n "${PYTHONPATH:-}" ]]; then
    path="${path}:${PYTHONPATH}"
  fi
  printf '%s\n' "${path}"
}
