#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

uv_bin="$B0_ROOT/env/tools/uv"
if [ ! -x "$uv_bin" ]; then
    uv_bin=$(command -v uv || true)
fi
[ -n "$uv_bin" ] && [ -x "$uv_bin" ] || die "uv is required but was not found"
require_clean_worktree

conda_bin=/home/wubowen/miniconda3/bin/conda
[ -x "$conda_bin" ] || die "Conda executable not found: $conda_bin"
if ! "$conda_bin" env list | awk '{print $1}' | grep -Fxq "$B0_CONDA_ENV"; then
    "$conda_bin" create -y -n "$B0_CONDA_ENV" python=3.11 pip
fi
[ -x "$B0_CONDA_PREFIX/bin/python" ] || die "unexpected Conda prefix for $B0_CONDA_ENV: $B0_CONDA_PREFIX"

# Historical successful baseline pins: PyTorch 2.8.0+cu128 and
# Transformers 4.57.6.  Install the editable veRL/SGLang stack first, then
# enforce these two versions so dependency resolution cannot silently drift
# back to the newer defaults.
"$uv_bin" pip install --python "$B0_CONDA_PREFIX/bin/python" 'datasets>=3.0,<5' 'huggingface-hub>=0.30' 'scipy>=1.11' 'jinja2>=3.1'
if "$B0_CONDA_PREFIX/bin/python" -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("vllm") is None)'; then
    "$uv_bin" pip uninstall --python "$B0_CONDA_PREFIX/bin/python" vllm
fi
"$uv_bin" pip install --python "$B0_CONDA_PREFIX/bin/python" --no-build-isolation -e "$B0_WORKTREE[sglang,math]"
"$uv_bin" pip uninstall --python "$B0_CONDA_PREFIX/bin/python" flash-attn || true
# sgl-kernel 0.3.21 is compiled for Torch 2.9; 0.3.18 is the compatible
# wheel for the historical Torch 2.8 stack.
"$uv_bin" pip install --python "$B0_CONDA_PREFIX/bin/python" --no-deps "sgl-kernel==$B0_SGL_KERNEL_VERSION"
"$uv_bin" pip install --python "$B0_CONDA_PREFIX/bin/python" --index-url https://download.pytorch.org/whl/cu128 --no-deps "torch==$B0_PYTORCH_VERSION"
"$uv_bin" pip install --python "$B0_CONDA_PREFIX/bin/python" --no-deps "transformers==$B0_TRANSFORMERS_VERSION"
"$uv_bin" pip install --python "$B0_CONDA_PREFIX/bin/python" --no-deps "peft==$B0_PEFT_VERSION"

"$uv_bin" pip freeze --python "$B0_CONDA_PREFIX/bin/python" > "$B0_ROOT/env/pip-freeze.txt"
RLFT_BASELINE_ROOT="$B0_ROOT" "$B0_CONDA_PREFIX/bin/python" - <<'PY'
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

root = Path(os.environ["RLFT_BASELINE_ROOT"])
names = ["verl", "sglang", "sgl-kernel", "flashinfer-python", "flash-attn", "torch", "transformers", "accelerate", "peft", "torchao", "ray", "datasets", "pyarrow", "tensordict"]
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
payload = {"python": sys.version, "platform": platform.platform(), "torch_cuda": __import__("torch").version.cuda, "packages": versions}
(root / "env" / "versions.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

printf 'Conda environment ready: %s (%s)\n' "$B0_CONDA_ENV" "$B0_CONDA_PREFIX"
