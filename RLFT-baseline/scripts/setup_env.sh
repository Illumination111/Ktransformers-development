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

# vLLM 0.12.0 is the upper bound accepted by this veRL commit. Install it
# first so its CUDA/PyTorch constraints resolve before the editable veRL tree.
"$uv_bin" pip install --python "$B0_CONDA_PREFIX/bin/python" 'datasets>=3.0,<5' 'huggingface-hub>=0.30' 'scipy>=1.11' 'jinja2>=3.1'
"$uv_bin" pip install --python "$B0_CONDA_PREFIX/bin/python" 'vllm==0.12.0'
"$uv_bin" pip install --python "$B0_CONDA_PREFIX/bin/python" --no-build-isolation -e "$B0_WORKTREE[vllm,math]"

"$uv_bin" pip freeze --python "$B0_CONDA_PREFIX/bin/python" > "$B0_ROOT/env/pip-freeze.txt"
"$B0_CONDA_PREFIX/bin/python" - <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

root = Path("/home/wubowen/Ktransformers-development/RLFT-baseline")
names = ["verl", "vllm", "torch", "transformers", "ray", "datasets", "pyarrow", "tensordict"]
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
payload = {"python": sys.version, "platform": platform.platform(), "packages": versions}
(root / "env" / "versions.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

printf 'Conda environment ready: %s (%s)\n' "$B0_CONDA_ENV" "$B0_CONDA_PREFIX"
