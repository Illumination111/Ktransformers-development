#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
baseline_root=$(cd -- "$script_dir/.." && pwd)
# shellcheck source=../configs/b0.env
source "$baseline_root/configs/b0.env"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [ -f "$1" ] || die "missing file: $1"
}

require_dir() {
    [ -d "$1" ] || die "missing directory: $1"
}

require_clean_worktree() {
    require_dir "$B0_WORKTREE"
    current_commit=$(git -C "$B0_WORKTREE" rev-parse HEAD)
    [ "$current_commit" = "$B0_COMMIT" ] || die "veRL commit is $current_commit, expected $B0_COMMIT"
    [ -z "$(git -C "$B0_WORKTREE" status --porcelain)" ] || die "veRL worktree is not clean"
}

require_conda_env() {
    [ -x "$B0_CONDA_PREFIX/bin/python" ] || die "Conda environment $B0_CONDA_ENV is absent; run scripts/setup_env.sh first"
}

count_visible_gpus() {
    "$B0_CONDA_PREFIX/bin/python" -c 'import torch; print(torch.cuda.device_count())'
}
