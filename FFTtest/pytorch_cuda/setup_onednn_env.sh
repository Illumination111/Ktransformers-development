#!/usr/bin/env bash
set -Eeuo pipefail
CONDA_BIN="${CONDA_BIN:-/mnt/data2/wbw/miniconda3/bin/conda}"
SOURCE_ENV="${SOURCE_ENV:-Aptmoe}"
TARGET_ENV="${TARGET_ENV:-Onednn}"
TARGET_PREFIX="/mnt/data2/wbw/conda/envs/${TARGET_ENV}"

repair_entrypoints() {
  local torchrun="${TARGET_PREFIX}/bin/torchrun"
  if [[ -f "$torchrun" ]]; then
    sed -i "1c#!${TARGET_PREFIX}/bin/python3" "$torchrun"
  fi
}
[[ -x "$CONDA_BIN" ]] || { echo "conda not found: $CONDA_BIN" >&2; exit 1; }
if CONDA_NO_PLUGINS=true "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$TARGET_ENV"; then
  if [[ -x "${TARGET_PREFIX}/bin/python3" ]]; then
    repair_entrypoints
    echo "Conda environment already exists: $TARGET_ENV"
    exit 0
  fi
  echo "Removing incomplete environment: $TARGET_ENV" >&2
  rm -rf "$TARGET_PREFIX"
fi
if CONDA_NO_PLUGINS=true CONDA_OFFLINE=true "$CONDA_BIN" create --offline --yes \
  --name "$TARGET_ENV" --clone "$SOURCE_ENV"; then
  echo "Created $TARGET_ENV by conda clone"
else
  echo "Conda package cache is incomplete; using a same-filesystem COW copy" >&2
  rm -rf "$TARGET_PREFIX"
  cp -a --reflink=auto "/mnt/data2/wbw/conda/envs/${SOURCE_ENV}" "$TARGET_PREFIX"
  printf '%s\n' "${TARGET_ENV}" > "${TARGET_PREFIX}/.codex_env_name"
  repair_entrypoints
  echo "Created $TARGET_ENV by COW copy of $SOURCE_ENV"
fi
