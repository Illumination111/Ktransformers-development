#!/usr/bin/env python3
"""Collect reproducibility metadata without starting CUDA workloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/wubowen/Ktransformers-development/RLFT-baseline")
MODEL = Path("/mnt/qjh007/models/Qwen3-30B-A3B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "manifests" / "environment_manifest.json")
    parser.add_argument("--hash-model-shards", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str) -> dict[str, Any]:
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=60, check=False)
        return {"argv": list(args), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": list(args), "error": f"{type(exc).__name__}: {exc}"}


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("verl", "vllm", "torch", "transformers", "ray", "datasets", "pyarrow", "tensordict"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def hashed_files(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(set(paths)):
        if path.is_file():
            rows.append({"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def main() -> int:
    args = parse_args()
    worktree = ROOT / "worktree"
    script_files = list((ROOT / "scripts").glob("*")) + list((ROOT / "configs").glob("*"))
    model_files = [
        MODEL / name
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        )
    ]
    if args.hash_model_shards:
        model_files.extend(MODEL.glob("*.safetensors"))

    payload = {
        "protocol": "qwen3-30b-a3b-verl-grpo-b0",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "NCCL_DEBUG", "WANDB_MODE", "PYTHONHASHSEED")
        },
        "verl": {
            "path": str(worktree.resolve()),
            "head": command("git", "-C", str(worktree), "rev-parse", "HEAD"),
            "status": command("git", "-C", str(worktree), "status", "--porcelain"),
        },
        "hardware": {
            "nvidia_smi": command("nvidia-smi", "-q"),
            "topology": command("nvidia-smi", "topo", "-m"),
            "lscpu": command("lscpu"),
            "numactl": command("numactl", "--hardware"),
        },
        "scripts_and_configs": hashed_files(script_files),
        "model_files": hashed_files(model_files),
        "model_shards_hashed": args.hash_model_shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
