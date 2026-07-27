#!/usr/bin/env python3
"""Shared helpers for a longest-first, process-persistent sequence sweep."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any


def load_manifest(path: str | Path, expected_backend: str) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported persistent sweep manifest: {manifest_path}")
    if manifest.get("backend") != expected_backend:
        raise ValueError(
            f"manifest backend={manifest.get('backend')!r}, "
            f"expected {expected_backend!r}"
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("persistent sweep manifest must contain cases")
    sequences = [int(case["sequence_length"]) for case in cases]
    if len(set(sequences)) != len(sequences):
        raise ValueError(f"duplicate sequence lengths: {sequences}")
    if sequences != sorted(sequences, reverse=True):
        raise ValueError(
            f"sequence lengths must run longest first, got: {sequences}"
        )
    for case in cases:
        run_dir = Path(case["run_dir"])
        if run_dir.name != f"seq_{int(case['sequence_length'])}":
            raise ValueError(f"run_dir does not match sequence: {run_dir}")
    manifest["_path"] = str(manifest_path)
    return manifest


def emit_monitor_phase(manifest: dict[str, Any], phase: str) -> None:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return
    fifo_value = manifest.get("monitor_fifo")
    if not fifo_value:
        return
    fifo_path = Path(fifo_value)
    fd: int | None = None
    for _ in range(50):
        try:
            fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
            break
        except OSError:
            time.sleep(0.1)
    if fd is None:
        raise RuntimeError(f"monitor FIFO is unavailable: {fifo_path}")
    try:
        os.write(fd, f"phase:{phase}\n".encode())
    finally:
        os.close(fd)


def activate_cuda_cache_hold(manifest: dict[str, Any]) -> None:
    marker = Path(manifest["cuda_cache_hold_marker"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def write_case_exit(case: dict[str, Any], exit_code: int | str) -> None:
    run_dir = Path(case["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "exit_code.txt").write_text(
        f"{exit_code}\n",
        encoding="utf-8",
    )


def write_case_cuda_snapshot(
    manifest: dict[str, Any],
    case: dict[str, Any],
    backend: str,
) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        snapshot = {
            "backend": backend,
            "profile": manifest["profile"],
            "sequence_length": int(case["sequence_length"]),
            "rank": int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "cuda_device": int(device),
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "hold_marker_active": Path(
                manifest["cuda_cache_hold_marker"]
            ).exists(),
        }
    except Exception:
        return
    out = (
        Path(case["run_dir"])
        / f"cuda_residency_rank_{snapshot['rank']}.json"
    )
    out.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def collect_without_releasing_cuda() -> None:
    """Drop Python objects while preserving the process CUDA caching allocator."""
    gc.collect()


def reset_accelerate_state() -> None:
    """Allow LLaMA-Factory to construct a fresh trainer in the same ranks."""
    try:
        from accelerate.state import AcceleratorState

        AcceleratorState._reset_state(reset_partial_state=True)
    except Exception:
        pass
    try:
        from accelerate.state import GradientState

        GradientState._reset_state()
    except Exception:
        pass
