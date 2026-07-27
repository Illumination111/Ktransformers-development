#!/usr/bin/env python3
"""Run APTMoE proxy sequences in persistent torchrun ranks."""

from __future__ import annotations

import argparse
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from aptmoe_qwen35_proxy_train import run, validate_args
from persistent_sweep import (
    activate_cuda_cache_hold,
    collect_without_releasing_cuda,
    emit_monitor_phase,
    load_manifest,
    write_case_cuda_snapshot,
    write_case_exit,
)


PATH_FIELDS = {
    "aptmoe_root",
    "simulation_root",
    "model_path",
    "dataset_dir",
    "output_dir",
    "step_timing_output_dir",
    "route_trace",
    "lookup_table",
}


def build_args(values: dict[str, Any]) -> Namespace:
    converted = dict(values)
    for name in PATH_FIELDS:
        if converted.get(name) not in (None, ""):
            converted[name] = Path(converted[name])
        elif name in {"route_trace", "lookup_table"}:
            converted[name] = None
    return Namespace(**converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-manifest", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.sweep_manifest, "aptmoe")
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for an APTMoE persistent sweep")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    try:
        for case in manifest["cases"]:
            sequence = int(case["sequence_length"])
            emit_monitor_phase(manifest, f"seq_{sequence}")
            print(
                f"[aptmoe-persistent-sweep] BEGIN seq={sequence}",
                flush=True,
            )
            case_args = build_args(case["aptmoe_arguments"])
            validate_args(case_args)
            run(case_args)
            activate_cuda_cache_hold(manifest)
            write_case_cuda_snapshot(manifest, case, "aptmoe")
            if rank == 0:
                write_case_exit(case, 0)
            collect_without_releasing_cuda()
            print(
                f"[aptmoe-persistent-sweep] END seq={sequence}; "
                "CUDA allocator and NCCL process group retained",
                flush=True,
            )
        emit_monitor_phase(manifest, "profile_release")
    except BaseException:
        if rank == 0:
            write_case_exit(case, 1)
        emit_monitor_phase(manifest, "profile_abort")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
