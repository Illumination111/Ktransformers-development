#!/usr/bin/env python3
"""Summarize JSONL samples emitted by resource_monitor.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MIB = 1024 * 1024


def mib(value: int) -> float:
    return round(value / MIB, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    samples = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        raise RuntimeError(f"no resource samples found in {args.input}")

    host_used = [sample["host_memory"]["panel_used_bytes"] for sample in samples]
    host_available = [sample["host_memory"]["available_bytes"] for sample in samples]
    host_total = samples[0]["host_memory"]["total_bytes"]
    by_gpu: dict[int, list[dict[str, Any]]] = {}
    aggregate_by_sample: list[int] = []
    for sample in samples:
        gpus = sample.get("gpus") or []
        aggregate_by_sample.append(sum(gpu["memory_used_mib"] for gpu in gpus))
        for gpu in gpus:
            by_gpu.setdefault(gpu["index"], []).append(gpu)
    if args.require_gpu and not by_gpu:
        errors = [sample.get("gpu_error") for sample in samples if sample.get("gpu_error")]
        raise RuntimeError(f"no GPU samples collected; errors={errors[:3]}")

    gpu_summary: list[dict[str, Any]] = []
    for index, records in sorted(by_gpu.items()):
        used = [record["memory_used_mib"] for record in records]
        gpu_summary.append(
            {
                "index": index,
                "name": records[0]["name"],
                "total_mib": records[0]["memory_total_mib"],
                "baseline_used_mib": used[0],
                "peak_used_mib": max(used),
                "peak_delta_mib": max(used) - used[0],
                "peak_utilization_percent": max(
                    record["utilization_percent"] for record in records
                ),
            }
        )

    summary = {
        "sample_count": len(samples),
        "duration_seconds": samples[-1]["elapsed_seconds"],
        "host_memory": {
            "measurement": (
                "htop/free-style physical memory panel; never summed process RSS"
            ),
            "total_mib": mib(host_total),
            "baseline_panel_used_mib": mib(host_used[0]),
            "peak_panel_used_mib": mib(max(host_used)),
            "peak_training_delta_mib": mib(max(host_used) - host_used[0]),
            "minimum_available_mib": mib(min(host_available)),
        },
        "gpu_memory": {
            "device_count": len(gpu_summary),
            "baseline_aggregate_used_mib": aggregate_by_sample[0],
            "peak_aggregate_used_mib": max(aggregate_by_sample),
            "peak_aggregate_delta_mib": max(aggregate_by_sample)
            - aggregate_by_sample[0],
            "devices": gpu_summary,
        },
        "measurement_note": (
            "Host used RAM follows the htop/free top-panel formula from /proc/meminfo "
            "and is not an RSS sum. GPU memory is sampled from nvidia-smi for the "
            "selected physical GPU indices. Training deltas subtract the first sample."
        ),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
