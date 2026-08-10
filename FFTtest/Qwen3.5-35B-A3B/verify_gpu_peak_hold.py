#!/usr/bin/env python3
"""Verify that the longest-sequence VRAM peak stayed held across a profile."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from persistent_sweep import load_manifest


AUTO_RELEASED_EXIT = 97
INSUFFICIENT_SAMPLES_EXIT = 98


def _memory_value(row: dict[str, str], gpu: int) -> float | None:
    for key in (f"proc_gpu{gpu}_mem_mb", f"gpu{gpu}_mem_used_mb"):
        raw = row.get(key)
        if raw not in (None, ""):
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def verify(
    manifest: dict[str, Any],
    monitor_path: Path,
    tolerance_mib: float,
) -> dict[str, Any]:
    with monitor_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sequences = [int(case["sequence_length"]) for case in manifest["cases"]]
    longest = sequences[0]
    devices = [int(value) for value in str(manifest["devices"]).split(",")]
    per_gpu: dict[str, Any] = {}
    confirmed = True
    insufficient = False

    for gpu in devices:
        phase_values: dict[str, list[float]] = {}
        for sequence in sequences:
            phase = f"seq_{sequence}"
            values = [
                value
                for row in rows
                if row.get("phase") == phase
                for value in [_memory_value(row, gpu)]
                if value is not None and value > 0
            ]
            phase_values[phase] = values
        first_values = phase_values[f"seq_{longest}"]
        if not first_values:
            insufficient = True
            confirmed = False
            per_gpu[str(gpu)] = {
                "status": "INSUFFICIENT_SAMPLES",
                "reason": f"no positive samples for seq_{longest}",
            }
            continue
        longest_peak = max(first_values)
        later: dict[str, Any] = {}
        gpu_confirmed = True
        for sequence in sequences[1:]:
            values = phase_values[f"seq_{sequence}"]
            if not values:
                insufficient = True
                gpu_confirmed = False
                later[str(sequence)] = {
                    "samples": 0,
                    "minimum_mib": None,
                    "held": False,
                }
                continue
            minimum = min(values)
            held = minimum + tolerance_mib >= longest_peak
            gpu_confirmed = gpu_confirmed and held
            later[str(sequence)] = {
                "samples": len(values),
                "minimum_mib": minimum,
                "held": held,
            }
        confirmed = confirmed and gpu_confirmed
        per_gpu[str(gpu)] = {
            "status": "CONFIRMED" if gpu_confirmed else "DROPPED_BELOW_PEAK",
            "longest_sequence": longest,
            "longest_peak_mib": longest_peak,
            "tolerance_mib": tolerance_mib,
            "later_sequences": later,
        }

    status = (
        "CONFIRMED"
        if confirmed
        else "INSUFFICIENT_SAMPLES"
        if insufficient
        else "AUTO_RELEASED"
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "confirmed": confirmed,
        "backend": manifest["backend"],
        "profile": manifest["profile"],
        "persistent_profile_process": True,
        "longest_sequence": longest,
        "sequences": sequences,
        "devices": devices,
        "tolerance_mib": tolerance_mib,
        "monitor_csv": str(monitor_path),
        "per_gpu": per_gpu,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-mib", type=float, default=512.0)
    args = parser.parse_args()
    raw_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = load_manifest(args.manifest, str(raw_manifest["backend"]))
    report = verify(manifest, args.monitor, args.tolerance_mib)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[gpu-peak-hold] status={report['status']} "
        f"report={args.output}"
    )
    if report["confirmed"]:
        return 0
    if report["status"] == "INSUFFICIENT_SAMPLES":
        return INSUFFICIENT_SAMPLES_EXIT
    return AUTO_RELEASED_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
