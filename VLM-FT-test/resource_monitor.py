#!/usr/bin/env python3
"""Sample host RAM and NVIDIA GPU memory into a JSONL file."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any


STOP = Event()


def stop(_signum: int, _frame: Any) -> None:
    STOP.set()


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    total = values["MemTotal"]
    available = values["MemAvailable"]
    # This is the physical-memory "used" value shown by procps/htop-style
    # top panels. In particular, it is not a sum of process RSS values.
    panel_used = total - sum(
        values.get(name, 0)
        for name in ("MemFree", "Buffers", "Cached", "SReclaimable")
    )
    return {
        "total_bytes": total,
        "available_bytes": available,
        "panel_used_bytes": max(panel_used, 0),
        "memory_pressure_bytes": total - available,
        "free_bytes": values.get("MemFree", 0),
        "buffers_bytes": values.get("Buffers", 0),
        "cached_bytes": values.get("Cached", 0),
        "reclaimable_bytes": values.get("SReclaimable", 0),
        "swap_used_bytes": values.get("SwapTotal", 0) - values.get("SwapFree", 0),
    }


def read_optional_integer(path: Path) -> int | str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value == "max":
        return value
    try:
        return int(value)
    except ValueError:
        return None


def read_gpus(devices: set[str]) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or fields[0] not in devices:
            continue
        gpus.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
            }
        )
    return gpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    devices = {item.strip() for item in args.devices.split(",") if item.strip()}
    if not devices:
        parser.error("--devices must contain at least one GPU index")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.output.open("w", encoding="utf-8", buffering=1) as output:
        while not STOP.is_set():
            sample: dict[str, Any] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "host_memory": read_meminfo(),
                "cgroup_memory": {
                    "current_bytes": read_optional_integer(
                        Path("/sys/fs/cgroup/memory.current")
                    ),
                    "max_bytes": read_optional_integer(Path("/sys/fs/cgroup/memory.max")),
                },
            }
            try:
                sample["gpus"] = read_gpus(devices)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                sample["gpus"] = []
                sample["gpu_error"] = str(exc)
            output.write(json.dumps(sample, sort_keys=True) + "\n")
            STOP.wait(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
