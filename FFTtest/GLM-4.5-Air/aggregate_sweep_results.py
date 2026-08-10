#!/usr/bin/env python3
"""Aggregate GLM-4.5-Air server/consumer sweep results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "status",
    "backend",
    "profile",
    "precision",
    "finetuning_type",
    "sequence_length",
    "num_gpus",
    "global_batch_size",
    "gradient_accumulation_steps",
    "tokens_per_step",
    "steps",
    "warmup_steps",
    "stable_steps",
    "mean_step_sec",
    "stable_tps",
    "forward_sec",
    "backward_sec",
    "optimizer_sec",
    "cpu_threads_per_rank",
    "kt_owner_threads",
    "process_tree_peak_gb",
    "max_gpu_task_peak_gib",
    "exit_code",
    "run_dir",
)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return data


def nested_mean(data: dict[str, Any], phase: str) -> float | None:
    value = (
        (data.get("aggregate_stable") or {})
        .get(phase, {})
        .get("mean_sec")
    )
    return float(value) if isinstance(value, (int, float)) else None


def collect_case(config_path: Path) -> dict[str, Any]:
    run_dir = config_path.parent
    config = read_json(config_path)
    exit_path = run_dir / "exit_code.txt"
    exit_text = (
        exit_path.read_text(encoding="utf-8").strip()
        if exit_path.is_file()
        else "MISSING"
    )
    timing_path = run_dir / "step_timing" / "step_timing.json"
    timing = read_json(timing_path) if timing_path.is_file() else {}
    memory_path = run_dir / "memory_summary.json"
    memory = read_json(memory_path) if memory_path.is_file() else {}

    if exit_text == "DRY_RUN":
        status = "DRY_RUN"
    elif exit_text == "0" and timing:
        status = "SUCCESS"
    else:
        status = "FAILED"

    gpu_peaks = [
        item.get("task_peak_gib")
        for item in (memory.get("gpu_peaks") or {}).values()
        if isinstance(item, dict)
        and isinstance(item.get("task_peak_gib"), (int, float))
    ]
    tps = timing.get("tps_attribution") or {}
    return {
        "status": status,
        "backend": config.get("backend"),
        "profile": config.get("profile"),
        "precision": config.get("precision"),
        "finetuning_type": config.get("finetuning_type"),
        "sequence_length": config.get("sequence_length"),
        "num_gpus": config.get("num_gpus"),
        "global_batch_size": config.get("global_batch_size"),
        "gradient_accumulation_steps": config.get(
            "gradient_accumulation_steps"
        ),
        "tokens_per_step": config.get("tokens_per_step"),
        "steps": config.get("steps"),
        "warmup_steps": config.get("warmup_steps"),
        "stable_steps": timing.get("num_stable_steps"),
        "mean_step_sec": tps.get("mean_stable_step_sec"),
        "stable_tps": tps.get("stable_tps"),
        "forward_sec": nested_mean(timing, "forward_sec"),
        "backward_sec": nested_mean(timing, "backward_sec"),
        "optimizer_sec": nested_mean(timing, "optimizer_sec"),
        "cpu_threads_per_rank": config.get("cpu_threads_per_rank"),
        "kt_owner_threads": config.get("kt_owner_threads"),
        "process_tree_peak_gb": memory.get(
            "process_tree_peak_gb_decimal"
        ),
        "max_gpu_task_peak_gib": max(gpu_peaks) if gpu_peaks else None,
        "exit_code": exit_text,
        "run_dir": str(run_dir),
    }


def display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(root: Path, rows: list[dict[str, Any]]) -> None:
    with (root / "sweep_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    success = sum(row["status"] == "SUCCESS" for row in rows)
    failed = sum(row["status"] == "FAILED" for row in rows)
    dry_run = sum(row["status"] == "DRY_RUN" for row in rows)
    backends = sorted(
        {
            str(row["backend"])
            for row in rows
            if row.get("backend") is not None
        }
    )
    lines = [
        "# GLM-4.5-Air BF16 Full-Finetuning Sweep",
        "",
        "- Profiles: `server` (8 GPUs, global batch 8) and/or "
        "`consumer` (2 GPUs, global batch 2).",
        f"- Backend: `{', '.join(backends)}`",
        "- Each sequence length runs in an independent process.",
        "- TPS excludes configured warm-up optimizer steps.",
        "- DeepSpeed/KTransformers use coarse host-wall timing without forced "
        "CUDA synchronization; MegaTrain retains backend-required synchronization.",
        "- CPU/GPU resource sampling runs outside the measured phase path.",
        "",
        f"Cases: {len(rows)}; success: {success}; failed: {failed}; dry-run: {dry_run}.",
        "",
        "| Profile | GPUs | Seq | Status | Stable steps | Step sec | TPS | Forward | Backward | Optimizer | CPU peak GB | GPU peak GiB |",
        "|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {num_gpus} | {sequence_length} | {status} | {stable_steps} | "
            "{mean_step_sec} | {stable_tps} | {forward_sec} | "
            "{backward_sec} | {optimizer_sec} | "
            "{process_tree_peak_gb} | {max_gpu_task_peak_gib} |".format(
                profile=display(row["profile"]),
                num_gpus=display(row["num_gpus"], 0),
                sequence_length=display(row["sequence_length"], 0),
                status=display(row["status"]),
                stable_steps=display(row["stable_steps"], 0),
                mean_step_sec=display(row["mean_step_sec"]),
                stable_tps=display(row["stable_tps"], 2),
                forward_sec=display(row["forward_sec"]),
                backward_sec=display(row["backward_sec"]),
                optimizer_sec=display(row["optimizer_sec"]),
                process_tree_peak_gb=display(
                    row["process_tree_peak_gb"], 2
                ),
                max_gpu_task_peak_gib=display(
                    row["max_gpu_task_peak_gib"], 2
                ),
            )
        )
    lines.extend(
        [
            "",
            "Machine-readable results: `sweep_results.csv`.",
            "",
        ]
    )
    (root / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    profile_order = {"server": 0, "consumer": 1}
    config_paths = [
        path
        for path in root.glob("*/seq_*/run_config.json")
        if path.parent.parent.name.startswith(("server_", "consumer_"))
    ]
    config_paths.sort(
        key=lambda path: (
            profile_order.get(path.parent.parent.name.split("_", 1)[0], 99),
            int(path.parent.name.removeprefix("seq_")),
        )
    )
    if not config_paths:
        raise FileNotFoundError(f"No run_config.json files under {root}")
    rows = [collect_case(path) for path in config_paths]
    write_csv(root, rows)
    write_summary(root, rows)
    print(
        f"[aggregate] cases={len(rows)} "
        f"summary={root / 'summary.md'} "
        f"csv={root / 'sweep_results.csv'}"
    )


if __name__ == "__main__":
    main()
