#!/usr/bin/env python3
"""Aggregate Qwen3.5-122B-A10B server sweep results for all three backends."""

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
    "benchmark_class",
    "result_validity",
    "result_marker",
    "formal_claim_allowed",
    "target_model",
    "proxy_architecture",
    "expected_text_parameters",
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
    "cpu_memory_scope",
    "cgroup_memory_peak_gb",
    "process_tree_peak_gb",
    "host_used_peak_gb",
    "max_gpu_task_peak_gib",
    "timing_mode",
    "full_update_verified",
    "route_mode",
    "placement_mode",
    "exit_code",
    "run_dir",
)

EXPECTED_PROXY_PARAMETERS = 122_111_526_912
EXPECTED_PROXY_SHAPE = {
    "num_hidden_layers": 48,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "hidden_size": 3072,
    "moe_intermediate_size": 1024,
}
EXPECTED_PROXY_ARCHITECTURE = "qwen35_122b_component_isomorphic"
EXPECTED_MODEL_LOAD_ARCHITECTURE = (
    "Qwen35_122BComponentIsomorphicAPTMoEProxy"
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


def proxy_contract_status(
    config: dict[str, Any],
    manifest: dict[str, Any],
    verification: dict[str, Any],
) -> str:
    if (
        config.get("model_load_architecture")
        != EXPECTED_MODEL_LOAD_ARCHITECTURE
        or config.get("proxy_architecture") != EXPECTED_PROXY_ARCHITECTURE
        or config.get("expected_text_parameters")
        != EXPECTED_PROXY_PARAMETERS
        or config.get("model_shape") != EXPECTED_PROXY_SHAPE
        or config.get("weight_source")
        != "deterministic_random_initialization"
        or config.get("checkpoint_compatible") is not False
        or config.get("llamafactory_backend") is not False
    ):
        return "PROXY_CONFIG_MISMATCH"
    if (
        manifest.get("benchmark_class") != "deployment_proxy"
        or manifest.get("proxy_architecture")
        != EXPECTED_PROXY_ARCHITECTURE
        or manifest.get("parameter_count") != EXPECTED_PROXY_PARAMETERS
        or {
            key: (manifest.get("model_shape") or {}).get(key)
            for key in EXPECTED_PROXY_SHAPE
        }
        != EXPECTED_PROXY_SHAPE
        or manifest.get("checkpoint_compatible") is not False
        or manifest.get("real_forward_backward_optimizer_update") is not True
    ):
        return "PROXY_MANIFEST_MISMATCH"
    if (
        verification.get("valid_full_update") is not True
        or verification.get("optimizer_parameter_count")
        != EXPECTED_PROXY_PARAMETERS
    ):
        return "FULL_UPDATE_AUDIT_FAILED"

    validity = config.get("result_validity")
    route = manifest.get("route") or {}
    placement = manifest.get("placement") or {}
    if validity == "smoke_only":
        if (
            config.get("result_marker") != "SMOKE_ONLY"
            or config.get("formal_claim_allowed") is not False
            or manifest.get("result_validity") != "smoke_only"
            or manifest.get("result_marker") != "SMOKE_ONLY"
            or manifest.get("formal_claim_allowed") is not False
            or (
                route.get("mode")
                not in {
                    "random_router_synthetic",
                    "synthetic_trace_smoke_only",
                    "replayed_qwen35_122b_topk_indices",
                }
            )
            or (
                placement.get("mode")
                not in {"unprofiled_fraction", "profiled_compute_load"}
            )
            or (
                route.get("mode") == "replayed_qwen35_122b_topk_indices"
                and placement.get("mode") == "profiled_compute_load"
            )
        ):
            return "SMOKE_GUARD_FAILED"
        return "SMOKE_ONLY"
    if validity != "formal_deployment_proxy":
        return "PROXY_VALIDITY_MISSING"
    versions = manifest.get("runtime_versions") or {}
    if (
        config.get("result_marker") != "FORMAL"
        or config.get("formal_claim_allowed") is not True
        or manifest.get("result_validity") != "formal_deployment_proxy"
        or manifest.get("result_marker") != "FORMAL"
        or manifest.get("formal_claim_allowed") is not True
        or route.get("mode") != "replayed_qwen35_122b_topk_indices"
        or not route.get("trace_sha256")
        or placement.get("mode") != "profiled_compute_load"
        or placement.get("deployment_profile") != config.get("profile")
        or not placement.get("lookup_sha256")
        or versions.get("linear_attention_fastpath") is not True
    ):
        return "FORMAL_PROXY_GUARD_FAILED"
    return "OK_PROXY"


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
    manifest_path = run_dir / "proxy_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    verification_path = run_dir / "full_update_verification.json"
    verification = (
        read_json(verification_path)
        if verification_path.is_file()
        else {}
    )
    benchmark_class = config.get(
        "benchmark_class",
        "exact_model_full_finetune",
    )

    if exit_text == "DRY_RUN":
        status = "DRY_RUN"
    elif exit_text == "0" and timing:
        if benchmark_class == "deployment_proxy":
            if not manifest:
                status = "PROXY_MANIFEST_MISSING"
            elif not verification:
                status = "FULL_UPDATE_AUDIT_MISSING"
            else:
                status = proxy_contract_status(
                    config,
                    manifest,
                    verification,
                )
        else:
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
        "benchmark_class": benchmark_class,
        "result_validity": config.get("result_validity"),
        "result_marker": config.get("result_marker"),
        "formal_claim_allowed": config.get("formal_claim_allowed"),
        "target_model": config.get("target_model"),
        "proxy_architecture": config.get("proxy_architecture"),
        "expected_text_parameters": config.get(
            "expected_text_parameters"
        ),
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
        "cpu_memory_scope": memory.get("cpu_memory_scope"),
        "cgroup_memory_peak_gb": memory.get(
            "cgroup_memory_peak_gb_decimal"
        ),
        "process_tree_peak_gb": memory.get(
            "process_tree_peak_gb_decimal"
        ),
        "host_used_peak_gb": memory.get("host_used_peak_gb_decimal"),
        "max_gpu_task_peak_gib": max(gpu_peaks) if gpu_peaks else None,
        "timing_mode": timing.get("timing_mode"),
        "full_update_verified": verification.get("valid_full_update"),
        "route_mode": (manifest.get("route") or {}).get("mode"),
        "placement_mode": (manifest.get("placement") or {}).get("mode"),
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
    success = sum(
        row["status"] in {"SUCCESS", "OK_PROXY"} for row in rows
    )
    smoke = sum(row["status"] == "SMOKE_ONLY" for row in rows)
    failed = sum(
        row["status"]
        not in {"SUCCESS", "OK_PROXY", "SMOKE_ONLY", "DRY_RUN"}
        for row in rows
    )
    dry_run = sum(row["status"] == "DRY_RUN" for row in rows)
    backends = sorted(
        {
            str(row["backend"])
            for row in rows
            if row.get("backend") is not None
        }
    )
    lines = [
        "# Qwen3.5-122B-A10B Text-Only BF16 Exact/Proxy Sweep",
        "",
        "- Profile: `server` (8 GPUs, global batch 8).",
        f"- Backend: `{', '.join(backends)}`",
        "- Each sequence length runs in an independent process.",
        "- TPS excludes configured warm-up optimizer steps.",
        "- KTransformers uses coarse host-wall timing without forced CUDA synchronization.",
        "- CPU memory reports dedicated cgroup v2 `memory.current` when available, plus process-tree RSS and whole-host used peaks; process-tree RSS can double-count shared pages.",
        "- CPU/GPU resource sampling runs outside the measured phase path.",
        "- APTMoE synthetic-routing or unprofiled-placement results are labeled `SMOKE_ONLY` and are excluded from formal proxy claims.",
        "",
        f"Cases: {len(rows)}; formal success: {success}; smoke-only: {smoke}; failed: {failed}; dry-run: {dry_run}.",
        "",
        "| Class | Marker | GPUs | Seq | Status | Stable steps | Step sec | TPS | Forward | Backward | Optimizer | CPU cgroup GB | Process RSS GB | Host used GB | GPU peak GiB |",
        "|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {benchmark_class} | {result_marker} | {num_gpus} | {sequence_length} | {status} | {stable_steps} | "
            "{mean_step_sec} | {stable_tps} | {forward_sec} | "
            "{backward_sec} | {optimizer_sec} | "
            "{cgroup_memory_peak_gb} | {process_tree_peak_gb} | "
            "{host_used_peak_gb} | {max_gpu_task_peak_gib} |".format(
                num_gpus=display(row["num_gpus"], 0),
                benchmark_class=display(row["benchmark_class"]),
                result_marker=display(row["result_marker"]),
                sequence_length=display(row["sequence_length"], 0),
                status=display(row["status"]),
                stable_steps=display(row["stable_steps"], 0),
                mean_step_sec=display(row["mean_step_sec"]),
                stable_tps=display(row["stable_tps"], 2),
                forward_sec=display(row["forward_sec"]),
                backward_sec=display(row["backward_sec"]),
                optimizer_sec=display(row["optimizer_sec"]),
                cgroup_memory_peak_gb=display(
                    row["cgroup_memory_peak_gb"], 2
                ),
                process_tree_peak_gb=display(
                    row["process_tree_peak_gb"], 2
                ),
                host_used_peak_gb=display(
                    row["host_used_peak_gb"], 2
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
    config_paths = [
        path
        for path in root.glob("*/seq_*/run_config.json")
        if path.parent.parent.name.startswith("server_")
    ]
    config_paths.sort(
        key=lambda path: int(path.parent.name.removeprefix("seq_"))
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
