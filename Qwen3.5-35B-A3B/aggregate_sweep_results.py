#!/usr/bin/env python3
"""Aggregate probe-free post-warmup phase timing and TPS for a BF16 sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


TIMING_MODE = "coarse_host_wall_no_cuda_sync"
MEGATRAIN_TIMING_MODE = "megatrain_host_wall_with_backend_cuda_sync"
RESULT_COLUMNS = [
    "backend",
    "profile",
    "benchmark_class",
    "result_validity",
    "weight_source",
    "checkpoint_compatible",
    "llamafactory_backend",
    "result_scope",
    "precision",
    "modality",
    "finetuning_type",
    "lora_rank",
    "lora_alpha",
    "lora_target",
    "model_load_architecture",
    "sequence_length",
    "num_gpus",
    "global_batch_size",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "tokens_per_step",
    "cpu_threads_per_rank",
    "kt_owner_threads",
    "cpu_thread_budget_total",
    "warmup_steps",
    "stable_steps",
    "mean_step_sec",
    "stable_tps",
    "forward_sec",
    "backward_sec",
    "optimizer_sec",
    "cpu_memory_peak_gb",
    "cpu_memory_exceeds_1tib",
    "oom_classification",
    "gpu_memory_peak_gib",
    "gpu_lifecycle_status",
    "gpu_release_confirmed",
    "memory_limit",
    "numa_policy",
    "timing_mode",
    "full_update_verified",
    "route_trace",
    "lookup_table",
    "status",
    "exit_code",
    "run_dir",
]
PLOT_DESCRIPTIONS = {
    "01_tps_vs_sequence.png": "稳定吞吐",
    "02_step_phase_times.png": "稳定 step 阶段耗时",
    "03_peak_memory_vs_sequence.png": "CPU/GPU 峰值内存",
}


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def aggregate_run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = config_path.parent
    exit_path = run_dir / "exit_code.txt"
    exit_code = exit_path.read_text(encoding="utf-8").strip() if exit_path.is_file() else "MISSING"
    timing_path = run_dir / "step_timing" / "step_timing.json"
    memory_path = run_dir / "memory_summary.json"
    gpu_lifecycle_path = run_dir / "gpu_lifecycle.json"
    row: dict[str, Any] = {
        **{key: config.get(key) for key in RESULT_COLUMNS},
        "benchmark_class": config.get(
            "benchmark_class",
            "exact_model_full_finetune",
        ),
        "stable_steps": None,
        "mean_step_sec": None,
        "stable_tps": None,
        "forward_sec": None,
        "backward_sec": None,
        "optimizer_sec": None,
        "cpu_memory_peak_gb": None,
        "cpu_memory_exceeds_1tib": None,
        "oom_classification": None,
        "gpu_memory_peak_gib": None,
        "gpu_lifecycle_status": None,
        "gpu_release_confirmed": None,
        "timing_mode": None,
        "full_update_verified": None,
        "exit_code": exit_code,
        "run_dir": str(run_dir),
    }
    if memory_path.is_file():
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        row["cpu_memory_peak_gb"] = (
            memory.get("process_tree_peak_gb_decimal")
            if memory.get("process_tree_peak_gb_decimal") is not None
            else memory.get("host_used_peak_gb_decimal")
        )
        row["cpu_memory_exceeds_1tib"] = memory.get(
            "observed_peak_exceeds_one_tib"
        )
        row["oom_classification"] = memory.get("oom_classification")
        task_gpu_peaks = [
            as_float(values.get("task_peak_gib"))
            for values in (memory.get("gpu_peaks") or {}).values()
        ]
        row["gpu_memory_peak_gib"] = max(
            (value for value in task_gpu_peaks if value is not None),
            default=None,
        )
    if gpu_lifecycle_path.is_file():
        lifecycle = json.loads(
            gpu_lifecycle_path.read_text(encoding="utf-8")
        )
        row["gpu_lifecycle_status"] = lifecycle.get("status")
        row["gpu_release_confirmed"] = lifecycle.get(
            "release_confirmed"
        )
    if exit_code == "DRY_RUN":
        row["status"] = "DRY_RUN"
        return row
    if exit_code == "87":
        row["status"] = "GPU_BUSY_NOT_OOM"
        row["oom_classification"] = "NOT_OOM_GPU_BUSY"
        return row
    if exit_code == "88":
        row["status"] = "GPU_RELEASE_UNCONFIRMED_NOT_OOM"
        row["oom_classification"] = "NOT_OOM_RELEASE_UNCONFIRMED"
        return row
    if exit_code != "0":
        row["status"] = "FAILED"
        return row
    if not timing_path.is_file():
        row["status"] = "TIMING_MISSING"
        return row

    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    stable = timing.get("aggregate_stable") or {}
    attribution = timing.get("tps_attribution") or {}
    instrumentation = timing.get("instrumentation") or {}
    row["timing_mode"] = timing.get("timing_mode")
    row["stable_steps"] = timing.get("num_stable_steps")
    row["mean_step_sec"] = (stable.get("step_total_sec") or {}).get("mean_sec")
    row["stable_tps"] = attribution.get("stable_tps")
    for key in ("forward_sec", "backward_sec", "optimizer_sec"):
        row[key] = (stable.get(key) or {}).get("mean_sec")

    expected_stable = int(config["steps"]) - int(config["warmup_steps"])
    required_values = (
        row["mean_step_sec"],
        row["stable_tps"],
        row["forward_sec"],
        row["backward_sec"],
        row["optimizer_sec"],
    )
    benchmark_class = row["benchmark_class"]
    contract_status: str | None = None
    if benchmark_class in {
        "exact_model_full_finetune",
        "exact_model_lora_finetune",
    }:
        if (
            config.get("modality") != "text_only"
            or config.get("model_load_architecture")
            != "Qwen3_5MoeForCausalLM"
            or config.get("weight_source") != "pretrained_checkpoint"
            or config.get("checkpoint_compatible") is not True
            or (
                config.get("backend") == "megatrain"
                and config.get("llamafactory_backend") is not False
            )
            or (
                config.get("backend") != "megatrain"
                and config.get("llamafactory_backend") is not True
            )
        ):
            contract_status = "MODEL_CONTRACT_MISMATCH"
        elif (
            benchmark_class == "exact_model_full_finetune"
            and config.get("finetuning_type", "full") != "full"
        ):
            contract_status = "FINETUNING_CONTRACT_MISMATCH"
        elif (
            benchmark_class == "exact_model_lora_finetune"
            and (
                config.get("finetuning_type") != "lora"
                or int(config.get("lora_rank") or 0) <= 0
                or int(config.get("lora_alpha") or 0) <= 0
                or config.get("lora_target") != "all"
            )
        ):
            contract_status = "FINETUNING_CONTRACT_MISMATCH"
    elif benchmark_class == "deployment_proxy":
        manifest_path = run_dir / "proxy_manifest.json"
        verification_path = run_dir / "full_update_verification.json"
        if (
            config.get("model_load_architecture")
            != "Qwen35ComponentIsomorphicAPTMoEProxy"
            or config.get("weight_source")
            != "deterministic_random_initialization"
            or config.get("checkpoint_compatible") is not False
            or config.get("llamafactory_backend") is not False
            or config.get("allow_end_to_end_qwen35_tps_claim") is not False
        ):
            contract_status = "PROXY_CONTRACT_MISMATCH"
        elif not manifest_path.is_file():
            contract_status = "PROXY_MANIFEST_MISSING"
        elif not verification_path.is_file():
            contract_status = "FULL_UPDATE_AUDIT_MISSING"
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verification = json.loads(
                verification_path.read_text(encoding="utf-8")
            )
            row["full_update_verified"] = verification.get(
                "valid_full_update"
            )
            if (
                manifest.get("benchmark_class") != "deployment_proxy"
                or manifest.get("proxy_architecture")
                != "qwen35_component_isomorphic"
                or manifest.get("parameter_count") != 34_660_610_688
                or manifest.get("checkpoint_compatible") is not False
                or manifest.get("real_forward_backward_optimizer_update")
                is not True
            ):
                contract_status = "PROXY_MANIFEST_MISMATCH"
            elif verification.get("valid_full_update") is not True:
                contract_status = "FULL_UPDATE_AUDIT_FAILED"
            elif config.get("result_validity") == "formal_deployment_proxy":
                route = manifest.get("route") or {}
                placement = manifest.get("placement") or {}
                versions = manifest.get("runtime_versions") or {}
                if (
                    manifest.get("result_validity")
                    != "formal_deployment_proxy"
                    or route.get("mode")
                    != "replayed_qwen35_topk_indices"
                    or not route.get("trace_sha256")
                    or placement.get("mode") != "profiled_compute_load"
                    or placement.get("deployment_profile")
                    != config.get("profile")
                    or not placement.get("lookup_sha256")
                    or versions.get("qwen35_linear_attention_fastpath")
                    is not True
                ):
                    contract_status = "FORMAL_PROXY_GUARD_FAILED"
            elif config.get("result_validity") != "smoke_only":
                contract_status = "PROXY_VALIDITY_MISSING"
    else:
        contract_status = "UNKNOWN_BENCHMARK_CLASS"

    if str(config.get("precision", "")).lower() != "bf16":
        row["status"] = "PRECISION_MISMATCH"
    elif contract_status is not None:
        row["status"] = contract_status
    elif row["timing_mode"] != (
        MEGATRAIN_TIMING_MODE
        if config.get("backend") == "megatrain"
        else TIMING_MODE
    ):
        row["status"] = "TIMING_MODE_MISMATCH"
    elif any(
        instrumentation.get(key)
        is not (
            config.get("backend") == "megatrain"
            if key == "forced_cuda_synchronize"
            else False
        )
        for key in (
            "forced_cuda_synchronize",
            "backend_internal_probes",
            "system_resource_monitor",
            "per_step_file_io",
        )
    ):
        row["status"] = "FORBIDDEN_INSTRUMENTATION"
    elif any(as_float(value) is None for value in required_values):
        row["status"] = "TIMING_FIELDS_MISSING"
    elif int(row["stable_steps"] or 0) != expected_stable:
        row["status"] = "INCOMPLETE_STABLE_WINDOW"
    else:
        row["status"] = (
            "SMOKE_ONLY"
            if benchmark_class == "deployment_proxy"
            and config.get("result_validity") == "smoke_only"
            else "OK_PROXY"
            if benchmark_class == "deployment_proxy"
            else "OK_BACKEND_SYNC"
            if config.get("backend") == "megatrain"
            else "OK"
        )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_groups(
    rows: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("benchmark_class")),
                str(row.get("backend")),
                str(row.get("profile")),
            )
        ].append(row)

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for (benchmark_class, backend, profile), group in sorted(grouped.items()):
        label = f"{backend}/{profile}"
        if len(grouped) > 1 and len({key[0] for key in grouped}) > 1:
            label = f"{benchmark_class}/{label}"
        groups.append(
            (
                label,
                sorted(group, key=lambda item: int(item["sequence_length"])),
            )
        )
    return groups


def _set_sequence_ticks(axis: Any, sequences: set[int]) -> None:
    if not sequences:
        return
    ticks = sorted(sequences)
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(value) for value in ticks])


def _show_no_data(axis: Any) -> None:
    axis.text(
        0.5,
        0.5,
        "No valid data",
        ha="center",
        va="center",
        transform=axis.transAxes,
    )


def generate_plots(output: Path, rows: list[dict[str, Any]]) -> list[Path]:
    plots_dir = output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    mpl_config = output / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = _plot_groups(rows)
    sequences = {
        int(row["sequence_length"])
        for row in rows
        if row.get("sequence_length") is not None
    }
    colors = plt.get_cmap("tab10")
    generated: list[Path] = []

    fig, axis = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for index, (label, group) in enumerate(groups):
        points = [
            (int(row["sequence_length"]), as_float(row.get("stable_tps")))
            for row in group
        ]
        valid = [(sequence, value) for sequence, value in points if value is not None]
        if not valid:
            continue
        axis.plot(
            [item[0] for item in valid],
            [item[1] for item in valid],
            marker="o",
            linewidth=2,
            color=colors(index % 10),
            label=label,
        )
        plotted = True
    _set_sequence_ticks(axis, sequences)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Stable throughput (tokens/s)")
    axis.set_title("Stable throughput vs. sequence length")
    axis.grid(True, which="both", alpha=0.25)
    if plotted:
        axis.legend()
    else:
        _show_no_data(axis)
    fig.tight_layout()
    path = plots_dir / "01_tps_vs_sequence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)

    fig, axis = plt.subplots(figsize=(10, 6))
    plotted = False
    phase_styles = {
        "forward_sec": ("Forward", "o", "-"),
        "backward_sec": ("Backward", "s", "--"),
        "optimizer_sec": ("Optimizer", "^", ":"),
    }
    for index, (label, group) in enumerate(groups):
        color = colors(index % 10)
        for key, (phase_label, marker, line_style) in phase_styles.items():
            points = [
                (int(row["sequence_length"]), as_float(row.get(key)))
                for row in group
            ]
            valid = [(sequence, value) for sequence, value in points if value is not None]
            if not valid:
                continue
            axis.plot(
                [item[0] for item in valid],
                [item[1] for item in valid],
                marker=marker,
                linestyle=line_style,
                linewidth=2,
                color=color,
                label=f"{label} {phase_label}",
            )
            plotted = True
    _set_sequence_ticks(axis, sequences)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Mean stable phase time (s)")
    axis.set_title("Stable step phase times vs. sequence length")
    axis.grid(True, which="both", alpha=0.25)
    if plotted:
        axis.legend(ncol=2)
    else:
        _show_no_data(axis)
    fig.tight_layout()
    path = plots_dir / "02_step_phase_times.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)

    fig, cpu_axis = plt.subplots(figsize=(10, 6))
    gpu_axis = cpu_axis.twinx()
    cpu_plotted = False
    gpu_plotted = False
    handles: list[Any] = []
    labels: list[str] = []
    for index, (label, group) in enumerate(groups):
        color = colors(index % 10)
        cpu_points = [
            (
                int(row["sequence_length"]),
                as_float(row.get("cpu_memory_peak_gb")),
            )
            for row in group
        ]
        valid_cpu = [
            (sequence, value)
            for sequence, value in cpu_points
            if value is not None
        ]
        if valid_cpu:
            line = cpu_axis.plot(
                [item[0] for item in valid_cpu],
                [item[1] for item in valid_cpu],
                marker="o",
                linewidth=2,
                color=color,
                label=f"{label} CPU",
            )[0]
            handles.append(line)
            labels.append(line.get_label())
            cpu_plotted = True

        gpu_points = [
            (
                int(row["sequence_length"]),
                as_float(row.get("gpu_memory_peak_gib")),
            )
            for row in group
        ]
        valid_gpu = [
            (sequence, value)
            for sequence, value in gpu_points
            if value is not None
        ]
        if valid_gpu:
            line = gpu_axis.plot(
                [item[0] for item in valid_gpu],
                [item[1] for item in valid_gpu],
                marker="^",
                linestyle="--",
                linewidth=2,
                color=color,
                label=f"{label} GPU",
            )[0]
            handles.append(line)
            labels.append(line.get_label())
            gpu_plotted = True
    _set_sequence_ticks(cpu_axis, sequences)
    cpu_axis.set_xlabel("Sequence length")
    cpu_axis.set_ylabel("CPU peak memory (decimal GB)")
    gpu_axis.set_ylabel("GPU peak memory (GiB)")
    cpu_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    gpu_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    cpu_axis.set_title("Peak CPU/GPU memory vs. sequence length")
    cpu_axis.grid(True, which="both", alpha=0.25)
    if handles:
        cpu_axis.legend(handles, labels, ncol=2)
    else:
        _show_no_data(cpu_axis)
    if not cpu_plotted:
        cpu_axis.set_yticks([])
    if not gpu_plotted:
        gpu_axis.set_yticks([])
    fig.tight_layout()
    path = plots_dir / "03_peak_memory_vs_sequence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)
    return generated


def write_markdown(
    path: Path,
    root: Path,
    rows: list[dict[str, Any]],
    plot_paths: list[Path] | None = None,
) -> None:
    lines = [
        "# Qwen3.5-35B-A3B BF16 TPS Sweep",
        "",
        f"- 结果根目录：`{root}`",
        "- 仅记录每个 optimizer step 的 forward、backward、optimizer 和 total host wall time。",
        "- CPU/GPU 内存由 step 计时路径之外的进程采样器记录，不计入 phase timer；consumer 不再设置 1 TiB cgroup hard limit，也不会因越线自动终止。",
        "- 每个 profile 从最长 sequence 开始递减；每项训练在独立进程会话中持有正常 CUDA 分配，退出后必须确认 worker 和 GPU context 已释放才会开始下一项。",
        "- GPU_BUSY_NOT_OOM 和 GPU_RELEASE_UNCONFIRMED_NOT_OOM 是资源隔离状态，不按训练 OOM 记录。",
        "- KTransformers、DeepSpeed、APTMoE 不强制 CUDA 同步；MegaTrain 后端自身包含必要的 CUDA 同步，并在 timing_mode/status 中单独标注。",
        "- CPU 峰值超过 1 TiB 与否仅作为观测结果，是否按 OOM 记录始终保留人工判断。",
        "- TPS 仅使用 `global_step > warmup_steps` 的稳定窗口。",
        "- exact-model run 从多模态源 checkpoint 仅加载 `Qwen3_5MoeForCausalLM`；proxy 只读取目标 config/tokenizer。",
        "- Full、LoRA exact-model 与 `deployment_proxy` 按 benchmark class 分组；proxy 使用随机权重，不能声明模型效果或真实 Qwen3.5 端到端训练 TPS。",
        "- 公式：`TPS = GPUs × per-device batch × sequence length × GAS / mean stable step seconds`。",
        "",
    ]
    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("benchmark_class")),
                str(row.get("backend")),
                str(row.get("profile")),
            )
        ].append(row)
    for (benchmark_class, backend, profile), group in sorted(groups.items()):
        first = group[0]
        kt_owner_threads = first.get("kt_owner_threads")
        if kt_owner_threads:
            cpu_line = (
                f"- CPU 线程：KT owner(rank0) {kt_owner_threads}；"
                f"其余 rank {first.get('cpu_threads_per_rank')}/rank；"
                f"计划合计 {first.get('cpu_thread_budget_total')}"
            )
        else:
            cpu_line = (
                f"- CPU 线程：{first.get('cpu_threads_per_rank')}/rank，"
                f"合计预算 {first.get('cpu_thread_budget_total')}"
            )
        lines.extend(
            [
                f"## {benchmark_class} / {backend} / {profile}",
                "",
                f"- GPU：{first.get('num_gpus')}；全局 batch：{first.get('global_batch_size')}；精度：{first.get('precision')}",
                f"- 模态：{first.get('modality')}；加载架构：{first.get('model_load_architecture')}",
                (
                    f"- 微调：{first.get('finetuning_type')}；LoRA rank/alpha/target："
                    f"{first.get('lora_rank')}/{first.get('lora_alpha')}/{first.get('lora_target')}"
                ),
                f"- 权重：{first.get('weight_source')}；结果有效性：{first.get('result_validity')}",
                f"- 结果范围：{first.get('result_scope')}",
                cpu_line,
                f"- 内存策略：{first.get('memory_limit')}；NUMA：{first.get('numa_policy')}",
                "",
                "| Seq | Stable steps | Mean step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | CPU peak (GB) | >1 TiB | GPU peak (GiB) | GPU release | OOM review | Status |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|",
            ]
        )
        for row in sorted(
            group,
            key=lambda item: int(item["sequence_length"]),
            reverse=True,
        ):
            lines.append(
                "| {seq} | {steps} | {step} | {tps} | {forward} | {backward} | {optimizer} | {cpu_peak} | {exceeds} | {gpu_peak} | {gpu_release} | {oom} | {status} |".format(
                    seq=row["sequence_length"],
                    steps=row.get("stable_steps") or "-",
                    step=fmt(row.get("mean_step_sec")),
                    tps=fmt(row.get("stable_tps"), 2),
                    forward=fmt(row.get("forward_sec")),
                    backward=fmt(row.get("backward_sec")),
                    optimizer=fmt(row.get("optimizer_sec")),
                    cpu_peak=fmt(row.get("cpu_memory_peak_gb"), 2),
                    exceeds=(
                        "yes"
                        if row.get("cpu_memory_exceeds_1tib") is True
                        else "no"
                        if row.get("cpu_memory_exceeds_1tib") is False
                        else "-"
                    ),
                    gpu_peak=fmt(row.get("gpu_memory_peak_gib"), 2),
                    gpu_release=(
                        "yes"
                        if row.get("gpu_release_confirmed") is True
                        else "no"
                        if row.get("gpu_release_confirmed") is False
                        else "-"
                    ),
                    oom=row.get("oom_classification") or "-",
                    status=row.get("status"),
                )
            )
        lines.append("")
    if not rows:
        lines.append("尚无 run_config.json。")
    if plot_paths:
        lines.extend(["## 聚合图", ""])
        for plot_path in plot_paths:
            description = PLOT_DESCRIPTIONS.get(plot_path.name, plot_path.stem)
            relative_path = plot_path.relative_to(path.parent)
            lines.append(f"- {description}：`{relative_path}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = sorted(
        root.glob("**/seq_*/run_config.json"),
        key=lambda path: (
            str(path.parent.parent),
            -int(re.search(r"seq_(\d+)", path.parent.name).group(1)),
        ),
    )
    rows = [aggregate_run(path) for path in configs]
    write_csv(output / "sweep_results.csv", rows)
    plot_paths = generate_plots(output, rows)
    write_markdown(output / "summary.md", root, rows, plot_paths)
    print(f"[aggregate] runs={len(rows)} -> {output / 'sweep_results.csv'}")
    for plot_path in plot_paths:
        print(f"[aggregate] plot -> {plot_path}")
    print(f"[aggregate] summary -> {output / 'summary.md'}")


if __name__ == "__main__":
    main()
