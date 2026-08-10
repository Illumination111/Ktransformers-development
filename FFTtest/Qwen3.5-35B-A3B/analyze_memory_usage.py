#!/usr/bin/env python3
"""Render CPU/GPU memory plots and a manual 1-TiB review report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze import load_monitor_csv, plot_cpu_ram, plot_gpu_memory


ONE_TIB_BYTES = 1 << 40


def _peak(values: list[float]) -> float | None:
    return max(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--monitor-csv", type=Path)
    parser.add_argument("--phase")
    parser.add_argument(
        "--require-cgroup-memory",
        action="store_true",
        help="Fail instead of falling back to RSS when cgroup samples are absent",
    )
    args = parser.parse_args()
    log_dir = args.log_dir.resolve()
    monitor_csv = (
        args.monitor_csv.resolve()
        if args.monitor_csv is not None
        else log_dir / "monitor.csv"
    )
    monitor = load_monitor_csv(monitor_csv, phase=args.phase)
    plots_dir = log_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_gpu_memory(monitor, plots_dir)
    plot_cpu_ram(monitor, plots_dir)

    proc_peak_gb = _peak(monitor.get("proc_ram_gb", []))
    cgroup_peak_gb = _peak(monitor.get("cgroup_memory_gb", []))
    cgroup_swap_peak_gb = _peak(monitor.get("cgroup_swap_gb", []))
    host_peak_gb = _peak(monitor.get("ram_used_gb", []))
    cgroup_metrics_valid = bool(monitor.get("has_cgroup_memory"))
    process_metrics_valid = proc_peak_gb is not None and proc_peak_gb > 0
    if args.require_cgroup_memory and not cgroup_metrics_valid:
        raise RuntimeError(
            "dedicated cgroup memory samples are required but absent from monitor.csv"
        )
    observed_peak_gb = (
        cgroup_peak_gb
        if cgroup_metrics_valid
        else proc_peak_gb
        if process_metrics_valid
        else host_peak_gb
    )
    observed_peak_bytes = (
        int(observed_peak_gb * 1_000_000_000)
        if observed_peak_gb is not None
        else None
    )
    exceeded = (
        observed_peak_bytes > ONE_TIB_BYTES
        if observed_peak_bytes is not None
        else None
    )
    gpu_peaks: dict[str, dict[str, float | None]] = {}
    for index, gpu in sorted((monitor.get("gpus") or {}).items()):
        task_series = gpu.get("proc_mem") or gpu.get("mem_used") or []
        gpu_peaks[str(index)] = {
            "task_peak_gib": _peak(task_series),
            "device_peak_gib": _peak(gpu.get("mem_used") or []),
            "device_total_gib": gpu.get("mem_total"),
        }

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "monitor_csv": str(monitor_csv),
        "monitor_phase": args.phase,
        "monitor_samples": len(monitor.get("elapsed", [])),
        "cpu_memory_scope": (
            "dedicated_cgroup_v2_memory_current"
            if cgroup_metrics_valid
            else "training_process_tree_rss_sum"
            if process_metrics_valid
            else "host_used_memory_fallback"
        ),
        "cpu_memory_peak_gb_decimal": observed_peak_gb,
        "cgroup_memory_peak_gb_decimal": cgroup_peak_gb,
        "cgroup_swap_peak_gb_decimal": cgroup_swap_peak_gb,
        "process_tree_peak_gb_decimal": proc_peak_gb,
        "host_used_peak_gb_decimal": host_peak_gb,
        "one_tib_threshold_bytes": ONE_TIB_BYTES,
        "one_tib_threshold_gb_decimal": ONE_TIB_BYTES / 1_000_000_000,
        "observed_peak_exceeds_one_tib": exceeded,
        "automatic_termination_enabled": False,
        "automatic_oom_classification_enabled": False,
        "oom_classification": "MANUAL_REVIEW_REQUIRED",
        "gpu_peaks": gpu_peaks,
        "plots": {
            "gpu_memory": str(plots_dir / "01_gpu_memory.png"),
            "cpu_memory": str(plots_dir / "02_cpu_ram.png"),
        },
    }
    (log_dir / "memory_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    peak_text = (
        f"{observed_peak_gb:.2f} GB"
        if observed_peak_gb is not None
        else "无有效采样"
    )
    exceeded_text = (
        "是" if exceeded is True else "否" if exceeded is False else "未知"
    )
    lines = [
        "# 全量微调 CPU/GPU 内存审阅",
        "",
        f"- 采样数：{report['monitor_samples']}",
        f"- CPU 内存统计口径：`{report['cpu_memory_scope']}`",
        f"- CPU 观测峰值：{peak_text}",
        f"- 1 TiB 阈值：{ONE_TIB_BYTES / 1_000_000_000:.2f} GB（{ONE_TIB_BYTES} bytes）",
        f"- 观测峰值是否超过 1 TiB：{exceeded_text}",
        "- 自动终止：关闭",
        "- 自动 OOM 归类：关闭",
        "- 最终 OOM 结论：**需要人工结合曲线、日志和采样口径判断**",
        "",
        "## 可视化",
        "",
        "- GPU 显存：`plots/01_gpu_memory.png`",
        "- CPU 内存：`plots/02_cpu_ram.png`",
        "",
        "CPU 主指标优先使用独立 cgroup 的 memory.current，共享页在 cgroup 内只计费一次。"
        "进程树 RSS 之和仍保留为诊断数据，不用于后端内存排名。",
        "",
    ]
    (log_dir / "memory_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        f"[memory] peak={peak_text}, exceeds_1TiB={exceeded_text}, "
        f"manual_review={log_dir / 'memory_summary.md'}"
    )


if __name__ == "__main__":
    main()
