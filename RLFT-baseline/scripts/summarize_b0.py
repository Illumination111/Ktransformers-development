#!/usr/bin/env python3
"""Combine step-0/final benchmark reports into B0 machine and Markdown summaries."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for phase in ("step0", "final"):
        for benchmark in ("math500", "aime2024"):
            parser.add_argument(f"--{phase}-{benchmark}", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=ROOT / "metrics" / "b0_summary.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "reports" / "b0_summary.md")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def avg_metric(report: dict[str, Any]) -> tuple[str, float]:
    count = int(report["metrics"]["samples_per_question"])
    key = f"avg_at_{count}"
    return key, float(report["metrics"][key])


def assert_same_protocol(before: dict[str, Any], after: dict[str, Any], benchmark: str) -> None:
    comparable_keys = ("data_sha256", "sampling")
    for key in comparable_keys:
        if before.get(key) != after.get(key):
            raise ValueError(f"{benchmark} protocol mismatch for {key}: {before.get(key)!r} != {after.get(key)!r}")
    if before["metrics"]["questions"] != after["metrics"]["questions"]:
        raise ValueError(f"{benchmark} question count differs")


def paired_delta_ci(before: dict[str, Any], after: dict[str, Any], seed: int = 42) -> list[float]:
    left = before["per_problem"]
    right = after["per_problem"]
    if len(left) != len(right):
        raise ValueError("paired reports contain different problem counts")
    deltas = [float(b["accuracy"]) - float(a["accuracy"]) for a, b in zip(left, right, strict=True)]
    rng = random.Random(seed)
    means = [statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(10_000)]
    means.sort()
    return [means[250], means[9749]]


def main() -> int:
    args = parse_args()
    reports = {
        "step0": {"math500": load(args.step0_math500), "aime2024": load(args.step0_aime2024)},
        "final": {"math500": load(args.final_math500), "aime2024": load(args.final_aime2024)},
    }
    summary: dict[str, Any] = {"protocol": "qwen3-30b-a3b-verl-grpo-b0", "benchmarks": {}}
    lines = ["# Qwen3-30B-A3B veRL GRPO B0 结果", "", "| Benchmark | Step 0 | Final | Delta |", "|---|---:|---:|---:|"]
    for benchmark in ("math500", "aime2024"):
        step0_report = reports["step0"][benchmark]
        final_report = reports["final"][benchmark]
        assert_same_protocol(step0_report, final_report, benchmark)
        before_key, before = avg_metric(step0_report)
        after_key, after = avg_metric(final_report)
        if before_key != after_key:
            raise ValueError(f"{benchmark} sampling count differs: {before_key} != {after_key}")
        key = before_key
        summary["benchmarks"][benchmark] = {
            "metric": key,
            "step0": before,
            "final": after,
            "delta": after - before,
            "step0_ci": step0_report["metrics"]["bootstrap_95_ci"],
            "final_ci": final_report["metrics"]["bootstrap_95_ci"],
            "paired_delta_95_ci": paired_delta_ci(step0_report, final_report),
            "step0_pass_at_n": step0_report["metrics"][f"pass_at_{step0_report['metrics']['samples_per_question']}"],
            "final_pass_at_n": final_report["metrics"][f"pass_at_{final_report['metrics']['samples_per_question']}"],
        }
        lines.append(f"| {benchmark} {key} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    lines.extend(["", "本文件由固定汇总脚本生成；完整生成、逐题评分与置信区间保存在对应 JSON。", ""])
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text("\n".join(lines), encoding="utf-8")
    print(args.json_output)
    print(args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
