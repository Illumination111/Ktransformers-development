#!/usr/bin/env python3
"""Combine step-0/final benchmark reports into B0 machine and Markdown summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/home/wubowen/Ktransformers-development/RLFT-baseline")


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


def main() -> int:
    args = parse_args()
    reports = {
        "step0": {"math500": load(args.step0_math500), "aime2024": load(args.step0_aime2024)},
        "final": {"math500": load(args.final_math500), "aime2024": load(args.final_aime2024)},
    }
    summary: dict[str, Any] = {"protocol": "qwen3-30b-a3b-verl-grpo-b0", "benchmarks": {}}
    lines = ["# Qwen3-30B-A3B veRL GRPO B0 结果", "", "| Benchmark | Step 0 | Final | Delta |", "|---|---:|---:|---:|"]
    for benchmark, key in (("math500", "avg_at_4"), ("aime2024", "avg_at_16")):
        before = float(reports["step0"][benchmark]["metrics"][key])
        after = float(reports["final"][benchmark]["metrics"][key])
        summary["benchmarks"][benchmark] = {
            "metric": key,
            "step0": before,
            "final": after,
            "delta": after - before,
            "step0_ci": reports["step0"][benchmark]["metrics"]["bootstrap_95_ci"],
            "final_ci": reports["final"][benchmark]["metrics"]["bootstrap_95_ci"],
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
