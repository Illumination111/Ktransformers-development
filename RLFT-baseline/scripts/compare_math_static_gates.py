#!/usr/bin/env python3
"""Compare static hard-math probes and emit a fail-closed length choice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-clip-rate", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = []
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report["metrics"]
        candidates.append(
            {
                "path": str(path.resolve()),
                "max_new_tokens": int(report["sampling"]["max_new_tokens"]),
                "mixed_group_rate": float(metrics["mixed_group_rate"]),
                "truncation_rate": float(metrics["truncation_rate"]),
                "parser_error_rate": float(metrics["parser_error_rate"]),
                "avg": float(metrics[f"avg_at_{metrics['samples_per_question']}"]),
            }
        )
    candidates.sort(key=lambda item: item["max_new_tokens"])
    eligible = [
        item
        for item in candidates
        if item["parser_error_rate"] == 0.0
        and item["mixed_group_rate"] > 0.0
        and item["truncation_rate"] <= args.max_clip_rate
    ]
    selected = eligible[0] if eligible else min(candidates, key=lambda item: item["truncation_rate"])
    payload = {
        "protocol": "math-grpo-static-length-gate-v2",
        "max_clip_rate": args.max_clip_rate,
        "gate_passed": bool(eligible),
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
