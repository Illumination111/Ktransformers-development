#!/usr/bin/env python3
"""Audit GRPO rollout JSONL files and enforce effective-signal gates."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-response-tokens", type=int)
    parser.add_argument("--min-effective-steps", type=int, default=0)
    parser.add_argument("--min-mixed-group-rate", type=float, default=0.0)
    parser.add_argument("--min-format-rate", type=float, default=0.0)
    parser.add_argument("--min-answer-extracted-rate", type=float, default=0.0)
    parser.add_argument("--max-clip-rate", type=float, default=1.0)
    return parser.parse_args()


def numbered_jsonl(path: Path) -> list[Path]:
    def key(item: Path) -> tuple[int, str]:
        try:
            return int(item.stem), item.name
        except ValueError:
            return 2**63 - 1, item.name

    return sorted(path.glob("*.jsonl"), key=key)


def mean_field(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return statistics.fmean(values) if values else None


def main() -> int:
    args = parse_args()
    files = numbered_jsonl(args.rollout_dir)
    if not files:
        raise FileNotFoundError(f"no JSONL rollout files in {args.rollout_dir}")

    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                for required in ("input", "output", "score", "step"):
                    if required not in row:
                        raise ValueError(f"missing {required!r} at {path}:{line_number}")
                rows.append(row)

    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, trust_remote_code=True)

    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    response_lengths: list[int] = []
    clipped = 0
    for row in rows:
        groups[(int(row["step"]), str(row["input"]))].append(row)
        if tokenizer is not None:
            length = len(tokenizer.encode(str(row["output"]), add_special_tokens=False))
            response_lengths.append(length)
            if args.max_response_tokens is not None and length >= args.max_response_tokens:
                clipped += 1

    mixed_groups = 0
    effective_steps: set[int] = set()
    group_sizes: list[int] = []
    all_zero_groups = 0
    all_one_groups = 0
    for (step, _), samples in groups.items():
        scores = [float(item["score"]) for item in samples]
        accuracies = [float(item["acc"]) for item in samples if "acc" in item]
        group_sizes.append(len(samples))
        if max(scores) - min(scores) > 1e-12:
            mixed_groups += 1
            effective_steps.add(step)
        elif len(accuracies) == len(samples) and all(acc == 0.0 for acc in accuracies):
            all_zero_groups += 1
        elif len(accuracies) == len(samples) and all(acc == 1.0 for acc in accuracies):
            all_one_groups += 1
        elif not accuracies and all(score == 0.0 for score in scores):
            all_zero_groups += 1
        elif not accuracies and all(score == 1.0 for score in scores):
            all_one_groups += 1

    steps = sorted({int(row["step"]) for row in rows})
    metrics: dict[str, Any] = {
        "protocol": "math-grpo-rollout-audit-v2",
        "rollout_dir": str(args.rollout_dir.resolve()),
        "files": len(files),
        "rows": len(rows),
        "steps": len(steps),
        "step_numbers": steps,
        "groups": len(groups),
        "group_size_min": min(group_sizes),
        "group_size_max": max(group_sizes),
        "mixed_groups": mixed_groups,
        "mixed_group_rate": mixed_groups / len(groups),
        "all_zero_groups": all_zero_groups,
        "all_one_groups": all_one_groups,
        "effective_steps": len(effective_steps),
        "effective_step_numbers": sorted(effective_steps),
        "score_mean": statistics.fmean(float(row["score"]) for row in rows),
        "format_rate": mean_field(rows, "format_ok"),
        "answer_extracted_rate": mean_field(rows, "answer_extracted"),
        "parser_error_rate": mean_field(rows, "parser_error"),
    }
    if response_lengths:
        metrics.update(
            {
                "response_tokens_mean": statistics.fmean(response_lengths),
                "response_tokens_max": max(response_lengths),
                "clip_rate": clipped / len(rows) if args.max_response_tokens is not None else None,
            }
        )

    failures = []
    if metrics["effective_steps"] < args.min_effective_steps:
        failures.append(f"effective_steps={metrics['effective_steps']} < {args.min_effective_steps}")
    if metrics["mixed_group_rate"] < args.min_mixed_group_rate:
        failures.append(f"mixed_group_rate={metrics['mixed_group_rate']:.4f} < {args.min_mixed_group_rate:.4f}")
    format_rate = metrics.get("format_rate")
    if format_rate is None or format_rate < args.min_format_rate:
        failures.append(f"format_rate={format_rate} < {args.min_format_rate:.4f}")
    answer_extracted_rate = metrics.get("answer_extracted_rate")
    if answer_extracted_rate is None or answer_extracted_rate < args.min_answer_extracted_rate:
        failures.append(
            f"answer_extracted_rate={answer_extracted_rate} < {args.min_answer_extracted_rate:.4f}"
        )
    clip_rate = metrics.get("clip_rate")
    if clip_rate is not None and clip_rate > args.max_clip_rate:
        failures.append(f"clip_rate={clip_rate:.4f} > {args.max_clip_rate:.4f}")
    metrics["gate_passed"] = not failures
    metrics["gate_failures"] = failures

    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
