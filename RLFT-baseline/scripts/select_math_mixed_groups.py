#!/usr/bin/env python3
"""Select non-truncated mixed-reward prompts from an evaluate_math report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-successes", type=int, required=True)
    parser.add_argument("--max-successes", type=int, required=True)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--require-all-answer-extracted", action="store_true")
    parser.add_argument("--require-all-format-ok", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(row: dict[str, Any], seed: int) -> str:
    extra = row.get("extra_info") or {}
    return hashlib.sha256(f"{seed}\0{extra.get('sample_hash')}".encode()).hexdigest()


def balanced_limit(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: stable_key(item, seed)):
        extra = row.get("extra_info") or {}
        buckets[(str(extra.get("raw_source", "unknown")), str(extra.get("problem_type", "unknown")))].append(row)
    selected = []
    active = sorted(buckets)
    while active and len(selected) < size:
        next_active = []
        for bucket in active:
            if len(selected) >= size:
                break
            selected.append(buckets[bucket].popleft())
            if buckets[bucket]:
                next_active.append(bucket)
        active = next_active
    return selected


def main() -> int:
    args = parse_args()
    for path in (args.report, args.source_data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.min_successes < 0 or args.max_successes < args.min_successes:
        raise ValueError("invalid success range")

    from datasets import Dataset

    report = json.loads(args.report.read_text(encoding="utf-8"))
    source = Dataset.from_parquet(str(args.source_data))
    source_hash = sha256_file(args.source_data)
    if report.get("data_sha256") != source_hash:
        raise RuntimeError("evaluation report and source parquet hashes differ")
    samples_per_question = int(report["metrics"]["samples_per_question"])
    if args.max_successes >= samples_per_question:
        raise ValueError("max successes must be less than samples per question for a mixed group")

    generations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for generation in report.get("generations", []):
        generations[int(generation["problem_index"])].append(generation)
    per_problem = {int(item["problem_index"]): item for item in report.get("per_problem", [])}

    qualified = []
    rejected = Counter()
    for problem_index, row in enumerate(source):
        summary = per_problem.get(problem_index)
        samples = generations.get(problem_index, [])
        if summary is None or len(samples) != samples_per_question:
            rejected["incomplete_report_group"] += 1
            continue
        successes = int(summary["successes"])
        if not args.min_successes <= successes <= args.max_successes:
            rejected["outside_success_range"] += 1
            continue
        if any(sample.get("finish_reason") == "length" for sample in samples):
            rejected["contains_length_truncation"] += 1
            continue
        if any(float((sample.get("verifier") or {}).get("parser_error", 0.0)) > 0 for sample in samples):
            rejected["contains_parser_error"] += 1
            continue
        if args.require_all_answer_extracted and any(
            not bool((sample.get("verifier") or {}).get("answer_extracted"))
            for sample in samples
        ):
            rejected["contains_unextracted_answer"] += 1
            continue
        if args.require_all_format_ok and any(
            not bool((sample.get("verifier") or {}).get("format_ok"))
            for sample in samples
        ):
            rejected["contains_format_failure"] += 1
            continue
        item = copy.deepcopy(dict(row))
        verifier_prompt = "\n".join(
            str(message.get("content", ""))
            for message in item["prompt"]
            if message.get("role") == "user"
        )
        item.setdefault("extra_info", {}).update(
            {
                "verifier_prompt": verifier_prompt,
                "mixed_selection_report": str(args.report.resolve()),
                "mixed_successes": successes,
                "mixed_samples": samples_per_question,
            }
        )
        qualified.append(item)

    selected = qualified
    if args.max_rows is not None and len(selected) > args.max_rows:
        selected = balanced_limit(selected, args.max_rows, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(selected).to_parquet(str(args.output))
    selection_report = {
        "protocol": "math-clean-mixed-selection-v1",
        "input_report": str(args.report.resolve()),
        "source_data": {"path": str(args.source_data.resolve()), "rows": len(source), "sha256": source_hash},
        "criteria": {
            "successes": [args.min_successes, args.max_successes],
            "samples_per_question": samples_per_question,
            "no_length_truncation": True,
            "no_parser_error": True,
            "all_answers_extracted": args.require_all_answer_extracted,
            "all_formats_ok": args.require_all_format_ok,
            "min_rows": args.min_rows,
            "max_rows": args.max_rows,
        },
        "qualified_rows": len(qualified),
        "selected_rows": len(selected),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
        "gate_passed": len(selected) >= args.min_rows,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(selection_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection_report, ensure_ascii=False, indent=2))
    return 0 if selection_report["gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
