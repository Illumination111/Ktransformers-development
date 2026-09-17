#!/usr/bin/env python3
"""Merge disjoint-seed evaluate_math reports into one aligned report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from evaluate_math import bootstrap_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.input) < 2:
        raise ValueError("at least two input reports are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    first = reports[0]
    aligned_fields = (
        "protocol",
        "benchmark",
        "model_path",
        "tokenizer_path",
        "data_path",
        "data_sha256",
    )
    sampling_fields = (
        "enable_thinking",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_new_tokens",
    )
    for report in reports[1:]:
        for field in aligned_fields:
            if report.get(field) != first.get(field):
                raise RuntimeError(f"unaligned reports: {field} differs")
        for field in sampling_fields:
            if (report.get("sampling") or {}).get(field) != (
                first.get("sampling") or {}
            ).get(field):
                raise RuntimeError(f"unaligned reports: sampling.{field} differs")

    seeds: list[int] = []
    generations: list[dict[str, Any]] = []
    seen_generation_keys: set[tuple[int, int]] = set()
    for report in reports:
        report_seeds = [int(seed) for seed in report["sampling"]["seeds"]]
        if set(seeds) & set(report_seeds):
            raise RuntimeError("input reports contain overlapping seeds")
        seeds.extend(report_seeds)
        for generation in report.get("generations", []):
            seed = int(generation["seed"])
            key = (int(generation["problem_index"]), seed)
            if key in seen_generation_keys:
                raise RuntimeError(f"duplicate generation: {key}")
            seen_generation_keys.add(key)
            generations.append(copy.deepcopy(generation))
    seeds.sort()

    # A mislabeled process shard is worse than a small sample: SGLang's native
    # Engine can replay its process RNG stream when separate runs share the
    # same Engine random_seed, even though report-level sampling_seed labels
    # differ. Reject any two seeds whose complete response corpus is identical.
    response_signatures: dict[int, str] = {}
    for seed in seeds:
        responses = [
            str(row.get("response", ""))
            for row in sorted(
                (row for row in generations if int(row["seed"]) == seed),
                key=lambda row: int(row["problem_index"]),
            )
        ]
        signature = hashlib.sha256(
            json.dumps(responses, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        if signature in response_signatures.values():
            duplicate_seed = next(
                prior for prior, prior_signature in response_signatures.items()
                if prior_signature == signature
            )
            raise RuntimeError(
                f"seeds {duplicate_seed} and {seed} have identical response corpora; "
                "the Engine RNG stream was likely replayed"
            )
        response_signatures[seed] = signature

    seed_to_sample_index = {seed: index for index, seed in enumerate(seeds)}
    for generation in generations:
        generation["sample_index"] = seed_to_sample_index[int(generation["seed"])]
    generations.sort(
        key=lambda row: (int(row["problem_index"]), int(row["sample_index"]))
    )

    question_count = int(first["metrics"]["questions"])
    per_problem = []
    for problem_index in range(question_count):
        samples = [
            row for row in generations if int(row["problem_index"]) == problem_index
        ]
        if {int(row["seed"]) for row in samples} != set(seeds):
            raise RuntimeError(f"incomplete seed coverage for problem {problem_index}")
        successes = sum(bool(row["correct"]) for row in samples)
        per_problem.append(
            {
                "problem_index": problem_index,
                "accuracy": successes / len(seeds),
                "successes": successes,
                "samples": len(seeds),
            }
        )
    first_sample = [row for row in generations if int(row["sample_index"]) == 0]
    question_means = [row["accuracy"] for row in per_problem]
    metrics = {
        "questions": question_count,
        "samples_per_question": len(seeds),
        "avg_at_1": statistics.fmean(float(row["correct"]) for row in first_sample),
        f"avg_at_{len(seeds)}": statistics.fmean(question_means),
        "pass_at_1": statistics.fmean(float(row["correct"]) for row in first_sample),
        f"pass_at_{len(seeds)}": statistics.fmean(
            row["successes"] > 0 for row in per_problem
        ),
        "mixed_group_rate": statistics.fmean(
            0 < row["successes"] < row["samples"] for row in per_problem
        ),
        "bootstrap_95_ci": bootstrap_ci(question_means),
        "truncation_rate": statistics.fmean(
            row.get("finish_reason") == "length" for row in generations
        ),
        "format_rate": statistics.fmean(
            float((row.get("verifier") or {}).get("format_ok", 0.0))
            for row in generations
        ),
        "answer_extracted_rate": statistics.fmean(
            float((row.get("verifier") or {}).get("answer_extracted", 0.0))
            for row in generations
        ),
        "parser_error_rate": statistics.fmean(
            float((row.get("verifier") or {}).get("parser_error", 0.0))
            for row in generations
        ),
        "mean_response_tokens": statistics.fmean(
            int(row["response_tokens"]) for row in generations
        ),
        "max_response_tokens": max(int(row["response_tokens"]) for row in generations),
        "elapsed_seconds": max(
            float(report["metrics"].get("elapsed_seconds", 0.0)) for report in reports
        ),
    }
    merged = copy.deepcopy(first)
    merged["sampling"]["seeds"] = seeds
    merged["runtime"] = {
        **(first.get("runtime") or {}),
        "parallel_seed_shards": len(reports),
        "input_reports": [str(path.resolve()) for path in args.input],
    }
    merged["diagnostic_subset"] = True
    merged["metrics"] = metrics
    merged["per_problem"] = per_problem
    merged["generations"] = generations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
