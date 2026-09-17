#!/usr/bin/env python3
"""Re-score a saved evaluate_math report without changing model generations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from evaluate_math import bootstrap_ci, correctness
from math_verify_reward import compute_score


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    for path in (args.input, args.data, ROOT / "scripts/math_verify_reward.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)

    from datasets import Dataset

    source = Dataset.from_parquet(str(args.data))
    report: dict[str, Any] = json.loads(args.input.read_text(encoding="utf-8"))
    if report.get("data_sha256") != sha256_file(args.data):
        raise RuntimeError("evaluation report and source parquet hashes differ")

    generations = []
    for old in report.get("generations", []):
        generation = copy.deepcopy(old)
        problem_index = int(generation["problem_index"])
        if not 0 <= problem_index < len(source):
            raise RuntimeError(f"invalid problem_index={problem_index}")
        row = source[problem_index]
        verifier = compute_score(
            data_source=row["data_source"],
            solution_str=generation["response"],
            ground_truth=row["reward_model"]["ground_truth"],
            extra_info={
                **(row.get("extra_info") or {}),
                "verifier_prompt": row["prompt"],
            },
        )
        correct, detail = correctness(verifier)
        generation["correct"] = correct
        generation["verifier"] = detail
        generations.append(generation)

    expected = len(source) * int(report["metrics"]["samples_per_question"])
    if len(generations) != expected:
        raise RuntimeError(f"expected {expected} generations, found {len(generations)}")

    per_problem = []
    for problem_index in range(len(source)):
        samples = [row for row in generations if int(row["problem_index"]) == problem_index]
        per_problem.append(
            {
                "problem_index": problem_index,
                "accuracy": statistics.fmean(float(row["correct"]) for row in samples),
                "successes": sum(bool(row["correct"]) for row in samples),
                "samples": len(samples),
            }
        )
    sample_indexes = sorted({int(row["sample_index"]) for row in generations})
    first_sample = [row for row in generations if int(row["sample_index"]) == sample_indexes[0]]
    question_means = [row["accuracy"] for row in per_problem]
    metrics = copy.deepcopy(report["metrics"])
    metrics.update(
        {
            "avg_at_1": statistics.fmean(float(row["correct"]) for row in first_sample),
            f"avg_at_{len(sample_indexes)}": statistics.fmean(question_means),
            "pass_at_1": statistics.fmean(float(row["correct"]) for row in first_sample),
            f"pass_at_{len(sample_indexes)}": statistics.fmean(row["successes"] > 0 for row in per_problem),
            "mixed_group_rate": statistics.fmean(
                0 < row["successes"] < row["samples"] for row in per_problem
            ),
            "bootstrap_95_ci": bootstrap_ci(question_means),
            "format_rate": statistics.fmean(
                float(row["verifier"].get("format_ok", 0.0)) for row in generations
            ),
            "answer_extracted_rate": statistics.fmean(
                float(row["verifier"].get("answer_extracted", 0.0)) for row in generations
            ),
            "parser_error_rate": statistics.fmean(
                float(row["verifier"].get("parser_error", 0.0)) for row in generations
            ),
        }
    )
    rescored = copy.deepcopy(report)
    rescored["metrics"] = metrics
    rescored["per_problem"] = per_problem
    rescored["generations"] = generations
    rescored["verifier_rescore"] = {
        "input_report": str(args.input.resolve()),
        "input_report_sha256": sha256_file(args.input),
        "verifier_path": str((ROOT / "scripts/math_verify_reward.py").resolve()),
        "verifier_sha256": sha256_file(ROOT / "scripts/math_verify_reward.py"),
        "generations_reused_without_model_inference": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rescored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
