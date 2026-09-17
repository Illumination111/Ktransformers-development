#!/usr/bin/env python3
"""Compare two aligned math-evaluation reports and enforce an SFT gate."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-answer-extracted-rate", type=float, default=0.99)
    parser.add_argument("--min-format-rate", type=float, default=0.95)
    parser.add_argument("--max-parser-error-rate", type=float, default=0.0)
    parser.add_argument("--max-truncation-rate", type=float, default=0.10)
    parser.add_argument("--max-bootstrap-regression", type=float, default=0.03)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def paired_bootstrap_ci(
    deltas: list[float], seed: int, replicates: int
) -> list[float]:
    if not deltas:
        raise ValueError("no paired question deltas")
    if replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    rng = random.Random(seed)
    means = [
        statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(replicates)
    ]
    means.sort()
    return [
        means[int(0.025 * replicates)],
        means[min(replicates - 1, int(0.975 * replicates))],
    ]


def keyed_generations(report: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
    keyed = {}
    for row in report.get("generations", []):
        key = (int(row["problem_index"]), int(row["sample_index"]), int(row["seed"]))
        if key in keyed:
            raise RuntimeError(f"duplicate generation key: {key}")
        keyed[key] = row
    return keyed


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = read_report(args.baseline)
    candidate = read_report(args.candidate)
    for field in ("protocol", "benchmark", "data_sha256", "sampling"):
        if baseline.get(field) != candidate.get(field):
            raise RuntimeError(f"unaligned reports: {field} differs")

    baseline_rows = keyed_generations(baseline)
    candidate_rows = keyed_generations(candidate)
    if baseline_rows.keys() != candidate_rows.keys():
        raise RuntimeError("unaligned reports: generation keys differ")

    problem_deltas: dict[int, list[float]] = {}
    for key in sorted(baseline_rows):
        problem_index = key[0]
        problem_deltas.setdefault(problem_index, []).append(
            float(candidate_rows[key]["correct"]) - float(baseline_rows[key]["correct"])
        )
    deltas = [statistics.fmean(values) for values in problem_deltas.values()]
    accuracy_delta = statistics.fmean(deltas)
    bootstrap_ci = paired_bootstrap_ci(deltas, args.bootstrap_seed, args.bootstrap_replicates)
    metrics = candidate["metrics"]
    checks = {
        "parser_error_rate": float(metrics["parser_error_rate"]) <= args.max_parser_error_rate,
        "answer_extracted_rate": float(metrics["answer_extracted_rate"])
        >= args.min_answer_extracted_rate,
        "format_rate": float(metrics["format_rate"]) >= args.min_format_rate,
        "truncation_rate": float(metrics["truncation_rate"]) <= args.max_truncation_rate,
        "accuracy_point_estimate_not_down": accuracy_delta >= 0.0,
        "paired_bootstrap_lower_bound": bootstrap_ci[0] > -args.max_bootstrap_regression,
    }
    report = {
        "protocol": "math-sft-gate-v1",
        "baseline_report": str(args.baseline.resolve()),
        "candidate_report": str(args.candidate.resolve()),
        "alignment": {
            "data_sha256": candidate["data_sha256"],
            "sampling": candidate["sampling"],
            "paired_questions": len(deltas),
            "paired_generations": len(candidate_rows),
        },
        "thresholds": {
            "min_answer_extracted_rate": args.min_answer_extracted_rate,
            "min_format_rate": args.min_format_rate,
            "max_parser_error_rate": args.max_parser_error_rate,
            "max_truncation_rate": args.max_truncation_rate,
            "min_accuracy_delta": 0.0,
            "min_paired_bootstrap_lower_bound": -args.max_bootstrap_regression,
        },
        "baseline_metrics": baseline["metrics"],
        "candidate_metrics": candidate["metrics"],
        "accuracy_delta": accuracy_delta,
        "paired_bootstrap_95_ci": bootstrap_ci,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
