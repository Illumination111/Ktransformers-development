#!/usr/bin/env python3
"""Run the fixed B0 MATH-500 or AIME-2024 protocol with native SGLang."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("math500", "aime2024"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, help="Diagnostic only: evaluate the first N questions")
    parser.add_argument("--seed-count", type=int, help="Diagnostic only: use the first N protocol seeds")
    parser.add_argument(
        "--allow-noncanonical-data",
        action="store_true",
        help="Allow a local train/validation subset instead of canonical MATH-500 (diagnostic only)",
    )
    return parser.parse_args()


def bootstrap_ci(values: list[float], seed: int = 42, replicates: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        means.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
    means.sort()
    return [means[int(0.025 * replicates)], means[min(replicates - 1, int(0.975 * replicates))]]


def correctness(result: Any) -> tuple[bool, Any]:
    if isinstance(result, dict):
        if "acc" in result:
            return bool(result["acc"]), result
        return float(result.get("score", 0.0)) > 0, result
    return float(result) > 0, result


def main() -> int:
    args = parse_args()
    default_data = ROOT / "data" / "processed" / f"{args.benchmark}.parquet"
    data_path = args.data or default_data
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)

    from datasets import Dataset
    from transformers import AutoTokenizer
    import sglang as sgl

    from verl.utils.reward_score import default_compute_score

    dataset = Dataset.from_parquet(str(data_path))
    expected = 500 if args.benchmark == "math500" else None
    if expected is not None and len(dataset) != expected and not args.allow_noncanonical_data:
        raise RuntimeError(f"{args.benchmark} must contain {expected} rows, got {len(dataset)}")
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(dataset):
            raise ValueError(f"invalid --limit={args.limit} for {len(dataset)} rows")
        dataset = dataset.select(range(args.limit))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    prompt_ids = [
        tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=True
        )
        for row in dataset
    ]
    max_prompt = max(map(len, prompt_ids))
    if max_prompt > 2048:
        raise RuntimeError(f"benchmark prompt exceeds 2048 tokens: {max_prompt}")

    seeds = list(range(42, 46)) if args.benchmark == "math500" else list(range(42, 58))
    if args.seed_count is not None:
        if args.seed_count < 1 or args.seed_count > len(seeds):
            raise ValueError(f"invalid --seed-count={args.seed_count}")
        seeds = seeds[: args.seed_count]
    engine = sgl.Engine(
        model_path=str(args.model_path),
        tp_size=args.tensor_parallel_size,
        dtype="bfloat16",
        mem_fraction_static=args.gpu_memory_utilization,
        context_length=max_prompt + args.max_new_tokens,
        chunked_prefill_size=8192,
        trust_remote_code=True,
        random_seed=42,
    )
    all_rows: list[dict[str, Any]] = []
    started = time.time()
    for sample_index, seed in enumerate(seeds):
        params = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "sampling_seed": seed,
        }
        outputs = engine.generate(input_ids=prompt_ids, sampling_params=params)
        if isinstance(outputs, dict):
            outputs = [outputs]
        for problem_index, (source, output) in enumerate(zip(dataset, outputs, strict=True)):
            text = output["text"]
            output_ids = list(output["output_ids"])
            finish_reason = output.get("meta_info", {}).get("finish_reason")
            if isinstance(finish_reason, dict):
                finish_reason = finish_reason.get("type")
            verifier = default_compute_score(
                data_source=source["data_source"],
                solution_str=text,
                ground_truth=source["reward_model"]["ground_truth"],
                extra_info=source.get("extra_info"),
            )
            correct, verifier_detail = correctness(verifier)
            all_rows.append(
                {
                    "problem_index": problem_index,
                    "sample_index": sample_index,
                    "seed": seed,
                    "correct": correct,
                    "response": text,
                    "response_token_ids": output_ids,
                    "response_tokens": len(output_ids),
                    "finish_reason": finish_reason,
                    "verifier": verifier_detail,
                }
            )

    per_problem = []
    for problem_index in range(len(dataset)):
        samples = [row for row in all_rows if row["problem_index"] == problem_index]
        per_problem.append(
            {
                "problem_index": problem_index,
                "accuracy": statistics.fmean(float(row["correct"]) for row in samples),
                "successes": sum(row["correct"] for row in samples),
                "samples": len(samples),
            }
        )
    first_sample = [row for row in all_rows if row["sample_index"] == 0]
    question_means = [row["accuracy"] for row in per_problem]
    metrics = {
        "questions": len(dataset),
        "samples_per_question": len(seeds),
        "avg_at_1": statistics.fmean(float(row["correct"]) for row in first_sample),
        f"avg_at_{len(seeds)}": statistics.fmean(question_means),
        "bootstrap_95_ci": bootstrap_ci(question_means),
        "truncation_rate": statistics.fmean(row["finish_reason"] == "length" for row in all_rows),
        "mean_response_tokens": statistics.fmean(row["response_tokens"] for row in all_rows),
        "max_response_tokens": max(row["response_tokens"] for row in all_rows),
        "elapsed_seconds": time.time() - started,
    }
    report = {
        "protocol": "qwen3-30b-a3b-verl-grpo-b0-eval",
        "benchmark": args.benchmark,
        "model_path": str(args.model_path.resolve()),
        "data_path": str(data_path.resolve()),
        "sampling": {
            "enable_thinking": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "seeds": seeds,
        },
        "diagnostic_subset": args.limit is not None or args.seed_count is not None,
        "metrics": metrics,
        "per_problem": per_problem,
        "generations": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(args.output)
    engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
