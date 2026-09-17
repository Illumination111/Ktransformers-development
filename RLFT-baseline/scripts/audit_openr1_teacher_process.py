#!/usr/bin/env python3
"""Audit short OpenR1 teacher solutions with a stronger local judge model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from math_verify_reward import compute_score
from prepare_openr1_small_experiment import compact_teacher_solution, normalized_target, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = (
    ROOT
    / "data/raw/hf-cache/open-r1___open_r1-math-220k/default/0.0.0"
    / "e4e141ec9dea9f8326f4d347be56105859b2bd68"
)
JUDGE_SYSTEM = """You are a strict mathematical proof auditor. Check the supplied candidate solution rather than merely solving the problem yourself. PASS only when every material inference, algebraic transformation, domain restriction, and required case is valid and the expected answer follows from the written reasoning. A correct final answer with a flawed, circular, incomplete, or unjustified argument is not a PASS. Return exactly two lines and keep REASON under 30 words:
VERDICT: PASS|FAIL|UNCERTAIN
REASON: one concise sentence"""
_VERDICT_RE = re.compile(r"(?im)^\s*VERDICT\s*:\s*(PASS|FAIL|UNCERTAIN)\s*$")
_REASON_RE = re.compile(r"(?im)^\s*REASON\s*:\s*(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-audit",
        type=Path,
        default=ROOT / "data/processed/openr1_small_compact_v3/candidate_audit.jsonl",
    )
    parser.add_argument(
        "--clean-train",
        type=Path,
        default=ROOT / "data/processed/openr1_grpo_final_v1/train.parquet",
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--judge-model",
        type=Path,
        default=Path("/mnt/qjh007/models/Qwen3-Next-80B-A3B-Instruct"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def build_judge_messages(problem: str, expected_answer: str, solution: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"PROBLEM:\n{problem.strip()}\n\n"
                f"EXPECTED ANSWER:\n{expected_answer.strip()}\n\n"
                f"CANDIDATE SOLUTION:\n{solution.strip()}"
            ),
        },
    ]


def parse_judgment(text: str, finish_reason: str | None) -> tuple[str, str | None]:
    if finish_reason == "length":
        return "PARSE_ERROR", "judge response reached the token limit"
    verdicts = _VERDICT_RE.findall(text)
    reasons = _REASON_RE.findall(text)
    if len(verdicts) != 1 or len(reasons) != 1:
        return "PARSE_ERROR", "judge response did not contain exactly one verdict and reason"
    return verdicts[0].upper(), reasons[0].strip()


def main() -> int:
    args = parse_args()
    for path in (
        args.candidate_audit,
        args.clean_train,
        args.judge_model / "config.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an existing process-audit output")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor parallel size must be positive")
    if args.max_new_tokens < 32:
        raise ValueError("max new tokens must be at least 32")

    from datasets import Dataset, concatenate_datasets
    from transformers import AutoTokenizer
    import sglang as sgl

    audit_records = [json.loads(line) for line in args.candidate_audit.read_text().splitlines()]
    candidates = [
        record
        for record in audit_records
        if str(record.get("status", "")).startswith(("eligible", "selected_sft"))
    ]
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(candidates):
            raise ValueError(f"invalid limit {args.limit} for {len(candidates)} candidates")
        candidates = candidates[: args.limit]

    arrow_paths = sorted(args.raw_cache_dir.glob("*.arrow"))
    if not arrow_paths:
        raise FileNotFoundError(f"no Arrow shards in {args.raw_cache_dir}")
    raw_dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in arrow_paths])
    clean_train = Dataset.from_parquet(str(args.clean_train))
    clean_by_source_index = {
        int((row.get("extra_info") or {})["source_index"]): dict(row) for row in clean_train
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.judge_model, trust_remote_code=True, local_files_only=True
    )
    prepared = []
    prompt_ids = []
    for record in candidates:
        source_index = int(record["source_index"])
        raw = dict(raw_dataset[source_index])
        clean = clean_by_source_index[source_index]
        truth = str((clean.get("reward_model") or {}).get("ground_truth", "")).strip()
        compact_solution, _ = compact_teacher_solution(str(raw.get("solution") or ""))
        target = normalized_target(compact_solution, truth)
        messages = build_judge_messages(str(raw.get("problem") or ""), truth, target)
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids.append(ids)

        verifier_extra = {
            **(clean.get("extra_info") or {}),
            "verifier_prompt": clean.get("prompt"),
        }
        generations = list(raw.get("generations") or [])
        complete = [bool(value) for value in (raw.get("is_reasoning_complete") or [])]
        independently_verified = 0
        generation_parser_errors = 0
        for generation in generations:
            score = compute_score("openr1", generation, truth, extra_info=verifier_extra)
            independently_verified += int(bool(score.get("acc")))
            generation_parser_errors += int(bool(score.get("parser_error")))
        prepared.append(
            {
                "source_index": source_index,
                "sample_hash": record.get("sample_hash"),
                "uuid": raw.get("uuid"),
                "raw_source": raw.get("source"),
                "problem_type": raw.get("problem_type"),
                "question_type": raw.get("question_type"),
                "teacher_response_tokens": record.get("teacher_response_tokens"),
                "generation_count": len(generations),
                "complete_generation_count": sum(complete),
                "independently_verified_generation_count": independently_verified,
                "generation_parser_error_count": generation_parser_errors,
                "cached_correctness_count": raw.get("correctness_count"),
                "full_generation_consensus": bool(
                    len(generations) >= 2
                    and len(complete) == len(generations)
                    and all(complete)
                    and independently_verified == len(generations)
                ),
            }
        )

    max_prompt_tokens = max(map(len, prompt_ids))
    os.environ.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    engine = sgl.Engine(
        model_path=str(args.judge_model),
        tokenizer_path=str(args.judge_model),
        model_impl="sglang",
        tp_size=args.tensor_parallel_size,
        dp_size=1,
        dtype="bfloat16",
        mem_fraction_static=args.gpu_memory_utilization,
        context_length=max_prompt_tokens + args.max_new_tokens + 1,
        chunked_prefill_size=4096,
        trust_remote_code=True,
        random_seed=42,
        disable_cuda_graph=True,
        enable_torch_compile=False,
    )
    started = time.time()
    outputs = engine.generate(
        input_ids=prompt_ids,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "sampling_seed": 42,
        },
    )
    if isinstance(outputs, dict):
        outputs = [outputs]
    if len(outputs) != len(prepared):
        raise RuntimeError(f"judge returned {len(outputs)} outputs for {len(prepared)} candidates")

    for record, output in zip(prepared, outputs, strict=True):
        finish_reason = output.get("meta_info", {}).get("finish_reason")
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type")
        response = str(output.get("text") or "")
        verdict, reason = parse_judgment(response, finish_reason)
        record.update(
            {
                "process_verdict": verdict,
                "process_reason": reason,
                "judge_finish_reason": finish_reason,
                "judge_response_tokens": len(output.get("output_ids") or []),
                "judge_response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "judge_response": response,
            }
        )

    if hasattr(engine, "shutdown"):
        engine.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in prepared),
        encoding="utf-8",
    )
    verdict_counts = Counter(record["process_verdict"] for record in prepared)
    report = {
        "protocol": "openr1-short-teacher-process-audit-v1",
        "judge_model": str(args.judge_model.resolve()),
        "candidate_audit": {
            "path": str(args.candidate_audit.resolve()),
            "sha256": sha256_file(args.candidate_audit),
        },
        "clean_train_sha256": sha256_file(args.clean_train),
        "verifier_sha256": sha256_file(ROOT / "scripts/math_verify_reward.py"),
        "candidates": len(prepared),
        "diagnostic_limit": args.limit,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "full_generation_consensus": sum(
            bool(record["full_generation_consensus"]) for record in prepared
        ),
        "judge": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "cuda_graph_disabled": True,
            "elapsed_seconds": time.time() - started,
        },
        "output": {
            "path": str(args.output.resolve()),
            "rows": len(prepared),
            "sha256": sha256_file(args.output),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
