#!/usr/bin/env python3
"""Conservatively clean the frozen OpenR1 GRPO split.

The source parquet files are never modified.  High-confidence unusable rows
are excluded automatically; ambiguous rows are quarantined for human review;
only rows with no triggered rule are written to the clean train/validation
files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from math_verify_reward import _split_top_level_commas


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data/processed/openr1_grpo"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/openr1_grpo_clean_v1"

TRANSLATION_RE = re.compile(
    r"translate|translation|untranslated|translated\s+(?:portion|text)|"
    r"翻译|保留源文本|перевед|tradu(?:ce|cción|za)",
    re.IGNORECASE,
)
EMBEDDED_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\([^)]*\)|<img\b|\[(?:image|diagram|figure)\s*(?:omitted|missing)?\]",
    re.IGNORECASE,
)
VISUAL_REFERENCE_RE = re.compile(
    r"\b(?:as\s+)?shown\s*(?:in\s+)?(?:the\s+)?(?:figure|diagram|below|above)?\s*[,.:]|"
    r"\b(?:figure|diagram)\s+(?:above|below)\b|"
    r"\baccording\s+to\s+(?:the\s+)?(?:figure|diagram)\b|"
    r"\bsee\s+(?:the\s+)?(?:figure|diagram)\b|如图|图中",
    re.IGNORECASE,
)
SOLUTION_LEAK_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:solution|answer)\s*:\s*\S|"
    r"the\s+(?:solution|answer)\s+is\s*[:：]",
)
PROOF_TASK_RE = re.compile(
    r"\bprove\b|\bshow\s+that\b|\bjustify\s+your\s+answer\b|证明",
    re.IGNORECASE,
)
LOWER_PART_RE = re.compile(
    r"(?m)(?:^|\n)\s*(?:\([a-e]\)|[a-e][.)]|(?i:part)\s+[a-e1-9])\s+"
)
MALFORMED_GROUND_TRUTH_RES = (
    re.compile(r"\\sqrt\{\^"),
    re.compile(r"\{\^"),
    re.compile(r"\\frac\{\}"),
    re.compile(r"\\frac\{[^{}]*\}\{\}"),
    re.compile(r"\\frac\{\}\{[^{}]*\}"),
)

AUTO_EXCLUDE_REASONS = {
    "benchmark_overlap",
    "duplicate_prompt",
    "embedded_image_unavailable",
    "empty_ground_truth",
    "empty_prompt",
    "invalid_prompt_schema",
    "translation_or_meta_task",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--discard-review",
        action="store_true",
        help="discard every rule-flagged row instead of emitting a human-review queue",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def user_prompt(prompt: Any) -> str:
    if not isinstance(prompt, list):
        return ""
    users = [str(item.get("content", "")) for item in prompt if str(item.get("role")) == "user"]
    return "\n".join(users).strip()


def ground_truth(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model")
    return str(reward_model.get("ground_truth", "")).strip() if isinstance(reward_model, dict) else ""


def malformed_ground_truth(truth: str) -> bool:
    if any(pattern.search(truth) for pattern in MALFORMED_GROUND_TRUTH_RES):
        return True
    # Do not require round-bracket balance: half-open intervals such as
    # ``(-\infty,-1]`` are valid mathematical answers.
    return truth.count("{") != truth.count("}")


def compound_truth_parts(truth: str) -> list[str]:
    return _split_top_level_commas(truth)


def classify_row(
    row: dict[str, Any],
    *,
    seen_prompts: set[str],
    benchmark_prompts: set[str],
) -> tuple[str, list[str]]:
    prompt_value = row.get("prompt")
    prompt = user_prompt(prompt_value)
    truth = ground_truth(row)
    reasons: list[str] = []

    if not isinstance(prompt_value, list) or not all(isinstance(item, dict) for item in prompt_value):
        reasons.append("invalid_prompt_schema")
    if not prompt:
        reasons.append("empty_prompt")
    if not truth:
        reasons.append("empty_ground_truth")

    fingerprint = normalized_text(prompt)
    if fingerprint:
        if fingerprint in seen_prompts:
            reasons.append("duplicate_prompt")
        if fingerprint in benchmark_prompts:
            reasons.append("benchmark_overlap")
        seen_prompts.add(fingerprint)

    if TRANSLATION_RE.search(prompt):
        reasons.append("translation_or_meta_task")
    if EMBEDDED_IMAGE_RE.search(prompt):
        reasons.append("embedded_image_unavailable")

    if malformed_ground_truth(truth):
        reasons.append("malformed_ground_truth")
    if len(truth) > 160:
        reasons.append("unusually_long_ground_truth")
    if SOLUTION_LEAK_RE.search(prompt):
        reasons.append("solution_or_answer_leak")
    if PROOF_TASK_RE.search(prompt):
        reasons.append("proof_quality_not_verifiable")
    if VISUAL_REFERENCE_RE.search(prompt) and not EMBEDDED_IMAGE_RE.search(prompt):
        reasons.append("visual_reference_needs_context_check")

    parts = compound_truth_parts(truth) if truth else []
    if len(parts) > 1:
        reasons.append("compound_ground_truth")
    if LOWER_PART_RE.search(prompt):
        reasons.append("multi_part_prompt")
    if prompt.count("?") >= 2 and len(parts) <= 1:
        reasons.append("multiple_questions_single_label")

    reasons = list(dict.fromkeys(reasons))
    if any(reason in AUTO_EXCLUDE_REASONS for reason in reasons):
        return "exclude", reasons
    if reasons:
        return "review", reasons
    return "keep", []


def review_record(row: dict[str, Any], split: str, row_index: int, reasons: list[str]) -> dict[str, Any]:
    extra = row.get("extra_info") or {}
    return {
        "input_split": split,
        "input_row_index": row_index,
        "source_index": extra.get("source_index"),
        "sample_hash": extra.get("sample_hash"),
        "reasons": reasons,
        "prompt": user_prompt(row.get("prompt")),
        "ground_truth": ground_truth(row),
        "human_decision": "",
        "human_notes": "",
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    from datasets import Dataset

    return [jsonable(row) for row in Dataset.from_parquet(str(path))]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    input_paths = {split: args.input_dir / f"{split}.parquet" for split in ("train", "validation")}
    benchmark_paths = [args.input_dir / "math500.parquet", args.input_dir / "aime2024.parquet"]
    for path in [*input_paths.values(), *benchmark_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.dry_run and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")

    benchmark_prompts: set[str] = set()
    for path in benchmark_paths:
        for row in read_rows(path):
            benchmark_prompts.add(normalized_text(user_prompt(row.get("prompt"))))

    seen_prompts: set[str] = set()
    kept: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    review: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    disposition_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    reason_dispositions: Counter[tuple[str, str]] = Counter()
    input_counts: dict[str, int] = {}

    for split, path in input_paths.items():
        rows = read_rows(path)
        input_counts[split] = len(rows)
        for row_index, row in enumerate(rows):
            disposition, reasons = classify_row(
                row,
                seen_prompts=seen_prompts,
                benchmark_prompts=benchmark_prompts,
            )
            disposition_counts[f"{split}:{disposition}"] += 1
            for reason in reasons:
                reason_counts[reason] += 1
                reason_dispositions[(disposition, reason)] += 1
            if disposition == "keep":
                kept[split].append(row)
            else:
                record = review_record(row, split, row_index, reasons)
                (excluded if disposition == "exclude" else review).append(record)

    report: dict[str, Any] = {
        "protocol": (
            "openr1-grpo-discard-all-flagged-v1"
            if args.discard_review
            else "openr1-grpo-conservative-clean-v1"
        ),
        "policy": {
            "source_files_are_immutable": True,
            "clean_outputs_contain_only_rows_with_no_triggered_rule": True,
            "review_rows_are_quarantined_until_human_decision": not args.discard_review,
            "all_rule_flagged_rows_are_discarded": args.discard_review,
            "auto_exclude_reasons": sorted(AUTO_EXCLUDE_REASONS),
        },
        "inputs": {
            split: {"path": str(path.resolve()), "rows": input_counts[split], "sha256": sha256_file(path)}
            for split, path in input_paths.items()
        },
        "counts": {
            "input_total": sum(input_counts.values()),
            "clean_total": sum(len(rows) for rows in kept.values()),
            "manual_review_total": 0 if args.discard_review else len(review),
            "rule_flagged_discarded_total": len(review) if args.discard_review else 0,
            "auto_excluded_total": len(excluded),
            "discarded_total": len(excluded) + (len(review) if args.discard_review else 0),
            "by_split_and_disposition": (
                {
                    f"{split}:{disposition}": (
                        disposition_counts[f"{split}:keep"]
                        if disposition == "keep"
                        else disposition_counts[f"{split}:review"]
                        + disposition_counts[f"{split}:exclude"]
                    )
                    for split in ("train", "validation")
                    for disposition in ("keep", "discard")
                }
                if args.discard_review
                else dict(sorted(disposition_counts.items()))
            ),
            "by_reason": dict(sorted(reason_counts.items())),
            "by_disposition_and_reason": (
                {
                    f"discard:{reason}": sum(
                        count
                        for (disposition, item_reason), count in reason_dispositions.items()
                        if item_reason == reason
                    )
                    for reason in sorted(reason_counts)
                }
                if args.discard_review
                else {
                    f"{disposition}:{reason}": count
                    for (disposition, reason), count in sorted(reason_dispositions.items())
                }
            ),
        },
        "rules": {
            "automatic_exclusion": "only structural invalidity, exact duplicate/benchmark overlap, explicit translation/meta tasks, or unavailable embedded images",
            "flagged_rows": (
                "discard without human review: malformed/long labels, solution leakage, proof tasks, unresolved visual references, compound labels, or prompt/label cardinality risk"
                if args.discard_review
                else "manual review: malformed/long labels, solution leakage, proof tasks, unresolved visual references, compound labels, or prompt/label cardinality risk"
            ),
        },
    }

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    from datasets import Dataset

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {"train": args.output_dir / "train.parquet", "validation": args.output_dir / "validation.parquet"}
    if args.discard_review:
        output_paths.update(
            {
                "discarded_parquet": args.output_dir / "discarded.parquet",
                "discarded_jsonl": args.output_dir / "discarded.jsonl",
                "discarded_tsv": args.output_dir / "discarded.tsv",
            }
        )
    else:
        output_paths.update(
            {
                "manual_review_parquet": args.output_dir / "manual_review.parquet",
                "manual_review_jsonl": args.output_dir / "manual_review.jsonl",
                "manual_review_tsv": args.output_dir / "manual_review.tsv",
                "auto_excluded_jsonl": args.output_dir / "auto_excluded.jsonl",
            }
        )
    Dataset.from_list(kept["train"]).to_parquet(str(output_paths["train"]))
    Dataset.from_list(kept["validation"]).to_parquet(str(output_paths["validation"]))
    if args.discard_review:
        discarded = [
            {**row, "discard_origin": "rule_flagged", "human_decision": "drop"} for row in review
        ] + [
            {**row, "discard_origin": "automatic_exclusion", "human_decision": "drop"}
            for row in excluded
        ]
        discarded.sort(key=lambda row: (row["input_split"], row["input_row_index"]))
        Dataset.from_list(discarded).to_parquet(str(output_paths["discarded_parquet"]))
        write_jsonl(output_paths["discarded_jsonl"], discarded)
        tabular_rows = discarded
        tsv_path = output_paths["discarded_tsv"]
    else:
        Dataset.from_list(review).to_parquet(str(output_paths["manual_review_parquet"]))
        write_jsonl(output_paths["manual_review_jsonl"], review)
        write_jsonl(output_paths["auto_excluded_jsonl"], excluded)
        tabular_rows = review
        tsv_path = output_paths["manual_review_tsv"]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "input_split",
                "input_row_index",
                "source_index",
                "sample_hash",
                "reasons",
                "discard_origin",
                "ground_truth",
                "prompt_preview",
            ]
        )
        for row in tabular_rows:
            writer.writerow(
                [
                    row["input_split"],
                    row["input_row_index"],
                    row["source_index"],
                    row["sample_hash"],
                    ",".join(row["reasons"]),
                    row.get("discard_origin", ""),
                    row["ground_truth"],
                    re.sub(r"\s+", " ", row["prompt"])[:240],
                ]
            )

    report["outputs"] = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": (
                len(kept[name])
                if name in kept
                else len(review) + len(excluded)
                if name.startswith("discarded")
                else len(review)
                if name.startswith("manual_review")
                else len(excluded)
            ),
        }
        for name, path in output_paths.items()
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
