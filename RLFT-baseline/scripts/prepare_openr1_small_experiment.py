#!/usr/bin/env python3
"""Prepare disjoint full-trace SFT, GRPO-audit, and held-out OpenR1 splits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

from math_verify_reward import ANSWER_INSTRUCTION, compute_score, ensure_answer_instruction


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = (
    ROOT
    / "data/raw/hf-cache/open-r1___open_r1-math-220k/default/0.0.0"
    / "e4e141ec9dea9f8326f4d347be56105859b2bd68"
)
_STRICT_ANSWER_RE = re.compile(r"^\s*Answer\s*:\s*.+?\s*$", re.IGNORECASE)
COMPACT_REASONING_INSTRUCTION = (
    "Keep the reasoning concise, do not repeat completed steps, and finish within the token budget."
)
STRICT_COMPACT_REASONING_INSTRUCTION = (
    "Use one direct line of reasoning, without alternative attempts or repeated checks. "
    "Keep the response under 512 tokens and stop immediately after the required Answer line."
)
_DELIBERATION_RE = re.compile(
    r"\b(?:wait|maybe|perhaps|reconsider|re-evaluate|rethink|try again|"
    r"another (?:way|approach)|alternatively|upon reflection|scratch that)\b|"
    r"\b(?:but|however),? actually\b",
    re.IGNORECASE,
)
_FINAL_ANSWER_PHRASE_RE = re.compile(r"\bfinal answer\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\s]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-train",
        type=Path,
        default=ROOT / "data/processed/openr1_grpo_final_v1/train.parquet",
    )
    parser.add_argument(
        "--clean-validation",
        type=Path,
        default=ROOT / "data/processed/openr1_grpo_final_v1/validation.parquet",
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--model", type=Path, default=Path("/mnt/qjh007/models/Qwen3-30B-A3B-Instruct-2507"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/openr1_small_v1")
    parser.add_argument("--sft-train-size", type=int, default=128)
    parser.add_argument("--sft-validation-size", type=int, default=16)
    parser.add_argument("--grpo-audit-size", type=int, default=256)
    parser.add_argument("--max-sft-length", type=int, default=4096)
    parser.add_argument("--max-teacher-response-tokens", type=int)
    parser.add_argument(
        "--compact-teacher",
        action="store_true",
        help="Deduplicate teacher paragraphs, remove presentation-only markup, and use a concise prompt contract.",
    )
    parser.add_argument(
        "--strict-compact-v3",
        action="store_true",
        help=(
            "Apply fail-closed short-trace quality filters, a strict termination prompt, "
            "and prompt-level near-duplicate isolation."
        ),
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.9,
        help="Token 3-gram Jaccard threshold used by --strict-compact-v3.",
    )
    parser.add_argument(
        "--process-audit",
        type=Path,
        help="JSONL process audit; only exact PASS rows remain eligible for SFT.",
    )
    parser.add_argument(
        "--process-balanced-v4",
        action="store_true",
        help=(
            "Require --process-audit and balance SFT selection across source, problem type, "
            "question type, and generation-consensus difficulty."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def user_text(prompt: Any) -> str:
    if not isinstance(prompt, list):
        return ""
    return "\n".join(str(item.get("content", "")) for item in prompt if item.get("role") == "user").strip()


def stable_key(row: dict[str, Any], seed: int) -> str:
    extra = row.get("extra_info") or {}
    value = str(extra.get("sample_hash") or "")
    payload = f"{seed}\0{value}\0{user_text(row.get('prompt'))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def round_robin(
    rows: Iterable[dict[str, Any]],
    size: int,
    seed: int,
    bucket_fields: Sequence[str] = ("raw_source", "problem_type", "question_type"),
    primary_balance_field: str | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: stable_key(item, seed)):
        extra = row.get("extra_info") or {}
        bucket = tuple(str(extra.get(field, "unknown")) for field in bucket_fields)
        buckets[bucket].append(row)
    selected: list[dict[str, Any]] = []
    if primary_balance_field is None:
        active = sorted(buckets)
    else:
        if primary_balance_field not in bucket_fields:
            raise ValueError("primary balance field must be one of the bucket fields")
        primary_index = tuple(bucket_fields).index(primary_balance_field)
        grouped_buckets: dict[str, deque[tuple[str, ...]]] = defaultdict(deque)
        for bucket in buckets:
            grouped_buckets[bucket[primary_index]].append(bucket)
        for primary, group in grouped_buckets.items():
            grouped_buckets[primary] = deque(
                sorted(
                    group,
                    key=lambda value: hashlib.sha256(
                        f"{seed}\0{value}".encode("utf-8")
                    ).hexdigest(),
                )
            )
        active = []
        primary_groups = sorted(grouped_buckets)
        while primary_groups:
            next_primary_groups = []
            for primary in primary_groups:
                active.append(grouped_buckets[primary].popleft())
                if grouped_buckets[primary]:
                    next_primary_groups.append(primary)
            primary_groups = next_primary_groups
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


def ensure_compact_instruction(
    prompt: list[dict[str, Any]], reasoning_instruction: str = COMPACT_REASONING_INSTRUCTION
) -> list[dict[str, str]]:
    messages = ensure_answer_instruction(prompt)
    if any(reasoning_instruction in item["content"] for item in messages):
        return messages
    user_indexes = [index for index, item in enumerate(messages) if item["role"] == "user"]
    index = user_indexes[-1]
    content = messages[index]["content"]
    if ANSWER_INSTRUCTION in content:
        messages[index]["content"] = content.replace(
            ANSWER_INSTRUCTION,
            f"{ANSWER_INSTRUCTION}\n{reasoning_instruction}",
            1,
        )
    else:
        messages[index]["content"] = f"{reasoning_instruction}\n\n{content}"
    return messages


def ensure_strict_compact_instruction(prompt: list[dict[str, Any]]) -> list[dict[str, str]]:
    return ensure_compact_instruction(prompt, STRICT_COMPACT_REASONING_INSTRUCTION)


def problem_text(prompt: Any) -> str:
    value = user_text(prompt)
    for instruction in (
        ANSWER_INSTRUCTION,
        COMPACT_REASONING_INSTRUCTION,
        STRICT_COMPACT_REASONING_INSTRUCTION,
    ):
        value = value.replace(instruction, "")
    return normalized_text(value)


def prompt_shingles(prompt: Any, width: int = 3) -> frozenset[tuple[str, ...]]:
    tokens = [token.casefold() for token in _TOKEN_RE.findall(problem_text(prompt))]
    if len(tokens) < width:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1))


def row_prompt(row: dict[str, Any]) -> Any:
    if row.get("prompt") is not None:
        return row["prompt"]
    messages = row.get("messages") or []
    return [message for message in messages if message.get("role") != "assistant"]


def jaccard_similarity(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left or not right:
        return float(left == right)
    return len(left & right) / len(left | right)


def is_near_duplicate(
    profile: frozenset[Any], references: Iterable[frozenset[Any]], threshold: float
) -> bool:
    return any(jaccard_similarity(profile, reference) >= threshold for reference in references)


def near_duplicate_pair_count(
    left_rows: Iterable[dict[str, Any]],
    right_rows: Iterable[dict[str, Any]],
    threshold: float,
) -> int:
    left_profiles = [prompt_shingles(row_prompt(row)) for row in left_rows]
    right_profiles = [prompt_shingles(row_prompt(row)) for row in right_rows]
    return sum(
        jaccard_similarity(left, right) >= threshold
        for left in left_profiles
        for right in right_profiles
    )


def round_robin_unique(
    rows: Iterable[dict[str, Any]],
    size: int,
    seed: int,
    threshold: float,
    forbidden_profiles: Iterable[frozenset[Any]] = (),
    bucket_fields: Sequence[str] = ("raw_source", "problem_type", "question_type"),
    primary_balance_field: str | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    row_list = list(rows)
    ordered = round_robin(
        row_list,
        len(row_list),
        seed,
        bucket_fields,
        primary_balance_field,
    )
    profiles = list(forbidden_profiles)
    selected: list[dict[str, Any]] = []
    skipped: set[str] = set()
    for row in ordered:
        profile = prompt_shingles(row_prompt(row))
        sample_hash = str((row.get("extra_info") or {}).get("sample_hash") or "")
        if is_near_duplicate(profile, profiles, threshold):
            skipped.add(sample_hash)
            continue
        selected.append(row)
        profiles.append(profile)
        if len(selected) == size:
            break
    return selected, skipped


def teacher_quality_issue(target: str) -> str | None:
    """Return the first fail-closed compact-v3 quality issue."""
    answer_lines = re.findall(r"(?im)^\s*Answer\s*:", target)
    if len(answer_lines) != 1:
        return "teacher_multiple_answer_lines"
    body = "\n".join(target.rstrip().splitlines()[:-1]).strip()
    if _DELIBERATION_RE.search(body):
        return "teacher_deliberation_marker"
    if _FINAL_ANSWER_PHRASE_RE.search(body):
        return "teacher_embedded_final_answer"
    if body.count("$") % 2:
        return "teacher_unbalanced_math_delimiter"

    units = [
        normalized_text(unit)
        for unit in re.split(r"(?<=[.!?])\s+|\n+", body)
        if len(_TOKEN_RE.findall(unit)) >= 8
    ]
    profiles = [prompt_shingles([{"role": "user", "content": unit}]) for unit in units]
    for index, profile in enumerate(profiles):
        if is_near_duplicate(profile, profiles[:index], 0.92):
            return "teacher_near_duplicate_reasoning_unit"
    return None


def compact_teacher_solution(solution: str) -> tuple[str, int]:
    """Apply conservative presentation cleanup without truncating reasoning."""
    value = str(solution).replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"</?think>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", value)
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", value) if paragraph.strip()]
    compact: list[str] = []
    seen: set[str] = set()
    removed = 0
    for paragraph in paragraphs:
        if re.fullmatch(r"[-*_]{3,}", paragraph):
            removed += 1
            continue
        normalized = re.sub(r"\s+", " ", paragraph).strip().casefold()
        if normalized in seen:
            removed += 1
            continue
        seen.add(normalized)
        compact.append("\n".join(line.rstrip() for line in paragraph.splitlines()))
    return "\n\n".join(compact).strip(), removed


def add_raw_metadata(
    row: dict[str, Any],
    raw: dict[str, Any],
    compact_prompt: bool = False,
    strict_compact_prompt: bool = False,
) -> dict[str, Any]:
    item = copy.deepcopy(row)
    extra = item.setdefault("extra_info", {})
    extra.update(
        {
            "uuid": raw.get("uuid"),
            "raw_source": raw.get("source"),
            "problem_type": raw.get("problem_type"),
            "question_type": raw.get("question_type"),
        }
    )
    if strict_compact_prompt:
        item["prompt"] = ensure_strict_compact_instruction(item["prompt"])
    elif compact_prompt:
        item["prompt"] = ensure_compact_instruction(item["prompt"])
    else:
        item["prompt"] = ensure_answer_instruction(item["prompt"])
    return item


def normalized_target(solution: str, ground_truth: str) -> str:
    lines = solution.rstrip().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and _STRICT_ANSWER_RE.match(lines[-1]):
        lines.pop()
    body = "\n".join(lines).rstrip()
    return f"{body}\n\nAnswer: {ground_truth}" if body else f"Answer: {ground_truth}"


def main() -> int:
    args = parse_args()
    required_sizes = args.sft_train_size + args.sft_validation_size + args.grpo_audit_size
    if min(args.sft_train_size, args.sft_validation_size, args.grpo_audit_size) < 1:
        raise ValueError("all split sizes must be positive")
    if args.max_teacher_response_tokens is not None and args.max_teacher_response_tokens < 1:
        raise ValueError("max teacher response tokens must be positive")
    if args.strict_compact_v3 and not args.compact_teacher:
        raise ValueError("--strict-compact-v3 requires --compact-teacher")
    if args.strict_compact_v3 and (
        args.max_teacher_response_tokens is None or args.max_teacher_response_tokens > 512
    ):
        raise ValueError("--strict-compact-v3 requires --max-teacher-response-tokens <= 512")
    if args.process_balanced_v4 and not args.strict_compact_v3:
        raise ValueError("--process-balanced-v4 requires --strict-compact-v3")
    if args.process_balanced_v4 and args.process_audit is None:
        raise ValueError("--process-balanced-v4 requires --process-audit")
    if not 0 < args.near_duplicate_threshold <= 1:
        raise ValueError("near duplicate threshold must be in (0, 1]")
    required_paths = [args.clean_train, args.clean_validation, args.model / "config.json"]
    if args.process_audit is not None:
        required_paths.append(args.process_audit)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    arrow_paths = sorted(args.raw_cache_dir.glob("*.arrow"))
    if not arrow_paths:
        raise FileNotFoundError(f"no Arrow shards in {args.raw_cache_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")

    from datasets import Dataset, concatenate_datasets
    from transformers import AutoTokenizer

    raw_dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in arrow_paths])
    clean_train = Dataset.from_parquet(str(args.clean_train))
    clean_validation = Dataset.from_parquet(str(args.clean_validation))
    if len(clean_train) < required_sizes:
        raise RuntimeError(f"need {required_sizes} clean train rows, found {len(clean_train)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)

    process_audit: dict[str, dict[str, Any]] = {}
    process_audit_report: dict[str, Any] = {}
    if args.process_audit is not None:
        for line_number, line in enumerate(args.process_audit.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_hash = str(record.get("sample_hash") or "")
            if not sample_hash:
                raise ValueError(f"process audit row {line_number} has no sample_hash")
            if sample_hash in process_audit:
                raise ValueError(f"duplicate process audit sample_hash: {sample_hash}")
            process_audit[sample_hash] = record
        report_path = args.process_audit.with_name("report.json")
        if report_path.is_file():
            process_audit_report = json.loads(report_path.read_text(encoding="utf-8"))
            expected_sha256 = str(
                (process_audit_report.get("output") or {}).get("sha256") or ""
            )
            if expected_sha256 and expected_sha256 != sha256_file(args.process_audit):
                raise ValueError("process audit SHA-256 disagrees with its report")

    heldout = []
    for row in clean_validation:
        item = copy.deepcopy(dict(row))
        if args.strict_compact_v3:
            item["prompt"] = ensure_strict_compact_instruction(item["prompt"])
        elif args.compact_teacher:
            item["prompt"] = ensure_compact_instruction(item["prompt"])
        else:
            item["prompt"] = ensure_answer_instruction(item["prompt"])
        heldout.append(item)
    heldout_profiles = [prompt_shingles(row.get("prompt")) for row in heldout]

    enriched: list[dict[str, Any]] = []
    sft_eligible: list[dict[str, Any]] = []
    candidate_audit: list[dict[str, Any]] = []
    rejected = Counter()
    for row in clean_train:
        item = copy.deepcopy(dict(row))
        extra = item.get("extra_info") or {}
        source_index = int(extra["source_index"])
        audit = {
            "source_index": source_index,
            "sample_hash": str(extra.get("sample_hash") or ""),
            "status": "pending",
            "reason": None,
        }

        def reject_candidate(reason: str) -> None:
            rejected[reason] += 1
            audit["status"] = "rejected"
            audit["reason"] = reason
            candidate_audit.append(audit)

        if source_index < 0 or source_index >= len(raw_dataset):
            reject_candidate("invalid_source_index")
            continue
        raw = dict(raw_dataset[source_index])
        audit.update(
            {
                "uuid": raw.get("uuid"),
                "raw_source": raw.get("source"),
                "problem_type": raw.get("problem_type"),
                "question_type": raw.get("question_type"),
            }
        )
        if normalized_text(raw.get("problem", "")) != normalized_text(user_text(item.get("prompt"))):
            reject_candidate("source_prompt_mismatch")
            continue
        item = add_raw_metadata(
            item,
            raw,
            compact_prompt=args.compact_teacher,
            strict_compact_prompt=args.strict_compact_v3,
        )
        enriched.append(item)

        if args.strict_compact_v3 and is_near_duplicate(
            prompt_shingles(item.get("prompt")), heldout_profiles, args.near_duplicate_threshold
        ):
            reject_candidate("heldout_near_duplicate")
            continue

        truth = str((item.get("reward_model") or {}).get("ground_truth", "")).strip()
        solution = str(raw.get("solution") or "").strip()
        raw_answer = str(raw.get("answer") or "").strip()
        if not solution:
            reject_candidate("empty_solution")
            continue
        if not raw_answer:
            reject_candidate("empty_raw_answer")
            continue
        verifier_extra = {**item["extra_info"], "verifier_prompt": item["prompt"]}
        raw_answer_score = compute_score(
            "openr1", f"Answer: {raw_answer}", truth, extra_info=verifier_extra
        )
        if raw_answer_score.get("parser_error") or not raw_answer_score.get("acc"):
            reject_candidate("raw_answer_disagrees")
            continue
        solution_score = compute_score("openr1", solution, truth, extra_info=verifier_extra)
        if solution_score.get("parser_error") or not solution_score.get("acc"):
            reject_candidate("solution_not_verified")
            continue
        if not any(bool(value) for value in (raw.get("is_reasoning_complete") or [])):
            reject_candidate("no_complete_generation_evidence")
            continue

        original_target = normalized_target(solution, truth)
        removed_paragraphs = 0
        if args.compact_teacher:
            compact_solution, removed_paragraphs = compact_teacher_solution(solution)
            target = normalized_target(compact_solution, truth)
            compact_score = compute_score("openr1", target, truth, extra_info=verifier_extra)
            if compact_score.get("parser_error") or not compact_score.get("acc"):
                reject_candidate("compact_solution_not_verified")
                continue
        else:
            target = original_target
        teacher_response_tokens = len(tokenizer.encode(target, add_special_tokens=False))
        audit.update(
            {
                "teacher_response_tokens": teacher_response_tokens,
                "teacher_original_chars": len(original_target),
                "teacher_compact_chars": len(target),
                "teacher_removed_duplicate_paragraphs": removed_paragraphs,
            }
        )
        if (
            args.max_teacher_response_tokens is not None
            and teacher_response_tokens > args.max_teacher_response_tokens
        ):
            reject_candidate("teacher_response_overlong")
            continue
        if args.strict_compact_v3:
            quality_issue = teacher_quality_issue(target)
            if quality_issue:
                reject_candidate(quality_issue)
                continue
        process_record = None
        if args.process_audit is not None:
            process_record = process_audit.get(str(audit["sample_hash"]))
            if process_record is None:
                reject_candidate("process_audit_missing")
                continue
            verdict = str(process_record.get("process_verdict") or "").upper()
            audit.update(
                {
                    "process_verdict": verdict,
                    "process_reason": process_record.get("process_reason"),
                    "process_full_generation_consensus": bool(
                        process_record.get("full_generation_consensus")
                    ),
                }
            )
            if verdict != "PASS":
                reject_candidate(f"process_audit_{verdict.casefold() or 'invalid'}")
                continue
        messages = [*copy.deepcopy(item["prompt"]), {"role": "assistant", "content": target}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if len(token_ids) > args.max_sft_length:
            audit["sft_sequence_tokens"] = len(token_ids)
            reject_candidate("sft_sequence_overlong")
            continue
        process_metadata = {}
        if process_record is not None:
            process_metadata = {
                "process_audit_verdict": "PASS",
                "process_audit_reason": process_record.get("process_reason"),
                "process_audit_judge_model": process_audit_report.get("judge_model"),
                "process_audit_protocol": process_audit_report.get("protocol"),
                "process_audit_response_sha256": process_record.get("judge_response_sha256"),
                "process_full_generation_consensus": bool(
                    process_record.get("full_generation_consensus")
                ),
                "process_verified_generation_count": int(
                    process_record.get("independently_verified_generation_count") or 0
                ),
                "process_difficulty": (
                    "consensus"
                    if process_record.get("full_generation_consensus")
                    else "hard_verified"
                ),
            }
        sft_eligible.append(
            {
                "messages": messages,
                "data_source": item.get("data_source", "open-r1_OpenR1-Math-220k"),
                "ability": item.get("ability", "math"),
                "extra_info": {
                    **item["extra_info"],
                    "sft_sequence_tokens": len(token_ids),
                    "teacher_response_tokens": teacher_response_tokens,
                    "teacher_original_chars": len(original_target),
                    "teacher_compact_chars": len(target),
                    "teacher_removed_duplicate_paragraphs": removed_paragraphs,
                    "teacher_trace": (
                        "OpenR1.solution compact-complete" if args.compact_teacher else "OpenR1.solution"
                    ),
                    **process_metadata,
                },
            }
        )
        audit.update({"status": "eligible", "sft_sequence_tokens": len(token_ids)})
        candidate_audit.append(audit)

    sft_validation_near_duplicate_skips: set[str] = set()
    if args.strict_compact_v3:
        sft_bucket_fields = (
            "raw_source",
            "problem_type",
            "question_type",
            "process_difficulty",
        ) if args.process_balanced_v4 else (
            "raw_source",
            "problem_type",
            "question_type",
        )
        if args.process_balanced_v4:
            problem_type_counts = Counter(
                str(row["extra_info"].get("problem_type", "unknown"))
                for row in sft_eligible
            )
            validation_candidates = [
                row
                for row in sft_eligible
                if problem_type_counts[
                    str(row["extra_info"].get("problem_type", "unknown"))
                ]
                >= 2
            ]
            selected_validation, sft_validation_near_duplicate_skips = round_robin_unique(
                validation_candidates,
                args.sft_validation_size,
                args.seed + 2,
                args.near_duplicate_threshold,
                heldout_profiles,
                sft_bucket_fields,
                "problem_type",
            )
            validation_hashes = {
                str(row["extra_info"]["sample_hash"]) for row in selected_validation
            }
            train_candidates = [
                row
                for row in sft_eligible
                if str(row["extra_info"]["sample_hash"]) not in validation_hashes
            ]
            selected_train, sft_near_duplicate_skips = round_robin_unique(
                train_candidates,
                args.sft_train_size,
                args.seed,
                args.near_duplicate_threshold,
                [
                    *heldout_profiles,
                    *(prompt_shingles(row_prompt(row)) for row in selected_validation),
                ],
                sft_bucket_fields,
                "problem_type",
            )
            selected_sft = [*selected_train, *selected_validation]
        else:
            selected_sft, sft_near_duplicate_skips = round_robin_unique(
                sft_eligible,
                args.sft_train_size + args.sft_validation_size,
                args.seed,
                args.near_duplicate_threshold,
                heldout_profiles,
                sft_bucket_fields,
            )
    else:
        selected_sft = round_robin(
            sft_eligible, args.sft_train_size + args.sft_validation_size, args.seed
        )
        sft_near_duplicate_skips = set()
    if len(selected_sft) < args.sft_train_size + args.sft_validation_size:
        raise RuntimeError(f"only {len(selected_sft)} SFT rows passed strict gates")
    sft_train = selected_sft[: args.sft_train_size]
    sft_validation = selected_sft[args.sft_train_size :]
    sft_hashes = {str(row["extra_info"]["sample_hash"]) for row in selected_sft}
    sft_train_hashes = {str(row["extra_info"]["sample_hash"]) for row in sft_train}
    sft_validation_hashes = {
        str(row["extra_info"]["sample_hash"]) for row in sft_validation
    }

    audit_candidates = [
        row for row in enriched if str((row.get("extra_info") or {}).get("sample_hash")) not in sft_hashes
    ]
    if args.strict_compact_v3:
        forbidden_profiles = [
            *heldout_profiles,
            *(prompt_shingles(row_prompt(row)) for row in selected_sft),
        ]
        grpo_audit, audit_near_duplicate_skips = round_robin_unique(
            audit_candidates,
            args.grpo_audit_size,
            args.seed + 1,
            args.near_duplicate_threshold,
            forbidden_profiles,
        )
    else:
        grpo_audit = round_robin(audit_candidates, args.grpo_audit_size, args.seed + 1)
        audit_near_duplicate_skips = set()
    if len(grpo_audit) < args.grpo_audit_size:
        raise RuntimeError(f"only {len(grpo_audit)} disjoint GRPO audit rows are available")
    grpo_audit_hashes = {
        str((row.get("extra_info") or {}).get("sample_hash") or "") for row in grpo_audit
    }

    for record in candidate_audit:
        sample_hash = str(record.get("sample_hash") or "")
        if sample_hash in sft_train_hashes:
            record["status"] = "selected_sft_train"
        elif sample_hash in sft_validation_hashes:
            record["status"] = "selected_sft_validation"
        elif record["status"] == "eligible" and sample_hash in sft_near_duplicate_skips:
            record["status"] = "eligible_not_selected_near_duplicate"
        elif record["status"] == "eligible":
            record["status"] = "eligible_not_selected"
        record["selected_grpo_audit"] = sample_hash in grpo_audit_hashes

    train_prompt_hashes = {normalized_text(user_text(row["prompt"])) for row in enriched}
    heldout_hashes = {normalized_text(user_text(row["prompt"])) for row in heldout}
    if train_prompt_hashes & heldout_hashes:
        raise RuntimeError("clean train and held-out validation contain prompt overlap")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "sft_train": args.output_dir / "sft_train.parquet",
        "sft_validation": args.output_dir / "sft_validation.parquet",
        "grpo_audit_stage1": args.output_dir / "grpo_audit_stage1.parquet",
        "heldout_validation": args.output_dir / "heldout_validation.parquet",
    }
    candidate_audit_path = args.output_dir / "candidate_audit.jsonl"
    Dataset.from_list(sft_train).to_parquet(str(outputs["sft_train"]))
    Dataset.from_list(sft_validation).to_parquet(str(outputs["sft_validation"]))
    Dataset.from_list(grpo_audit).to_parquet(str(outputs["grpo_audit_stage1"]))
    Dataset.from_list(heldout).to_parquet(str(outputs["heldout_validation"]))
    candidate_audit_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in candidate_audit),
        encoding="utf-8",
    )

    selected_hash_sets = {
        "sft_train": {row["extra_info"]["sample_hash"] for row in sft_train},
        "sft_validation": {row["extra_info"]["sample_hash"] for row in sft_validation},
        "grpo_audit_stage1": {row["extra_info"]["sample_hash"] for row in grpo_audit},
    }
    names = list(selected_hash_sets)
    overlaps = {
        f"{left}:{right}": len(selected_hash_sets[left] & selected_hash_sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"output split overlap: {overlaps}")

    split_rows = {
        "sft_train": sft_train,
        "sft_validation": sft_validation,
        "grpo_audit_stage1": grpo_audit,
        "heldout_validation": heldout,
    }
    near_duplicate_overlaps = {}
    if args.strict_compact_v3:
        split_names = list(split_rows)
        near_duplicate_overlaps = {
            f"{left}:{right}": near_duplicate_pair_count(
                split_rows[left], split_rows[right], args.near_duplicate_threshold
            )
            for index, left in enumerate(split_names)
            for right in split_names[index + 1 :]
        }
        if any(near_duplicate_overlaps.values()):
            raise RuntimeError(f"near-duplicate output split overlap: {near_duplicate_overlaps}")

    manifest = {
        "protocol": (
            "openr1-small-process-audited-coverage-sft-grpo-v4"
            if args.process_balanced_v4
            else "openr1-small-strict-compact-full-trace-sft-grpo-v3"
            if args.strict_compact_v3
            else (
                "openr1-small-compact-full-trace-sft-grpo-v2"
                if args.compact_teacher
                else "openr1-small-full-trace-sft-grpo-v1"
            )
        ),
        "seed": args.seed,
        "policy": {
            "human_review": "none; every ambiguous or failed row is discarded",
            "answer_contract": "last non-empty line is Answer: <ground_truth>",
            "teacher_trace": "complete OpenR1 solution, accepted only after repaired-verifier agreement",
            "teacher_compaction": (
                "remove duplicate paragraphs, separators, think wrappers, and hyperlink targets; never truncate"
                if args.compact_teacher
                else "disabled"
            ),
            "concise_prompt_contract": (
                STRICT_COMPACT_REASONING_INSTRUCTION
                if args.strict_compact_v3
                else (COMPACT_REASONING_INSTRUCTION if args.compact_teacher else None)
            ),
            "strict_compact_v3": args.strict_compact_v3,
            "process_balanced_v4": args.process_balanced_v4,
            "process_audit_policy": (
                "accept exact PASS only; discard FAIL, UNCERTAIN, PARSE_ERROR, invalid, and missing"
                if args.process_audit is not None
                else "disabled"
            ),
            "ambiguity_policy": "discard; no human review candidates are retained",
            "near_duplicate_policy": (
                f"token 3-gram Jaccard < {args.near_duplicate_threshold} across all output splits"
                if args.strict_compact_v3
                else "disabled"
            ),
            "split_method": (
                "problem-type-first deterministic source/problem/question-type/process-difficulty round robin; singleton problem types reserved for SFT train"
                if args.process_balanced_v4
                else "deterministic source/problem/question-type round robin"
            ),
        },
        "inputs": {
            "clean_train": {"path": str(args.clean_train.resolve()), "rows": len(clean_train), "sha256": sha256_file(args.clean_train)},
            "clean_validation": {"path": str(args.clean_validation.resolve()), "rows": len(clean_validation), "sha256": sha256_file(args.clean_validation)},
            "raw_cache_dir": str(args.raw_cache_dir.resolve()),
            "raw_rows": len(raw_dataset),
            "model": str(args.model.resolve()),
            "verifier_sha256": sha256_file(ROOT / "scripts/math_verify_reward.py"),
        },
        "eligibility": {
            "enriched_train_rows": len(enriched),
            "strict_sft_eligible_rows": len(sft_eligible),
            "rejected_by_reason": dict(sorted(rejected.items())),
            "max_sft_length": args.max_sft_length,
            "max_teacher_response_tokens": args.max_teacher_response_tokens,
            "sft_selection_near_duplicate_skips": len(sft_near_duplicate_skips),
            "sft_validation_near_duplicate_skips": len(
                sft_validation_near_duplicate_skips
            ),
            "grpo_audit_near_duplicate_skips": len(audit_near_duplicate_skips),
        },
        "outputs": {
            name: {"path": str(path.resolve()), "rows": len(Dataset.from_parquet(str(path))), "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
        "overlaps": overlaps,
        "near_duplicate_overlaps": near_duplicate_overlaps,
    }
    if args.process_audit is not None:
        manifest["inputs"]["process_audit"] = {
            "path": str(args.process_audit.resolve()),
            "rows": len(process_audit),
            "sha256": sha256_file(args.process_audit),
        }
        report_path = args.process_audit.with_name("report.json")
        if report_path.is_file():
            manifest["inputs"]["process_audit_report"] = {
                "path": str(report_path.resolve()),
                "sha256": sha256_file(report_path),
                "judge_model": process_audit_report.get("judge_model"),
                "protocol": process_audit_report.get("protocol"),
            }
    manifest["outputs"]["candidate_audit"] = {
        "path": str(candidate_audit_path.resolve()),
        "rows": len(candidate_audit),
        "sha256": sha256_file(candidate_audit_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
