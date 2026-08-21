#!/usr/bin/env python3
"""Build the deterministic DAPO 2048/256 B0 split and benchmark files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path("/home/wubowen/Ktransformers-development/RLFT-baseline")
MODEL = Path("/mnt/qjh007/models/Qwen3-30B-A3B")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dapo-id", default="BytedTsinghua-SIA/DAPO-Math-17k")
    parser.add_argument("--dapo-revision", required=True)
    parser.add_argument("--dapo-split", default="train")
    parser.add_argument("--math500-id", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--math500-revision", required=True)
    parser.add_argument("--math500-split", default="test")
    parser.add_argument("--aime2024-id", default="BytedTsinghua-SIA/AIME-2024")
    parser.add_argument("--aime2024-revision", required=True)
    parser.add_argument("--aime2024-split", default="train")
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "data_manifest.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "raw" / "hf-cache")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def prompt_text(value: Any) -> str:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        user_parts = [normalize_text(item.get("content", "")) for item in value if item.get("role") == "user"]
        if user_parts:
            return "\n".join(user_parts)
        return "\n".join(normalize_text(item.get("content", "")) for item in value)
    if isinstance(value, dict):
        return normalize_text(value.get("content", value))
    return normalize_text(value)


def record_prompt(record: dict[str, Any]) -> str:
    for key in ("prompt", "problem", "question", "query"):
        if key in record and record[key] not in (None, ""):
            return prompt_text(record[key])
    raise KeyError(f"no prompt-like field; keys={sorted(record)}")


def record_ground_truth(record: dict[str, Any]) -> str:
    reward = record.get("reward_model")
    if isinstance(reward, dict) and reward.get("ground_truth") not in (None, ""):
        return normalize_text(reward["ground_truth"])
    for key in ("answer", "ground_truth", "solution", "target"):
        if key in record and record[key] not in (None, ""):
            return normalize_text(record[key])
    raise KeyError(f"no ground-truth field; keys={sorted(record)}")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    def percentile(q: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def require_revision(value: str, label: str) -> None:
    if not REVISION_RE.fullmatch(value):
        raise ValueError(f"{label} must be an immutable 40-64 digit commit hash, got {value!r}")


def resolve_revision(api: Any, dataset_id: str, revision: str) -> str:
    resolved = api.dataset_info(dataset_id, revision=revision).sha
    if resolved != revision:
        raise RuntimeError(f"revision mismatch for {dataset_id}: requested={revision}, resolved={resolved}")
    return resolved


def benchmark_records(dataset: Any, source: str) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    prompts: set[str] = set()
    for index, raw in enumerate(dataset):
        text = record_prompt(raw)
        truth = record_ground_truth(raw)
        if text in prompts:
            continue
        prompts.add(text)
        rows.append(
            {
                "data_source": source,
                "prompt": [{"role": "user", "content": text}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": truth},
                "extra_info": {"source_index": index, "split": source},
            }
        )
    return rows, prompts


def main() -> int:
    args = parse_args()
    for value, label in (
        (args.dapo_revision, "DAPO revision"),
        (args.math500_revision, "MATH-500 revision"),
        (args.aime2024_revision, "AIME-2024 revision"),
    ):
        require_revision(value, label)
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)

    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    api = HfApi()
    revisions = {
        "dapo": resolve_revision(api, args.dapo_id, args.dapo_revision),
        "math500": resolve_revision(api, args.math500_id, args.math500_revision),
        "aime2024": resolve_revision(api, args.aime2024_id, args.aime2024_revision),
    }
    load_kwargs = {"cache_dir": str(args.cache_dir)}
    dapo = load_dataset(args.dapo_id, split=args.dapo_split, revision=revisions["dapo"], **load_kwargs)
    math500 = load_dataset(
        args.math500_id, split=args.math500_split, revision=revisions["math500"], **load_kwargs
    )
    aime2024 = load_dataset(
        args.aime2024_id, split=args.aime2024_split, revision=revisions["aime2024"], **load_kwargs
    )
    math_rows, math_prompts = benchmark_records(math500, "HuggingFaceH4/MATH-500")
    aime_rows, aime_prompts = benchmark_records(aime2024, "aime2024")
    benchmark_prompts = math_prompts | aime_prompts

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    # The published DAPO parquet stores the same 17,917 source IDs in 100
    # repeated blocks (1,791,700 physical rows). Collapse that storage-level
    # expansion first, then perform the protocol's normalized-prompt dedupe.
    source_infos = dapo["extra_info"]
    unique_source_ids: set[str] = set()
    unique_positions: list[int] = []
    for position, info in enumerate(source_infos):
        source_id = str(info.get("index", position))
        if source_id not in unique_source_ids:
            unique_source_ids.add(source_id)
            unique_positions.append(position)
    unique_dapo = dapo.select(unique_positions)
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for unique_index, raw in enumerate(unique_dapo):
        index = unique_positions[unique_index]
        text = record_prompt(raw)
        truth = record_ground_truth(raw)
        sample_hash = sha256_text("math_dapo\0" + text + "\0" + truth)
        reason = None
        if text in seen_prompts:
            reason = "duplicate_prompt"
        elif text in benchmark_prompts:
            reason = "benchmark_overlap"
        else:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
            )
            if len(rendered) > args.max_prompt_length:
                reason = "overlong_prompt"
        seen_prompts.add(text)
        if reason:
            exclusions.append({"source_index": index, "sample_hash": sample_hash, "reason": reason})
            continue
        candidates.append(
            {
                "data_source": "math_dapo",
                "prompt": [{"role": "user", "content": text}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": truth},
                "extra_info": {"source_index": index, "sample_hash": sample_hash, "split": "unassigned"},
                "_prompt_tokens": len(rendered),
            }
        )

    candidates.sort(key=lambda row: row["extra_info"]["sample_hash"])
    required = args.train_size + args.validation_size
    if len(candidates) < required:
        raise RuntimeError(f"only {len(candidates)} eligible samples, need {required}")
    train_rows = candidates[: args.train_size]
    validation_rows = candidates[args.train_size : required]
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            row["extra_info"]["split"] = split

    prompt_lengths = [row.pop("_prompt_tokens") for row in train_rows + validation_rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": args.output_dir / "train.parquet",
        "validation": args.output_dir / "validation.parquet",
        "math500": args.output_dir / "math500.parquet",
        "aime2024": args.output_dir / "aime2024.parquet",
    }
    Dataset.from_list(train_rows).to_parquet(str(outputs["train"]))
    Dataset.from_list(validation_rows).to_parquet(str(outputs["validation"]))
    Dataset.from_list(math_rows).to_parquet(str(outputs["math500"]))
    Dataset.from_list(aime_rows).to_parquet(str(outputs["aime2024"]))

    manifest = {
        "protocol": "qwen3-30b-a3b-verl-grpo-b0",
        "datasets": {
            "dapo": {"id": args.dapo_id, "revision": revisions["dapo"], "split": args.dapo_split},
            "math500": {"id": args.math500_id, "revision": revisions["math500"], "split": args.math500_split},
            "aime2024": {
                "id": args.aime2024_id,
                "revision": revisions["aime2024"],
                "split": args.aime2024_split,
            },
        },
        "source_rows": {"dapo": len(dapo), "math500": len(math500), "aime2024": len(aime2024)},
        "source_unique_rows": {
            "dapo_ids": len(unique_dapo),
            "math500_prompts": len(math_rows),
            "aime2024_prompts": len(aime_rows),
        },
        "source_repetition": {
            "dapo_physical_rows_minus_unique_ids": len(dapo) - len(unique_dapo),
            "aime2024_physical_rows_minus_unique_prompts": len(aime2024) - len(aime_rows),
        },
        "eligible_rows": len(candidates),
        "selected": {
            "train": [row["extra_info"] for row in train_rows],
            "validation": [row["extra_info"] for row in validation_rows],
        },
        "exclusions": exclusions,
        "exclusion_counts": {
            reason: sum(item["reason"] == reason for item in exclusions)
            for reason in ("duplicate_prompt", "benchmark_overlap", "overlong_prompt")
        },
        "prompt_token_lengths": describe(prompt_lengths),
        "model_path": str(args.model_path.resolve()),
        "max_prompt_length": args.max_prompt_length,
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(Dataset.from_parquet(str(path)))}
            for name, path in outputs.items()
        },
        "selection": "normalize -> exact prompt dedupe -> benchmark exclusion -> overlength exclusion -> sha256 sort",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "outputs": manifest["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
