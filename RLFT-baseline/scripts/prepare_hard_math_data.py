#!/usr/bin/env python3
"""Prepare deduplicated DAPO-Math training data and a frozen MATH-500 gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from math_verify_reward import ensure_answer_instruction


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
DEFAULT_DATASET_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_SOURCE_PREFIX_RE = re.compile(
    r"^Solve the following math problem step by step\.\s*"
    r"The last line of your response should be of the form Answer:.*?\n\n",
    re.IGNORECASE | re.DOTALL,
)
_SOURCE_SUFFIX_RE = re.compile(r"\n\nRemember to put your answer.*\Z", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/raw/hf-cache")
    parser.add_argument("--math500", type=Path, default=ROOT / "data/processed/math500.parquet")
    parser.add_argument("--aime2024", type=Path, default=ROOT / "data/processed/aime2024.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/math_grpo_v2")
    parser.add_argument("--validation-gate-size", type=int, default=64)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def user_content(prompt: list[dict[str, Any]]) -> str:
    users = [str(item["content"]) for item in prompt if str(item.get("role")) == "user"]
    if not users:
        raise ValueError("row has no user prompt")
    return users[-1]


def source_problem(text: str) -> str:
    text = _SOURCE_PREFIX_RE.sub("", str(text).strip(), count=1)
    text = _SOURCE_SUFFIX_RE.sub("", text, count=1)
    return text.strip()


def normalized_problem(text: str) -> str:
    text = unicodedata.normalize("NFKC", source_problem(text))
    return re.sub(r"\s+", " ", text).strip().casefold()


def prompt_fingerprint(prompt: list[dict[str, Any]]) -> str:
    return sha256_bytes(normalized_problem(user_content(prompt)))


def heldout_fingerprints(paths: list[Path]) -> set[str]:
    from datasets import Dataset

    fingerprints: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in Dataset.from_parquet(str(path)):
            fingerprints.add(prompt_fingerprint(jsonable(row["prompt"])))
    return fingerprints


def row_identity(row: dict[str, Any]) -> str:
    extra = row.get("extra_info") or {}
    source_id = extra.get("index")
    if source_id not in (None, ""):
        return str(source_id)
    payload = {
        "prompt": row["prompt"],
        "ground_truth": row["reward_model"]["ground_truth"],
    }
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def canonical_training_row(row: dict[str, Any], dataset_id: str, revision: str) -> dict[str, Any]:
    prompt = jsonable(row.get("prompt"))
    reward_model = jsonable(row.get("reward_model"))
    if not isinstance(prompt, list) or not reward_model or reward_model.get("ground_truth") in (None, ""):
        raise ValueError("each row must contain prompt and reward_model.ground_truth")
    problem = source_problem(user_content(prompt))
    formatted_prompt = ensure_answer_instruction([{"role": "user", "content": problem}])
    source_id = row_identity(row)
    return {
        "data_source": "math_dapo",
        "prompt": formatted_prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(reward_model["ground_truth"])},
        "extra_info": {
            "source_id": source_id,
            "source_dataset": dataset_id,
            "source_revision": revision,
            "problem_sha256": prompt_fingerprint(formatted_prompt),
        },
    }


def deterministic_gate(math500_path: Path, size: int) -> list[dict[str, Any]]:
    from datasets import Dataset

    dataset = Dataset.from_parquet(str(math500_path))
    if len(dataset) != 500:
        raise RuntimeError(f"expected canonical MATH-500 with 500 rows, got {len(dataset)}")
    rows = []
    for source in dataset:
        row = copy.deepcopy(jsonable(source))
        row["prompt"] = ensure_answer_instruction(row["prompt"])
        key = sha256_bytes(
            json.dumps(
                {"prompt": row["prompt"], "ground_truth": row["reward_model"]["ground_truth"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        row.setdefault("extra_info", {})["heldout_gate_key"] = key
        row["extra_info"]["split"] = "heldout_validation_gate"
        rows.append((key, row))
    rows.sort(key=lambda item: item[0])
    return [row for _, row in rows[:size]]


def main() -> int:
    args = parse_args()
    if not REVISION_RE.fullmatch(args.dataset_revision):
        raise ValueError("--dataset-revision must be an immutable commit hash")
    if args.validation_gate_size < 1 or args.validation_gate_size > 500:
        raise ValueError("--validation-gate-size must be between 1 and 500")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")

    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi

    resolved_revision = HfApi().dataset_info(args.dataset_id, revision=args.dataset_revision).sha
    if resolved_revision != args.dataset_revision:
        raise RuntimeError(f"dataset revision mismatch: requested={args.dataset_revision}, resolved={resolved_revision}")

    heldout = heldout_fingerprints([args.math500, args.aime2024])
    source = load_dataset(
        args.dataset_id,
        split=args.dataset_split,
        revision=args.dataset_revision,
        cache_dir=str(args.cache_dir),
    )
    selected: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    duplicate_rows = 0
    overlap_rows = 0
    for raw in source:
        raw = jsonable(raw)
        source_id = row_identity(raw)
        if source_id in seen_ids:
            duplicate_rows += 1
            continue
        seen_ids.add(source_id)
        row = canonical_training_row(raw, args.dataset_id, args.dataset_revision)
        if row["extra_info"]["problem_sha256"] in heldout:
            overlap_rows += 1
            continue
        selected[source_id] = row

    train_rows = sorted(
        selected.values(), key=lambda item: (item["extra_info"]["problem_sha256"], item["extra_info"]["source_id"])
    )
    if len(train_rows) < 10_000:
        raise RuntimeError(f"unexpectedly small deduplicated hard-math dataset: {len(train_rows)} rows")
    gate_rows = deterministic_gate(args.math500, args.validation_gate_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    gate_path = args.output_dir / "validation_gate.parquet"
    Dataset.from_list(train_rows).to_parquet(str(train_path))
    Dataset.from_list(gate_rows).to_parquet(str(gate_path))
    manifest = {
        "protocol": "qwen3-30b-hard-math-grpo-v2",
        "dataset": {
            "id": args.dataset_id,
            "revision": args.dataset_revision,
            "split": args.dataset_split,
            "physical_rows": len(source),
            "deduplicated_rows": len(train_rows),
            "duplicate_rows_discarded": duplicate_rows,
            "heldout_overlap_rows_discarded": overlap_rows,
        },
        "heldout": {
            "math500": {"path": str(args.math500.resolve()), "sha256": sha256_file(args.math500)},
            "aime2024": {"path": str(args.aime2024.resolve()), "sha256": sha256_file(args.aime2024)},
            "validation_gate_rows": len(gate_rows),
        },
        "outputs": {
            "train": {"path": str(train_path.resolve()), "rows": len(train_rows), "sha256": sha256_file(train_path)},
            "validation_gate": {
                "path": str(gate_path.resolve()),
                "rows": len(gate_rows),
                "sha256": sha256_file(gate_path),
            },
        },
        "selection": "deduplicate by extra_info.index; retain first row in the pinned dataset order",
        "answer_contract": 'last non-empty line is "Answer: <answer>"',
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
