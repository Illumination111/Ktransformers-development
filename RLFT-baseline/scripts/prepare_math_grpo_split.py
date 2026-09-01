#!/usr/bin/env python3
"""Create a deterministic local MATH-500 GRPO train/validation split.

MATH-500 is a benchmark, so this split is intended for local diagnostics. The
held-out rows must not be reported as an uncontaminated public benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/math500.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/math_grpo")
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--validation-size", type=int, default=100)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def row_key(row: dict[str, Any]) -> str:
    payload = {
        "prompt": jsonable(row["prompt"]),
        "ground_truth": jsonable(row["reward_model"]["ground_truth"]),
        "source_index": jsonable(row.get("extra_info", {}).get("source_index")),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.train_size < 1 or args.validation_size < 1:
        raise ValueError("train and validation sizes must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")

    from datasets import Dataset

    dataset = Dataset.from_parquet(str(args.input))
    if len(dataset) != 500:
        raise RuntimeError(f"expected canonical MATH-500 input with 500 rows, got {len(dataset)}")

    rows = []
    seen: set[str] = set()
    for row in dataset:
        if not row.get("prompt") or not row.get("reward_model", {}).get("ground_truth"):
            raise ValueError("every row must contain prompt and reward_model.ground_truth")
        item = copy.deepcopy(jsonable(row))
        key = row_key(item)
        if key in seen:
            raise RuntimeError(f"duplicate row identity: {key}")
        seen.add(key)
        item.setdefault("extra_info", {})["math_grpo_split_key"] = key
        rows.append((key, item))

    rows.sort(key=lambda pair: pair[0])
    required = args.train_size + args.validation_size
    if required > len(rows):
        raise ValueError(f"requested {required} rows from {len(rows)}")
    train = []
    validation = []
    for index, (key, item) in enumerate(rows[:required]):
        split = "train" if index < args.train_size else "validation"
        item["extra_info"]["split"] = split
        (train if split == "train" else validation).append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    validation_path = args.output_dir / "validation.parquet"
    Dataset.from_list(train).to_parquet(str(train_path))
    Dataset.from_list(validation).to_parquet(str(validation_path))

    manifest = {
        "protocol": "qwen3-30b-a3b-local-math-grpo-diagnostic",
        "warning": "MATH-500 is a benchmark; this split is not an uncontaminated public benchmark result.",
        "input": {"path": str(args.input.resolve()), "rows": len(dataset), "sha256": sha256_file(args.input)},
        "train": {"path": str(train_path.resolve()), "rows": len(train), "sha256": sha256_file(train_path)},
        "validation": {
            "path": str(validation_path.resolve()),
            "rows": len(validation),
            "sha256": sha256_file(validation_path),
        },
        "split": {
            "method": "sha256(prompt, ground_truth, source_index) ascending",
            "train_keys": [item["extra_info"]["math_grpo_split_key"] for item in train],
            "validation_keys": [item["extra_info"]["math_grpo_split_key"] for item in validation],
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train), "validation": len(validation), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
