#!/usr/bin/env python3
"""Convert the hard-math GRPO parquet into veRL multi-turn SFT parquet files.

The current GRPO source contains questions and verified final answers, but no
teacher reasoning traces.  The generated assistant turn therefore teaches the
answer contract only (``Answer: <ground_truth>``); it is not reasoning
distillation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/processed/math_grpo_v2/train.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/math_sft",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.01,
        help="Deterministic fraction held out from the DAPO training source.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def stable_key(row: dict[str, Any], seed: int) -> str:
    extra = row.get("extra_info") or {}
    source_id = extra.get("source_id") or extra.get("index")
    payload = {
        "seed": seed,
        "source_id": source_id,
        "prompt": row.get("prompt"),
        "ground_truth": (row.get("reward_model") or {}).get("ground_truth"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def make_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = jsonable(row.get("prompt"))
    reward_model = jsonable(row.get("reward_model")) or {}
    if not isinstance(prompt, list) or not reward_model.get("ground_truth"):
        raise ValueError("every row needs prompt and reward_model.ground_truth")
    messages = []
    for message in prompt:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
            raise ValueError("SFT source prompt must contain only system/user messages")
        messages.append({"role": str(message["role"]), "content": str(message.get("content", ""))})
    messages.append({"role": "assistant", "content": f"Answer: {reward_model['ground_truth']}"})
    return {
        "messages": messages,
        "data_source": str(row.get("data_source", "math_dapo")),
        "ability": str(row.get("ability", "math")),
        "extra_info": jsonable(row.get("extra_info") or {}),
    }


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")

    from datasets import Dataset

    source = Dataset.from_parquet(str(args.input))
    rows = [make_sft_row(jsonable(row)) for row in source]
    keyed = sorted(
        ((stable_key(row, args.seed), row) for row in rows),
        key=lambda item: item[0],
    )
    validation_size = max(1, round(len(keyed) * args.validation_fraction))
    validation_rows = [row for _, row in keyed[:validation_size]]
    train_rows = [row for _, row in keyed[validation_size:]]
    if not train_rows:
        raise RuntimeError("validation split consumed the entire dataset")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    validation_path = args.output_dir / "validation.parquet"
    Dataset.from_list(train_rows).to_parquet(str(train_path))
    Dataset.from_list(validation_rows).to_parquet(str(validation_path))
    manifest = {
        "protocol": "math-sft-answer-contract-v1",
        "source": {"path": str(args.input.resolve()), "rows": len(source)},
        "outputs": {
            "train": {"path": str(train_path.resolve()), "rows": len(train_rows)},
            "validation": {"path": str(validation_path.resolve()), "rows": len(validation_rows)},
        },
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "assistant_target": "Answer: <reward_model.ground_truth>",
        "warning": "No teacher reasoning is present in the GRPO parquet; this SFT stage only normalizes the final-answer contract.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
