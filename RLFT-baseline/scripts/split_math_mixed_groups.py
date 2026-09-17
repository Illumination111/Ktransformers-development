#!/usr/bin/env python3
"""Deterministically split verified mixed-reward prompts into disjoint gates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path, required=True)
    parser.add_argument("--direction-output", type=Path, required=True)
    parser.add_argument("--smoke-size", type=int, default=8)
    parser.add_argument("--direction-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(row: dict[str, Any]) -> str:
    extra = row.get("extra_info") or {}
    sample_hash = str(extra.get("sample_hash") or "")
    if not sample_hash:
        raise ValueError("every input row must have extra_info.sample_hash")
    return sample_hash


def stable_key(row: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{identity(row)}".encode()).hexdigest()


def balanced_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: stable_key(item, seed)):
        extra = row.get("extra_info") or {}
        bucket = (str(extra.get("raw_source", "unknown")), str(extra.get("problem_type", "unknown")))
        buckets[bucket].append(row)
    ordered: list[dict[str, Any]] = []
    active = sorted(buckets, key=lambda key: hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest())
    while active:
        next_active = []
        for bucket in active:
            ordered.append(buckets[bucket].popleft())
            if buckets[bucket]:
                next_active.append(bucket)
        active = next_active
    return ordered


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str((row.get("extra_info") or {}).get(field, "unknown")) for row in rows).items()))


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.smoke_size < 1 or args.direction_size < 1:
        raise ValueError("split sizes must be positive")
    for path in (args.smoke_output, args.direction_output):
        if path.exists():
            raise FileExistsError(path)

    from datasets import Dataset

    source = Dataset.from_parquet(str(args.input))
    rows = [copy.deepcopy(dict(row)) for row in source]
    identities = [identity(row) for row in rows]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate sample_hash in input")
    requested = args.smoke_size + args.direction_size
    if len(rows) < requested:
        raise ValueError(f"requested {requested} rows from {len(rows)}")

    ordered = balanced_order(rows, args.seed)
    smoke = ordered[: args.smoke_size]
    direction = ordered[args.smoke_size : requested]
    for split_name, selected in (("smoke", smoke), ("direction", direction)):
        for row in selected:
            row.setdefault("extra_info", {})["grpo_gate_split"] = split_name

    smoke_hashes = {identity(row) for row in smoke}
    direction_hashes = {identity(row) for row in direction}
    if smoke_hashes & direction_hashes:
        raise RuntimeError("smoke and direction splits overlap")

    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    args.direction_output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(smoke).to_parquet(str(args.smoke_output))
    Dataset.from_list(direction).to_parquet(str(args.direction_output))
    manifest = {
        "protocol": "math-verified-mixed-gate-split-v1",
        "input": {"path": str(args.input.resolve()), "rows": len(rows), "sha256": sha256_file(args.input)},
        "method": "seeded SHA-256 order, round-robin balanced by raw_source and problem_type",
        "seed": args.seed,
        "disjoint": True,
        "smoke": {
            "path": str(args.smoke_output.resolve()),
            "rows": len(smoke),
            "sha256": sha256_file(args.smoke_output),
            "sample_hashes": sorted(smoke_hashes),
            "raw_source": distribution(smoke, "raw_source"),
            "problem_type": distribution(smoke, "problem_type"),
        },
        "direction": {
            "path": str(args.direction_output.resolve()),
            "rows": len(direction),
            "sha256": sha256_file(args.direction_output),
            "sample_hashes": sorted(direction_hashes),
            "raw_source": distribution(direction, "raw_source"),
            "problem_type": distribution(direction, "problem_type"),
        },
    }
    manifest_path = args.smoke_output.parent / "split.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
