#!/usr/bin/env python3
"""Merge per-rank exact Qwen3.5 routes into one APTMoE replay pattern."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aptmoe_proxy.storage import (
    DEFAULT_SIMULATION_ROOT,
    require_within_simulation_root,
)


def group_route_patterns_by_optimizer_step(
    routes: np.ndarray,
    accumulation_steps: int,
) -> np.ndarray:
    """Combine exact microbatch routes that used the same model weights.

    During gradient accumulation, all microbatches in one optimizer step route
    with the same parameter state. Concatenating their token axes therefore
    gives the exact route pattern for an equivalent larger microbatch without
    increasing the per-rank activation footprint during capture.
    """

    if routes.ndim != 4:
        raise ValueError("route patterns must have rank 4")
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if routes.shape[0] % accumulation_steps:
        raise ValueError(
            "captured route patterns must be divisible by accumulation_steps"
        )
    if accumulation_steps == 1:
        return routes
    output_patterns = routes.shape[0] // accumulation_steps
    grouped = routes.reshape(
        output_patterns,
        accumulation_steps,
        routes.shape[1],
        routes.shape[2],
        routes.shape[3],
    )
    return grouped.transpose(0, 2, 1, 3, 4).reshape(
        output_patterns,
        routes.shape[1],
        accumulation_steps * routes.shape[2],
        routes.shape[3],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--expected-patterns", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument(
        "--source-accumulation-steps",
        type=int,
        default=1,
        help=(
            "group this many captured microbatch patterns from each optimizer "
            "step into one equivalent larger-batch output pattern"
        ),
    )
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=DEFAULT_SIMULATION_ROOT,
    )
    args = parser.parse_args()
    if min(
        args.expected_ranks,
        args.expected_patterns,
        args.sequence_length,
        args.global_batch_size,
        args.source_accumulation_steps,
    ) <= 0:
        raise SystemExit(
            "rank, pattern, sequence, batch, and accumulation sizes "
            "must be positive"
        )
    if args.expected_patterns % args.source_accumulation_steps:
        raise SystemExit(
            "expected patterns must be divisible by source accumulation steps"
        )

    output = require_within_simulation_root(
        args.output,
        args.simulation_root,
    )
    rank_files = [
        args.input_dir / f"rank_{rank:02d}.npz"
        for rank in range(args.expected_ranks)
    ]
    missing = [str(path) for path in rank_files if not path.is_file()]
    if missing:
        raise SystemExit(f"missing route rank files: {missing}")

    arrays: list[np.ndarray] = []
    source_metadata: list[dict] = []
    source_backend: str | None = None
    source_partitions = (
        args.expected_ranks * args.source_accumulation_steps
    )
    if args.global_batch_size % source_partitions != 0:
        raise SystemExit(
            "output global batch must be divisible by "
            "expected ranks * source accumulation steps"
        )
    expected_tokens_per_rank = (
        args.sequence_length
        * args.global_batch_size
        // source_partitions
    )
    for expected_rank, path in enumerate(rank_files):
        with np.load(path, allow_pickle=False) as data:
            array = np.asarray(data["topk_indices"])
            raw_metadata = data["metadata_json"].item()
        if (
            array.ndim != 4
            or array.shape[0] != args.expected_patterns
            or array.shape[1] != 40
            or array.shape[3] != 8
        ):
            raise SystemExit(f"invalid route shape in {path}: {array.shape}")
        metadata = json.loads(str(raw_metadata))
        expected_metadata = {
            "source": "exact_qwen35_router_forward_hook",
            "rank": expected_rank,
            "world_size": args.expected_ranks,
            "sequence_length": args.sequence_length,
            "patterns": args.expected_patterns,
            "layers": 40,
            "tokens_on_rank": expected_tokens_per_rank,
            "top_k": 8,
        }
        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                raise SystemExit(
                    f"{path} metadata {key}={metadata.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        backend = metadata.get("backend")
        if backend not in {"kt", "ktransformers", "deepspeed"}:
            raise SystemExit(
                f"{path} metadata backend={backend!r} is not an exact backend"
            )
        if source_backend is None:
            source_backend = backend
        elif backend != source_backend:
            raise SystemExit(
                f"{path} metadata backend={backend!r}, "
                f"expected {source_backend!r}"
            )
        arrays.append(array.astype(np.int16, copy=False))
        source_metadata.append(metadata)

    merged_microbatches = np.concatenate(arrays, axis=2)
    source_global_batch_size = (
        args.global_batch_size // args.source_accumulation_steps
    )
    expected_microbatch_tokens = (
        args.sequence_length * source_global_batch_size
    )
    expected_microbatch_shape = (
        args.expected_patterns,
        40,
        expected_microbatch_tokens,
        8,
    )
    if merged_microbatches.shape != expected_microbatch_shape:
        raise SystemExit(
            f"merged source route shape={merged_microbatches.shape}, "
            f"expected={expected_microbatch_shape}"
        )
    merged = group_route_patterns_by_optimizer_step(
        merged_microbatches,
        args.source_accumulation_steps,
    )
    output_patterns = (
        args.expected_patterns // args.source_accumulation_steps
    )
    expected_tokens = args.sequence_length * args.global_batch_size
    if merged.shape != (output_patterns, 40, expected_tokens, 8):
        raise SystemExit(
            f"merged route shape={merged.shape}, "
            f"expected={(output_patterns, 40, expected_tokens, 8)}"
        )
    metadata = {
        "schema_version": 1,
        "source": "merged_exact_qwen35_router_trace",
        "source_backend": source_backend,
        "sequence_length": args.sequence_length,
        "global_batch_size": args.global_batch_size,
        "patterns": output_patterns,
        "layers": 40,
        "tokens": expected_tokens,
        "top_k": 8,
        "source_world_size": args.expected_ranks,
        "source_global_microbatch_size": source_global_batch_size,
        "source_per_device_microbatch_size": (
            args.global_batch_size // source_partitions
        ),
        "source_accumulation_steps": args.source_accumulation_steps,
        "source_capture_patterns": args.expected_patterns,
        "aggregation": (
            "rank_concat"
            if args.source_accumulation_steps == 1
            else "rank_concat_then_optimizer_step_accumulation_concat"
        ),
        "rank_files": [str(path.resolve()) for path in rank_files],
        "source_metadata": source_metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        topk_indices=merged,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"[merge_routes] {merged.shape} -> {output}")


if __name__ == "__main__":
    main()
