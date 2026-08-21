#!/usr/bin/env python3
"""Generate one exact 64-token trajectory with native vLLM."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from probability_utils import write_json


def chosen_logprob(candidates: dict[Any, Any], token_id: int) -> float:
    item = candidates.get(token_id, candidates.get(str(token_id)))
    if item is None:
        raise RuntimeError(f"chosen token {token_id} absent from returned logprobs")
    return float(item.logprob)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(source["model_path"], trust_remote_code=True, local_files_only=True)
    engine = LLM(
        model=source["model_path"],
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        logprobs_mode="processed_logprobs",
        seed=args.seed,
    )
    params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=64,
        min_tokens=64,
        ignore_eos=True,
        seed=args.seed,
        logprobs=1,
    )
    output = engine.generate({"prompt_token_ids": source["prompt_ids"]}, params, use_tqdm=False)[0].outputs[0]
    response_ids = [int(token_id) for token_id in output.token_ids]
    if len(response_ids) != 64 or output.logprobs is None or len(output.logprobs) != 64:
        raise RuntimeError("vLLM did not return exactly 64 response tokens and logprobs")
    logprobs = [
        chosen_logprob(candidates, token_id)
        for token_id, candidates in zip(response_ids, output.logprobs, strict=True)
    ]
    result = {
        **source,
        "response_ids": response_ids,
        "response_text": tokenizer.decode(response_ids, skip_special_tokens=False),
        "rollout_logprobs": logprobs,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 64,
            "ignore_eos": True,
            "seed": args.seed,
        },
        "engine": {
            "name": "vllm",
            "version": importlib.metadata.version("vllm"),
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": "bfloat16",
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "logprobs_mode": "processed_logprobs",
        },
    }
    write_json(args.output, result)
    print(args.output)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
