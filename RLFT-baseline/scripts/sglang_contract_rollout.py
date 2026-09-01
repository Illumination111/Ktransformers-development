#!/usr/bin/env python3
"""Generate one exact 64-token trajectory with native SGLang."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from probability_utils import write_json


def output_logprob(item: Any, token_id: int) -> float:
    if not isinstance(item, (list, tuple)) or not item:
        raise RuntimeError(f"invalid SGLang output logprob entry: {item!r}")
    if len(item) > 1 and item[1] is not None and int(item[1]) != token_id:
        raise RuntimeError(f"SGLang logprob token id {item[1]} does not match output id {token_id}")
    return float(item[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import sglang as sgl
    from transformers import AutoTokenizer

    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(source["model_path"], trust_remote_code=True, local_files_only=True)
    engine = sgl.Engine(
        model_path=source["model_path"],
        tp_size=args.tensor_parallel_size,
        dtype="bfloat16",
        mem_fraction_static=args.gpu_memory_utilization,
        context_length=8192,
        max_running_requests=1,
        chunked_prefill_size=8192,
        trust_remote_code=True,
        random_seed=args.seed,
    )
    try:
        sampling = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": 64,
            "min_new_tokens": 64,
            "ignore_eos": True,
            "sampling_seed": args.seed,
        }
        output = engine.generate(
            input_ids=source["prompt_ids"],
            sampling_params=sampling,
            return_logprob=True,
        )
        response_ids = [int(token_id) for token_id in output.get("output_ids", [])]
        raw_logprobs = output.get("meta_info", {}).get("output_token_logprobs") or []
        if len(response_ids) != 64 or len(raw_logprobs) != 64:
            raise RuntimeError("SGLang did not return exactly 64 response tokens and logprobs")
        logprobs = [
            output_logprob(item, token_id)
            for token_id, item in zip(response_ids, raw_logprobs, strict=True)
        ]
        result = {
            **source,
            "response_ids": response_ids,
            "response_text": output.get("text") or tokenizer.decode(response_ids, skip_special_tokens=False),
            "rollout_logprobs": logprobs,
            "sampling": sampling,
            "engine": {
                "name": "sglang",
                "version": importlib.metadata.version("sglang"),
                "tensor_parallel_size": args.tensor_parallel_size,
                "dtype": "bfloat16",
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
        }
        write_json(args.output, result)
        print(args.output)
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
