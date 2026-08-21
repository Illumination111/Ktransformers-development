#!/usr/bin/env python3
"""Validate plain-HF rollout logprobs against a fixed-response HF score."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt", default="Explain why deterministic inference is useful for RL training.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--rollout-mode", choices=("cached", "no_cache"), default="cached")
    parser.add_argument("--hf-device-map", choices=("auto", "balanced"), default="balanced")
    parser.add_argument("--hf-max-memory-gib", type=int, default=44)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise RuntimeError(f"length mismatch: {len(reference)} vs {len(candidate)}")
    diffs = [candidate[i] - reference[i] for i in range(len(reference))]
    abs_diffs = [abs(x) for x in diffs]
    ratios = [math.exp(x) for x in diffs]
    return {
        "tokens": len(diffs),
        "mean_abs_diff": sum(abs_diffs) / len(diffs),
        "max_abs_diff": max(abs_diffs),
        "p95_abs_diff": sorted(abs_diffs)[max(0, int(len(abs_diffs) * 0.95) - 1)],
        "sum_diff": sum(diffs),
        "sequence_ratio": math.exp(sum(diffs)),
        "mean_ratio": sum(ratios) / len(ratios),
        "max_ratio_deviation": max(abs(x - 1.0) for x in ratios),
        "clip_fraction_eps_0.2": sum(abs(x - 1.0) > 0.2 for x in ratios) / len(ratios),
        "diffs": diffs,
    }


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required for the 30B plain-HF validation")
    model_path = args.model_path.expanduser()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "eager"
    max_memory = {
        index: f"{args.hf_max_memory_gib}GiB" for index in range(torch.cuda.device_count())
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.hf_device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    prompt_ids = tokenizer(args.prompt, add_special_tokens=True, return_tensors=None)["input_ids"]
    if len(prompt_ids) < 2:
        raise RuntimeError("prompt must contain at least two tokens")
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=input_device)

    # Rollout path: incremental cached forward, preserving raw (unwarped)
    # model logprob before sampling. top-p=1 has no filtering effect.
    torch.manual_seed(args.seed)
    current_ids = prompt
    attention_mask = torch.ones_like(prompt)
    past = None
    response_ids: list[int] = []
    rollout_logprobs: list[float] = []
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            if args.rollout_mode == "no_cache":
                current_ids = torch.tensor(
                    [prompt_ids + response_ids], dtype=torch.long, device=input_device
                )
                attention_mask = torch.ones_like(current_ids)
                past = None
            outputs = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :].float()
            if args.temperature <= 0:
                raise ValueError("temperature must be positive")
            logits = logits / args.temperature
            logprobs = torch.log_softmax(logits, dim=-1)
            probs = logprobs.exp()
            next_id = torch.multinomial(probs, num_samples=1)
            response_ids.append(int(next_id.item()))
            rollout_logprobs.append(float(logprobs[0, next_id.item()].item()))
            if args.rollout_mode == "cached":
                past = outputs.past_key_values
                current_ids = next_id
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=torch.long, device=input_device)], dim=1
                )

        full_ids = prompt_ids + response_ids
        full = torch.tensor([full_ids], dtype=torch.long, device=input_device)
        full_mask = torch.ones_like(full)
        scored = model(
            input_ids=full,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
        )
        full_logprobs = torch.log_softmax(scored.logits[0].float(), dim=-1)
        targets = torch.tensor(full_ids[1:], dtype=torch.long, device=full_logprobs.device)
        selected = full_logprobs[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
        score_logprobs = selected[len(prompt_ids) - 1 :].cpu().tolist()

    result = {
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(model_path),
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "config": {
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "device_map": args.hf_device_map,
            "max_memory_gib_per_visible_gpu": args.hf_max_memory_gib,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "rollout_mode": args.rollout_mode,
            "visible_gpu_count": torch.cuda.device_count(),
        },
        "rollout_logprobs": rollout_logprobs,
        "score_logprobs": score_logprobs,
        "rollout_vs_score": summarize(rollout_logprobs, score_logprobs),
        "method": (
            f"plain HF {args.rollout_mode} rollout followed by fixed-response "
            "full teacher-forced score"
        ),
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"report={args.output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
