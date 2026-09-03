#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into an HF model and save standalone weights."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--chat-eos-token",
        default="<|im_end|>",
        help="EOS token used by the chat template after SFT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    adapter_path = args.adapter.resolve()
    output_path = args.output.resolve()
    incomplete_path = output_path.with_name(output_path.name + ".incomplete")

    for required in (model_path / "config.json", adapter_path / "adapter_config.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    if incomplete_path.exists():
        raise FileExistsError(f"incomplete output already exists: {incomplete_path}")

    print(f"Loading base model from {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapter from {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    print("Merging LoRA into base weights", flush=True)
    model = model.merge_and_unload(safe_merge=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    chat_eos_id = tokenizer.convert_tokens_to_ids(args.chat_eos_token)
    if chat_eos_id == tokenizer.unk_token_id:
        raise ValueError(f"unknown chat EOS token: {args.chat_eos_token}")
    tokenizer.eos_token = args.chat_eos_token
    model.config.eos_token_id = chat_eos_id
    model.generation_config.eos_token_id = chat_eos_id
    print(f"Using chat EOS {args.chat_eos_token!r} (token id {chat_eos_id})", flush=True)

    incomplete_path.mkdir(parents=True)
    print(f"Saving standalone model to {incomplete_path}", flush=True)
    model.save_pretrained(
        incomplete_path,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(incomplete_path)

    if not (incomplete_path / "model.safetensors.index.json").is_file():
        raise RuntimeError("merged model index was not written")
    if (incomplete_path / "adapter_model.safetensors").exists():
        raise RuntimeError("output unexpectedly contains a separate PEFT adapter")
    os.rename(incomplete_path, output_path)
    print(f"Merged standalone model ready: {output_path}", flush=True)


if __name__ == "__main__":
    main()
