#!/usr/bin/env python3
"""Freeze one prompt as exact token IDs for the logprob contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from probability_utils import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Solve carefully and put the final answer in \\boxed{}: If 3x + 7 = 25, what is x?",
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    messages = [{"role": "user", "content": args.prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
    )
    write_json(
        args.output,
        {
            "model_path": str(args.model_path.resolve()),
            "prompt": args.prompt,
            "messages": messages,
            "prompt_ids": prompt_ids,
            "prompt_length": len(prompt_ids),
            "chat_template": tokenizer.chat_template,
            "enable_thinking": True,
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
