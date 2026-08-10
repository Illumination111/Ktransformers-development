#!/usr/bin/env python3
"""Render the immutable smoke-test YAML template into a per-run config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument(
        "--lora-scope", choices=("text", "vision", "all"), default="text"
    )
    args = parser.parse_args()
    if args.max_steps < 1 or args.cutoff_len < 32:
        parser.error("max-steps must be >= 1 and cutoff-len must be >= 32")

    replacements = {
        "__MODEL_PATH__": str(args.model_path.resolve()),
        "__DATASET_DIR__": str(args.dataset_dir.resolve()),
        "__DATASET_NAME__": args.dataset_name,
        "__OUTPUT_DIR__": str(args.model_output.resolve()),
        "__MAX_STEPS__": str(args.max_steps),
        "__CUTOFF_LEN__": str(args.cutoff_len),
        "__LORA_SCOPE__": args.lora_scope,
    }
    rendered = args.template.read_text(encoding="utf-8")
    for marker, value in replacements.items():
        if marker not in rendered:
            parser.error(f"template marker is missing: {marker}")
        rendered = rendered.replace(marker, value)
    unresolved = sorted(set(re.findall(r"__[A-Z0-9_]+__", rendered)))
    if unresolved:
        parser.error(f"unresolved template markers: {unresolved}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
