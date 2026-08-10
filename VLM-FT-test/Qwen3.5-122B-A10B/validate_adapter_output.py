#!/usr/bin/env python3
"""Validate that the smoke run saved a language-side LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def validate_adapter(output_dir: Path) -> dict[str, int | str]:
    output_dir = output_dir.resolve()
    config_path = output_dir / "adapter_config.json"
    weights_path = output_dir / "adapter_model.safetensors"
    if not config_path.is_file():
        raise RuntimeError(f"missing adapter config: {config_path}")
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty adapter weights: {weights_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise RuntimeError(f"unexpected peft_type: {config.get('peft_type')!r}")
    with safe_open(weights_path, framework="pt", device="cpu") as adapter:
        keys = list(adapter.keys())
    lora_keys = [key for key in keys if "lora_" in key]
    if not lora_keys:
        raise RuntimeError("saved adapter contains no LoRA tensors")
    visual_lora = [
        key for key in lora_keys if ".visual." in key or key.startswith("visual.")
    ]
    if visual_lora:
        raise RuntimeError(
            f"saved adapter contains frozen-vision LoRA tensors: {visual_lora[:5]}"
        )
    return {
        "file": str(weights_path),
        "tensors": len(keys),
        "lora_tensors": len(lora_keys),
        "visual_lora": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = validate_adapter(args.output_dir)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        "[qwen35_vlm_adapter] PASS "
        f"file={summary['file']} tensors={summary['tensors']} "
        f"lora_tensors={summary['lora_tensors']} visual_lora=0"
    )


if __name__ == "__main__":
    main()
