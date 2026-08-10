#!/usr/bin/env python3
"""Validate that the smoke run saved the requested VLM LoRA scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


VALID_LORA_SCOPES = ("text", "vision", "all")


def is_visual_lora(key: str) -> bool:
    return ".visual." in key or key.startswith("visual.")


def validate_adapter(
    output_dir: Path, lora_scope: str = "text"
) -> dict[str, int | str]:
    if lora_scope not in VALID_LORA_SCOPES:
        raise ValueError(f"invalid LoRA scope: {lora_scope!r}")
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
    visual_lora = [key for key in lora_keys if is_visual_lora(key)]
    text_lora = [key for key in lora_keys if not is_visual_lora(key)]
    if lora_scope == "text" and (not text_lora or visual_lora):
        raise RuntimeError(
            f"text adapter scope mismatch: text_lora={len(text_lora)}, visual_lora={len(visual_lora)}"
        )
    if lora_scope == "vision" and (not visual_lora or text_lora):
        raise RuntimeError(
            f"vision adapter scope mismatch: text_lora={len(text_lora)}, visual_lora={len(visual_lora)}"
        )
    if lora_scope == "all" and (not text_lora or not visual_lora):
        raise RuntimeError(
            f"all adapter scope requires both modalities: text_lora={len(text_lora)}, visual_lora={len(visual_lora)}"
        )
    return {
        "file": str(weights_path),
        "scope": lora_scope,
        "tensors": len(keys),
        "lora_tensors": len(lora_keys),
        "text_lora": len(text_lora),
        "visual_lora": len(visual_lora),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-scope", choices=VALID_LORA_SCOPES, default="text")
    args = parser.parse_args()
    try:
        summary = validate_adapter(args.output_dir, args.lora_scope)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        "[qwen35_vlm_adapter] PASS "
        f"scope={summary['scope']} file={summary['file']} tensors={summary['tensors']} "
        f"lora_tensors={summary['lora_tensors']} text_lora={summary['text_lora']} "
        f"visual_lora={summary['visual_lora']}"
    )


if __name__ == "__main__":
    main()
