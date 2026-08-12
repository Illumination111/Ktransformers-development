#!/usr/bin/env python3
"""Validate the requested modality scope in a saved Qwen3-VL LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


SCOPES = ("text", "vision", "all")


def is_visual(key: str) -> bool:
    return ".visual." in key or key.startswith("visual.")


def validate_adapter(output_dir: Path, scope: str) -> dict[str, int | str]:
    config_path = output_dir / "adapter_config.json"
    weights_path = output_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise RuntimeError(f"missing adapter output below {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise RuntimeError(f"unexpected peft_type: {config.get('peft_type')!r}")
    with safe_open(weights_path, framework="pt", device="cpu") as adapter:
        keys = [key for key in adapter.keys() if "lora_" in key]
    text = [key for key in keys if not is_visual(key)]
    vision = [key for key in keys if is_visual(key)]
    valid = {
        "text": bool(text) and not vision,
        "vision": bool(vision) and not text,
        "all": bool(text) and bool(vision),
    }[scope]
    if not valid:
        raise RuntimeError(f"adapter scope mismatch: scope={scope}, text={len(text)}, vision={len(vision)}")
    return {"scope": scope, "lora_tensors": len(keys), "text_lora": len(text), "visual_lora": len(vision)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-scope", choices=SCOPES, required=True)
    args = parser.parse_args()
    summary = validate_adapter(args.output_dir.resolve(), args.lora_scope)
    print(f"[qwen3vl_adapter] PASS {json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
