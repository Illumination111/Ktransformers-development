#!/usr/bin/env python3
"""Weight-free preflight for the Qwen3.5-122B-A10B VLM LoRA smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_ARCH = "Qwen3_5MoeForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_5_moe"
EXPECTED_LAYERS = 48
EXPECTED_EXPERTS = 256
EXPECTED_TOP_K = 8


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = load_json(path)
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, dict) for row in rows)
    ):
        fail(f"dataset must be a non-empty JSON/JSONL list of objects: {path}")
    return rows


def validate_model(model_path: Path) -> tuple[Any, dict[str, Any]]:
    config_raw = load_json(model_path / "config.json")
    archs = config_raw.get("architectures") or []
    text = config_raw.get("text_config") or {}
    vision = config_raw.get("vision_config") or {}
    checks = {
        "architecture": archs == [EXPECTED_ARCH],
        "model_type": config_raw.get("model_type") == EXPECTED_MODEL_TYPE,
        "text_model_type": text.get("model_type") == "qwen3_5_moe_text",
        "layers": text.get("num_hidden_layers") == EXPECTED_LAYERS,
        "experts": text.get("num_experts") == EXPECTED_EXPERTS,
        "top_k": text.get("num_experts_per_tok") == EXPECTED_TOP_K,
        "vision_depth": vision.get("depth") == 27,
        "conv3d_kernel": [
            vision.get("temporal_patch_size"),
            vision.get("patch_size"),
            vision.get("patch_size"),
        ]
        == [2, 16, 16],
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        fail(f"checkpoint contract mismatch: {bad}")

    for name in (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer.json",
    ):
        if not (model_path / name).is_file():
            fail(f"full VLM processor asset is missing: {model_path / name}")

    index = load_json(model_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map") or {}
    keys = weight_map.keys()
    if not any(key.startswith("model.visual.") for key in keys):
        fail("checkpoint index has no model.visual.* weights")
    if not any(key.startswith("model.language_model.layers.") for key in keys):
        fail("checkpoint index has no model.language_model.layers.* weights")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    from kt_kernel.sft.arch import _get_layers_prefix, get_moe_arch_config

    moe = get_moe_arch_config(config)
    if _get_layers_prefix(config) != "model.language_model.layers":
        fail("KT did not resolve the Qwen3.5 VLM language-layer prefix")
    if (moe.expert_num, moe.intermediate_size, moe.num_experts_per_tok) != (
        256,
        1024,
        8,
    ):
        fail("KT MoE architecture values do not match the 122B-A10B checkpoint")
    return config, config_raw


def validate_dataset(
    dataset_dir: Path, dataset_name: str
) -> tuple[Path, list[dict[str, Any]], list[Path]]:
    from PIL import Image

    info = load_json(dataset_dir / "dataset_info.json")
    entry = info.get(dataset_name)
    if not isinstance(entry, dict):
        fail(
            f"dataset {dataset_name!r} is not registered in {dataset_dir / 'dataset_info.json'}"
        )
    file_name = entry.get("file_name")
    if not file_name:
        fail(
            f"dataset {dataset_name!r} is hub-only; this smoke test requires a local file_name"
        )
    data_path = dataset_dir / file_name
    rows = load_rows(data_path)
    image_column = (entry.get("columns") or {}).get("images", "images")
    image_paths: list[Path] = []
    placeholder_count = 0
    assistant_targets = 0
    for row_index, row in enumerate(rows):
        refs = row.get(image_column)
        if not isinstance(refs, list) or not refs:
            fail(f"row {row_index} has no non-empty {image_column!r} image list")
        messages = row.get((entry.get("columns") or {}).get("messages", "messages"), [])
        if not isinstance(messages, list) or not messages:
            fail(f"row {row_index} has no messages")
        if not any(
            isinstance(message, dict)
            and message.get("role") == "user"
            and "<image>" in str(message.get("content", ""))
            for message in messages
        ):
            fail(f"row {row_index} has no user image prompt")
        row_targets = [
            message
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and str(message.get("content", "")).strip()
        ]
        if not row_targets:
            fail(f"row {row_index} has no non-empty assistant training target")
        assistant_targets += len(row_targets)
        placeholder_count += sum(
            str(message.get("content", "")).count("<image>")
            for message in messages
            if isinstance(message, dict)
        )
        for ref in refs:
            path = Path(ref)
            if not path.is_absolute():
                path = dataset_dir / path
            if not path.is_file():
                fail(f"row {row_index} references missing image: {path}")
            image_paths.append(path)
    if placeholder_count != len(image_paths):
        fail(
            f"image placeholder/reference mismatch: placeholders={placeholder_count}, images={len(image_paths)}"
        )
    for path in sorted(set(image_paths)):
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            fail(f"image cannot be decoded: {path}: {exc}")
    if assistant_targets < len(rows):
        fail(
            f"dataset has too few assistant targets: {assistant_targets} for {len(rows)} rows"
        )
    return data_path, rows, image_paths


def validate_processor(
    model_path: Path, image_path: Path, config_raw: dict[str, Any]
) -> dict[str, Any]:
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    batch = processor(text=[prompt], images=[image], return_tensors="pt")
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    missing = required.difference(batch)
    if missing:
        fail(f"processor output is missing VLM tensors: {sorted(missing)}")
    vision = config_raw["vision_config"]
    patch_width = (
        vision["in_channels"]
        * vision["temporal_patch_size"]
        * vision["patch_size"] ** 2
    )
    if (
        batch["pixel_values"].ndim != 2
        or batch["pixel_values"].shape[-1] != patch_width
    ):
        fail(
            f"unexpected pixel_values shape: {tuple(batch['pixel_values'].shape)}; expected width {patch_width}"
        )
    return {
        "processor": type(processor).__name__,
        "image_processor": type(processor.image_processor).__name__,
        "pixel_values": list(batch["pixel_values"].shape),
        "image_grid_thw": batch["image_grid_thw"].tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers

    model_path = args.model_path.resolve()
    dataset_dir = args.dataset_dir.resolve()
    config, config_raw = validate_model(model_path)
    data_path, rows, images = validate_dataset(dataset_dir, args.dataset_name)
    processor_summary = validate_processor(model_path, images[0], config_raw)
    from load_conv3d_compat import load_conv3d_compat

    conv3d_summary = load_conv3d_compat().self_test_swift_conv3d_patch()

    if args.require_cuda and not torch.cuda.is_available():
        fail(
            "CUDA is not visible; an actual 122B smoke run requires eight visible GPUs"
        )

    summary = {
        "status": "ok",
        "model": str(model_path),
        "architecture": config.architectures[0],
        "language_layer_prefix": "model.language_model.layers",
        "dataset": args.dataset_name,
        "dataset_file": str(data_path),
        "rows": len(rows),
        "image_references": len(images),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_visible": torch.cuda.is_available(),
        "dataset_functional_scope": "image processor + frozen vision forward + language LoRA gradient/optimizer smoke",
        "conv3d_compatibility": conv3d_summary,
        **processor_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PRECHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
