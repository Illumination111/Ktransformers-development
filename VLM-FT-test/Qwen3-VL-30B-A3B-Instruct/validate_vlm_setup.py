#!/usr/bin/env python3
"""Weight-free adaptation and data preflight for Qwen3-VL-30B-A3B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_ARCH = "Qwen3VLMoeForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_vl_moe"
EXPECTED_LAYERS = 48
EXPECTED_EXPERTS = 128
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


def validate_checkpoint(model_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    raw = load_json(model_path / "config.json")
    text = raw.get("text_config") or {}
    vision = raw.get("vision_config") or {}
    checks = {
        "architecture": raw.get("architectures") == [EXPECTED_ARCH],
        "model_type": raw.get("model_type") == EXPECTED_MODEL_TYPE,
        "text_model_type": text.get("model_type") == "qwen3_vl_moe_text",
        "layers": text.get("num_hidden_layers") == EXPECTED_LAYERS,
        "experts": text.get("num_experts") == EXPECTED_EXPERTS,
        "top_k": text.get("num_experts_per_tok") == EXPECTED_TOP_K,
        "moe_intermediate_size": text.get("moe_intermediate_size") == 768,
        "vision_depth": vision.get("depth") == 27,
        "conv3d_kernel": [
            vision.get("temporal_patch_size"),
            vision.get("patch_size"),
            vision.get("patch_size"),
        ] == [2, 16, 16],
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        fail(f"checkpoint contract mismatch: {bad}")

    for name in (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        if not (model_path / name).is_file():
            fail(f"full VLM asset is missing: {model_path / name}")

    index = load_json(model_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map") or {}
    if not any(key.startswith("model.visual.") for key in weight_map):
        fail("checkpoint index has no model.visual.* weights")
    if not any(key.startswith("model.language_model.layers.") for key in weight_map):
        fail("checkpoint index has no model.language_model.layers.* weights")
    shards = sorted(set(weight_map.values()))
    missing_shards = [name for name in shards if not (model_path / name).is_file()]
    empty_shards = [name for name in shards if (model_path / name).is_file() and (model_path / name).stat().st_size == 0]
    if missing_shards or empty_shards:
        fail(f"checkpoint shard mismatch: missing={missing_shards}, empty={empty_shards}")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    return config, raw, {"weight_keys": len(weight_map), "shards": len(shards)}


def validate_llamafactory() -> dict[str, Any]:
    from llamafactory.data.template import TEMPLATES
    from llamafactory.hparams import FinetuningArguments
    from llamafactory.model.model_utils.kt_vlm import _KT_VLM_CONV3D_MODEL_TYPES
    from llamafactory.model.model_utils.visual import COMPOSITE_MODELS
    from llamafactory.model.model_utils.vlm_lora import find_vlm_lora_modules

    if "qwen3_vl" not in TEMPLATES:
        fail("LLaMA-Factory has no qwen3_vl template")
    if EXPECTED_MODEL_TYPE not in COMPOSITE_MODELS:
        fail("LLaMA-Factory has no qwen3_vl_moe composite-model registration")
    if EXPECTED_MODEL_TYPE not in _KT_VLM_CONV3D_MODEL_TYPES:
        fail("LLaMA-Factory KT Conv3D path does not recognize qwen3_vl_moe")
    if "vlm_lora_scope" not in FinetuningArguments.__dataclass_fields__:
        fail("LLaMA-Factory lacks scoped VLM LoRA arguments")
    return {
        "source": str(Path(sys.modules["llamafactory"].__file__).resolve()),
        "template": "qwen3_vl",
        "composite_model": EXPECTED_MODEL_TYPE,
        "scoped_lora": callable(find_vlm_lora_modules),
        "kt_conv3d": True,
    }


def validate_dataset(dataset_dir: Path, dataset_name: str) -> tuple[Path, list[dict[str, Any]], list[Path]]:
    from PIL import Image

    info = load_json(dataset_dir / "dataset_info.json")
    entry = info.get(dataset_name)
    if not isinstance(entry, dict) or not entry.get("file_name"):
        fail(f"dataset {dataset_name!r} is not a registered local dataset")
    data_path = dataset_dir / entry["file_name"]
    rows = load_json(data_path)
    if not isinstance(rows, list) or not rows:
        fail(f"dataset must be a non-empty JSON list: {data_path}")
    columns = entry.get("columns") or {}
    image_column = columns.get("images", "images")
    message_column = columns.get("messages", "messages")
    images: list[Path] = []
    placeholders = 0
    for row_index, row in enumerate(rows):
        refs = row.get(image_column)
        messages = row.get(message_column)
        if not isinstance(refs, list) or not refs:
            fail(f"row {row_index} has no image references")
        if not isinstance(messages, list) or not messages:
            fail(f"row {row_index} has no messages")
        placeholders += sum(str(message.get("content", "")).count("<image>") for message in messages)
        if not any(message.get("role") == "assistant" and str(message.get("content", "")).strip() for message in messages):
            fail(f"row {row_index} has no assistant target")
        for ref in refs:
            path = Path(ref)
            if not path.is_absolute():
                path = dataset_dir / path
            if not path.is_file():
                fail(f"row {row_index} references missing image: {path}")
            with Image.open(path) as image:
                image.verify()
            images.append(path)
    if placeholders != len(images):
        fail(f"image placeholder/reference mismatch: {placeholders} != {len(images)}")
    return data_path, rows, images


def validate_processor(model_path: Path, image_path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image."},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[prompt], images=[image], return_tensors="pt")
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    missing = required.difference(batch)
    if missing:
        fail(f"processor output is missing VLM tensors: {sorted(missing)}")
    vision = raw["vision_config"]
    width = vision["in_channels"] * vision["temporal_patch_size"] * vision["patch_size"] ** 2
    if batch["pixel_values"].ndim != 2 or batch["pixel_values"].shape[-1] != width:
        fail(f"unexpected pixel_values shape: {tuple(batch['pixel_values'].shape)}")
    return {
        "processor": type(processor).__name__,
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
    from load_conv3d_compat import load_conv3d_compat
    from load_kt_arch_compat import audit_kt_architecture

    model_path = args.model_path.resolve()
    dataset_dir = args.dataset_dir.resolve()
    config, raw, checkpoint = validate_checkpoint(model_path)
    kt = audit_kt_architecture(config)
    if kt["source_prefix"] != "model.language_model.layers":
        fail(f"KT source resolved the wrong layer prefix: {kt['source_prefix']}")
    if (kt["source_experts"], kt["source_moe_intermediate_size"], kt["source_top_k"]) != (128, 768, 8):
        fail("KT source MoE values do not match the checkpoint")
    llamafactory = validate_llamafactory()
    data_path, rows, images = validate_dataset(dataset_dir, args.dataset_name)
    processor = validate_processor(model_path, images[0], raw)
    conv3d = load_conv3d_compat().self_test_swift_conv3d_patch()
    if args.require_cuda and not torch.cuda.is_available():
        fail("CUDA is not visible; a real smoke run requires eight visible GPUs")

    print(json.dumps({
        "status": (
            "development_only_requires_distributed_smoke"
            if not kt["installed_supported"]
            else "preflight_passed_requires_distributed_smoke"
        ),
        "model": str(model_path),
        "architecture": config.architectures[0],
        "checkpoint": checkpoint,
        "dataset": str(data_path),
        "rows": len(rows),
        "image_references": len(images),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "llamafactory_version_check_bypassed": True,
        "cuda_visible": torch.cuda.is_available(),
        "llamafactory": llamafactory,
        "ktransformers": kt,
        "conv3d_compatibility": conv3d,
        **processor,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PRECHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
