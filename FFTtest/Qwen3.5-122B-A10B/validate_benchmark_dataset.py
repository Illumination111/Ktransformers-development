#!/usr/bin/env python3
"""Validate Qwen3.5-122B-A10B text-only BF16 checkpoint and dataset."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EXPECTED_SOURCE_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
EXPECTED_SOURCE_MODEL_TYPE = "qwen3_5_moe"
EXPECTED_TEXT_ARCHITECTURE = "Qwen3_5MoeForCausalLM"
EXPECTED_TEXT_MODEL_TYPE = "qwen3_5_moe_text"
EXPECTED_SHARDS = 39
EXPECTED_TEXT_PARAMETERS = 122_111_526_912
EXPECTED_TEXT_FIELDS = {
    "hidden_size": 3072,
    "num_hidden_layers": 48,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "full_attention_interval": 4,
}


def resolve_dataset_file(dataset_dir: Path, dataset_name: str) -> Path:
    info_path = dataset_dir / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    entry = info.get(dataset_name)
    if not isinstance(entry, dict) or not entry.get("file_name"):
        raise ValueError(f"Dataset {dataset_name!r} is not registered in {info_path}")
    path = dataset_dir / str(entry["file_name"])
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file is missing: {path}")
    return path


def validate_model(model_path: Path) -> dict[str, object]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if EXPECTED_SOURCE_ARCHITECTURE not in architectures:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_ARCHITECTURE}, got {architectures}"
        )
    if config.get("model_type") != EXPECTED_SOURCE_MODEL_TYPE:
        raise ValueError(
            f"Expected source model_type={EXPECTED_SOURCE_MODEL_TYPE!r}, "
            f"got {config.get('model_type')!r}"
        )
    text_config = config.get("text_config") or {}
    if text_config.get("model_type") != EXPECTED_TEXT_MODEL_TYPE:
        raise ValueError(
            f"Expected {EXPECTED_TEXT_MODEL_TYPE} in the multimodal "
            "checkpoint's text_config, "
            f"got {text_config.get('model_type')!r}"
        )
    mismatches = {
        key: {"actual": text_config.get(key), "expected": expected}
        for key, expected in EXPECTED_TEXT_FIELDS.items()
        if text_config.get(key) != expected
    }
    expected_layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(EXPECTED_TEXT_FIELDS["num_hidden_layers"])
    ]
    if text_config.get("layer_types") != expected_layer_types:
        mismatches["layer_types"] = {
            "actual": text_config.get("layer_types"),
            "expected": expected_layer_types,
        }
    if mismatches:
        raise ValueError(
            "Checkpoint does not match Qwen3.5-122B-A10B text configuration: "
            f"{mismatches}"
        )
    dtype = str(text_config.get("dtype", text_config.get("torch_dtype", ""))).lower()
    if dtype not in {"bfloat16", "bf16"}:
        raise ValueError(f"Model is not declared BF16: text_config dtype={dtype!r}")

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Safetensors index is missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    shards = sorted(set(weight_map.values()))
    if len(shards) != EXPECTED_SHARDS:
        raise ValueError(
            f"Expected {EXPECTED_SHARDS} checkpoint shards, got {len(shards)}"
        )
    missing = [name for name in shards if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} model shards: {missing[:3]}")
    text_weight_keys = sum(
        key.startswith(("model.language_model.", "lm_head."))
        for key in weight_map
    )
    visual_weight_keys = sum(key.startswith("model.visual.") for key in weight_map)
    mtp_weight_keys = sum(key.startswith("mtp.") for key in weight_map)
    if text_weight_keys == 0 or visual_weight_keys == 0:
        raise ValueError(
            "Expected a multimodal checkpoint containing both language and visual weights; "
            f"found language={text_weight_keys}, visual={visual_weight_keys}"
        )

    from transformers import AutoConfig, AutoModelForCausalLM

    source_config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    runtime_text_config = source_config.text_config
    runtime_text_config.architectures = [EXPECTED_TEXT_ARCHITECTURE]
    mapped_class = AutoModelForCausalLM._model_mapping[type(runtime_text_config)].__name__
    if mapped_class != EXPECTED_TEXT_ARCHITECTURE:
        raise ValueError(
            "Transformers cannot map text_config to "
            f"{EXPECTED_TEXT_ARCHITECTURE}; got {mapped_class}"
        )

    from safetensors import safe_open

    text_parameters = 0
    for shard in shards:
        with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.startswith(("model.language_model.", "lm_head.")):
                    continue
                numel = 1
                for dimension in handle.get_slice(key).get_shape():
                    numel *= dimension
                text_parameters += numel
    if text_parameters != EXPECTED_TEXT_PARAMETERS:
        raise ValueError(
            "Unexpected text-only logical parameter count: "
            f"got={text_parameters}, expected={EXPECTED_TEXT_PARAMETERS}"
        )

    return {
        "source_architecture": architectures[0],
        "source_model_type": config.get("model_type"),
        "load_architecture": EXPECTED_TEXT_ARCHITECTURE,
        "load_model_type": text_config.get("model_type"),
        "modality": "text_only",
        "dtype": "bfloat16",
        "model_shards": len(shards),
        "text_logical_parameters": text_parameters,
        "language_weight_keys": text_weight_keys,
        "excluded_visual_weight_keys": visual_weight_keys,
        "excluded_mtp_weight_keys": mtp_weight_keys,
    }


def validate_text_only_rows(rows: list[dict[str, object]]) -> None:
    multimodal_fields = {
        "image",
        "images",
        "video",
        "videos",
        "audio",
        "audios",
    }
    invalid: list[tuple[int, list[str]]] = []
    for index, row in enumerate(rows):
        fields = sorted(multimodal_fields.intersection(row))
        if fields:
            invalid.append((index, fields))
    if invalid:
        raise ValueError(
            "Text-only benchmark dataset contains multimodal fields; "
            f"first rows: {invalid[:5]}"
        )


def token_lengths(model_path: Path, rows: list[dict[str, object]]) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    texts = [
        "\n".join(
            str(row.get(key, "")) for key in ("instruction", "input", "output")
        )
        for row in rows
    ]
    lengths: list[int] = []
    for start in range(0, len(texts), 8):
        encoded = tokenizer(
            texts[start : start + 8], add_special_tokens=True, truncation=False
        )["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--required-length", type=int, default=4096)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    model_info = validate_model(args.model_path)
    dataset_file = resolve_dataset_file(args.dataset_dir, args.dataset_name)
    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Dataset must be a non-empty JSON list: {dataset_file}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Every dataset sample must be an object: {dataset_file}")
    validate_text_only_rows(rows)
    lengths = token_lengths(args.model_path, rows)
    too_short = [i for i, length in enumerate(lengths) if length <= args.required_length]
    if too_short:
        raise ValueError(
            f"{len(too_short)} samples are <= {args.required_length} tokens; "
            f"first indices: {too_short[:8]}"
        )

    result = {
        **model_info,
        "model_path": str(args.model_path.resolve()),
        "dataset_name": args.dataset_name,
        "dataset_file": str(dataset_file.resolve()),
        "samples": len(rows),
        "required_length_exclusive": args.required_length,
        "token_length_min": min(lengths),
        "token_length_max": max(lengths),
        "token_length_mean": statistics.fmean(lengths),
        "status": "OK",
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
