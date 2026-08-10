"""Force the Qwen3.5-122B-A10B multimodal checkpoint into a text-only CausalLM.

The local Qwen3.5-122B-A10B checkpoint is packaged as
``Qwen3_5MoeForConditionalGeneration``.  For this TPS benchmark we deliberately
load only its ``text_config`` as ``Qwen3_5MoeForCausalLM``.  Transformers 5.x
then remaps ``model.language_model.*`` checkpoint keys to ``model.*`` and
ignores the visual and MTP checkpoint keys.

This module patches only the benchmark process.  It does not modify the model
checkpoint or the shared LLaMA-Factory checkout.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from transformers import PreTrainedModel


SOURCE_MODEL_TYPE = "qwen3_5_moe"
SOURCE_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
TEXT_MODEL_TYPE = "qwen3_5_moe_text"
TEXT_ARCHITECTURE = "Qwen3_5MoeForCausalLM"
TEXT_FSDP_LAYER_CLASS = "Qwen3_5MoeDecoderLayer"
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
_INSTALLED = False


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _rank0_print(message: str) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(message, flush=True)


def _extract_text_config(source_config: Any) -> Any:
    source_type = getattr(source_config, "model_type", None)
    source_architectures = list(getattr(source_config, "architectures", None) or [])
    if source_type != SOURCE_MODEL_TYPE or SOURCE_ARCHITECTURE not in source_architectures:
        raise RuntimeError(
            "The text-only benchmark requires a Qwen3.5-MoE multimodal source checkpoint; "
            f"got model_type={source_type!r}, architectures={source_architectures!r}."
        )

    source_text_config = getattr(source_config, "text_config", None)
    if source_text_config is None or getattr(source_text_config, "model_type", None) != TEXT_MODEL_TYPE:
        raise RuntimeError("The source checkpoint does not expose a Qwen3.5-MoE text_config.")
    mismatches = {
        name: (getattr(source_text_config, name, None), expected)
        for name, expected in EXPECTED_TEXT_FIELDS.items()
        if getattr(source_text_config, name, None) != expected
    }
    layer_types = list(getattr(source_text_config, "layer_types", None) or [])
    expected_layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(EXPECTED_TEXT_FIELDS["num_hidden_layers"])
    ]
    if layer_types != expected_layer_types:
        mismatches["layer_types"] = (layer_types, expected_layer_types)
    if mismatches:
        raise RuntimeError(
            "Checkpoint is not the expected Qwen3.5-122B-A10B text model: "
            f"{mismatches}"
        )

    text_config = copy.deepcopy(source_text_config)
    text_config.architectures = [TEXT_ARCHITECTURE]
    text_config._name_or_path = getattr(source_config, "_name_or_path", "")
    text_config.tie_word_embeddings = bool(getattr(source_config, "tie_word_embeddings", False))
    return text_config


def assert_text_only_model(model: "PreTrainedModel", finetuning_type: str) -> None:
    """Fail before optimizer construction if any multimodal component survived."""
    import torch

    if finetuning_type not in {"full", "lora"}:
        raise RuntimeError(
            "This benchmark supports only full or LoRA fine-tuning, "
            f"got {finetuning_type!r}."
        )

    get_base_model = getattr(model, "get_base_model", None)
    base_model = get_base_model() if callable(get_base_model) else model
    is_adapter_wrapped = base_model is not model
    if finetuning_type == "full" and is_adapter_wrapped:
        raise RuntimeError(
            "Full fine-tuning unexpectedly constructed an adapter-wrapped model: "
            f"{type(model).__name__}."
        )
    if finetuning_type == "lora" and not is_adapter_wrapped:
        raise RuntimeError(
            "LoRA fine-tuning did not construct an adapter-wrapped model; "
            f"got {type(model).__name__}."
        )

    model_type = getattr(base_model.config, "model_type", None)
    architectures = list(getattr(base_model.config, "architectures", None) or [])
    if model_type != TEXT_MODEL_TYPE or architectures != [TEXT_ARCHITECTURE]:
        raise RuntimeError(
            "Text-only model contract failed: "
            f"model_type={model_type!r}, architectures={architectures!r}."
        )
    if type(base_model).__name__ != TEXT_ARCHITECTURE:
        raise RuntimeError(
            f"Expected {TEXT_ARCHITECTURE}, but Transformers constructed "
            f"{type(base_model).__name__} inside {type(model).__name__}."
        )

    conv3d_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv3d)]
    if conv3d_modules:
        raise RuntimeError(f"Text-only model unexpectedly contains Conv3d modules: {conv3d_modules[:5]}")

    multimodal_markers = (
        "visual",
        "vision_tower",
        "multi_modal_projector",
        "image_tower",
        "video_tower",
    )
    multimodal_parameters = [
        name for name, _ in model.named_parameters() if any(marker in name.lower() for marker in multimodal_markers)
    ]
    if multimodal_parameters:
        raise RuntimeError(
            "Text-only model unexpectedly contains multimodal parameters: "
            f"{multimodal_parameters[:5]}"
        )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    _rank0_print(
        "[qwen35_122b_text_only] contract=OK "
        f"class={type(model).__name__} base_class={type(base_model).__name__} "
        f"finetuning_type={finetuning_type} model_type={model_type} "
        f"conv3d=0 multimodal_params=0 trainable={trainable} total={total}"
    )


def _normalize_text_only_distributed_metadata(model: "PreTrainedModel") -> None:
    """Remove the absent vision block from Transformers' inherited FSDP metadata."""
    decoder_layer_count = sum(
        module.__class__.__name__ == TEXT_FSDP_LAYER_CLASS for module in model.modules()
    )
    expected_layers = EXPECTED_TEXT_FIELDS["num_hidden_layers"]
    if decoder_layer_count != expected_layers:
        raise RuntimeError(
            f"Text-only model contains {decoder_layer_count} "
            f"{TEXT_FSDP_LAYER_CLASS} modules; expected {expected_layers}."
        )

    get_base_model = getattr(model, "get_base_model", None)
    base_model = get_base_model() if callable(get_base_model) else model
    metadata_owners = [
        model,
        base_model,
        getattr(base_model, "model", None),
    ]
    for owner in metadata_owners:
        if owner is not None:
            owner._no_split_modules = [TEXT_FSDP_LAYER_CLASS]

    _rank0_print(
        "[qwen35_122b_text_only] fsdp_wrap="
        f"{TEXT_FSDP_LAYER_CLASS} decoder_layers={decoder_layer_count}; vision block excluded"
    )


def install_text_only_loading() -> None:
    """Install the strict text-only loader before LLaMA-Factory imports workflows."""
    global _INSTALLED
    if _INSTALLED:
        return
    if not _enabled(os.environ.get("FFT_TEXT_ONLY")):
        raise RuntimeError("FFT_TEXT_ONLY=1 is required for the Qwen3.5 TPS benchmark.")

    import llamafactory.model as model_api
    from llamafactory.model import loader
    from transformers import AutoConfig, AutoTokenizer

    original_load_model = loader.load_model

    def load_text_config(model_args: Any) -> Any:
        init_kwargs = loader._get_init_kwargs(model_args)
        source_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        text_config = _extract_text_config(source_config)
        _rank0_print(
            "[qwen35_122b_text_only] source="
            f"{SOURCE_ARCHITECTURE} -> load={TEXT_ARCHITECTURE}; visual and MTP weights excluded"
        )
        return text_config

    def load_text_tokenizer(model_args: Any) -> dict[str, Any]:
        init_kwargs = loader._get_init_kwargs(model_args)
        attempts = [model_args.use_fast_tokenizer, not model_args.use_fast_tokenizer]
        tokenizer = None
        last_error: Exception | None = None
        for use_fast in dict.fromkeys(attempts):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_args.model_name_or_path,
                    use_fast=use_fast,
                    split_special_tokens=model_args.split_special_tokens,
                    padding_side="right",
                    **init_kwargs,
                )
                break
            except Exception as error:  # match LLaMA-Factory's fast/slow fallback
                last_error = error
        if tokenizer is None:
            raise OSError("Failed to load the text tokenizer.") from last_error

        loader.patch_tokenizer(tokenizer, model_args)
        _rank0_print(
            "[qwen35_122b_text_only] tokenizer loaded; AutoProcessor disabled"
        )
        return {"tokenizer": tokenizer, "processor": None}

    def load_text_model(
        tokenizer: Any,
        model_args: Any,
        finetuning_args: Any,
        is_trainable: bool = False,
        add_valuehead: bool = False,
    ) -> "PreTrainedModel":
        model = original_load_model(
            tokenizer,
            model_args,
            finetuning_args,
            is_trainable=is_trainable,
            add_valuehead=add_valuehead,
        )
        assert_text_only_model(model, str(finetuning_args.finetuning_type))
        _normalize_text_only_distributed_metadata(model)
        return model

    loader.load_config = load_text_config
    loader.load_tokenizer = load_text_tokenizer
    loader.load_model = load_text_model
    model_api.load_config = load_text_config
    model_api.load_tokenizer = load_text_tokenizer
    model_api.load_model = load_text_model
    _INSTALLED = True
