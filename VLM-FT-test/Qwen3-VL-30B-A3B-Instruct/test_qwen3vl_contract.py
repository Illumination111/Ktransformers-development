from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_kt_arch_compat import audit_kt_architecture  # noqa: E402
from train_vlm_contract import (  # noqa: E402
    is_visual_lora,
    required_lora_groups,
    validate_patch_embed_conv3d,
)


def qwen3vl_config():
    return SimpleNamespace(
        architectures=["Qwen3VLMoeForConditionalGeneration"],
        text_config=SimpleNamespace(
            num_experts=128,
            moe_intermediate_size=768,
            num_experts_per_tok=8,
        ),
    )


def test_development_kt_arch_supports_qwen3vl_moe():
    summary = audit_kt_architecture(qwen3vl_config())
    assert summary["source_supported"] is True
    assert summary["source_prefix"] == "model.language_model.layers"
    assert summary["source_experts"] == 128
    assert summary["source_moe_intermediate_size"] == 768
    assert summary["source_top_k"] == 8


def test_scope_helpers():
    assert is_visual_lora("base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight")
    assert not is_visual_lora("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight")
    assert required_lora_groups("text") == ("text",)
    assert required_lora_groups("vision") == ("vision",)
    assert required_lora_groups("all") == ("text", "vision")


def make_visual(wrapped: bool):
    visual = torch.nn.Module()
    visual.patch_embed = torch.nn.Module()
    if not wrapped:
        visual.patch_embed.proj = torch.nn.Conv3d(3, 4, kernel_size=2)
        return visual
    proj = torch.nn.Module()
    proj.base_layer = torch.nn.Conv3d(3, 4, kernel_size=2)
    proj.lora_A = torch.nn.ModuleDict({"default": torch.nn.Conv3d(3, 2, kernel_size=2, bias=False)})
    proj.lora_B = torch.nn.ModuleDict({"default": torch.nn.Conv3d(2, 4, kernel_size=1, bias=False)})
    visual.patch_embed.proj = proj
    return visual


def test_patch_conv3d_scope_contract():
    assert validate_patch_embed_conv3d(make_visual(False), "text") == ["patch_embed.proj"]
    assert "patch_embed.proj.base_layer" in validate_patch_embed_conv3d(make_visual(True), "all")
    with pytest.raises(RuntimeError, match="unwrapped"):
        validate_patch_embed_conv3d(make_visual(True), "text")


def test_llamafactory_discovers_qwen3vl_text_and_vision_lora_modules():
    from transformers import (
        Qwen3VLMoeConfig,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3VLMoeTextConfig,
        Qwen3VLMoeVisionConfig,
    )

    from llamafactory.model.model_utils.vlm_lora import find_vlm_lora_modules

    text = Qwen3VLMoeTextConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
    )
    vision = Qwen3VLMoeVisionConfig(
        depth=3,
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        out_hidden_size=8,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=1,
        num_position_embeddings=4,
        deepstack_visual_indexes=[0, 1, 2],
    )
    config = Qwen3VLMoeConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=61,
        video_token_id=62,
        vision_start_token_id=59,
        vision_end_token_id=60,
    )
    model = Qwen3VLMoeForConditionalGeneration(config)

    text_targets = find_vlm_lora_modules(model, "text")
    vision_targets = find_vlm_lora_modules(model, "vision")
    all_targets = find_vlm_lora_modules(model, "all")

    assert text_targets and all("language_model" in name for name in text_targets)
    assert vision_targets and all("visual" in name for name in vision_targets)
    assert "model.visual.patch_embed.proj" in vision_targets
    assert set(all_targets) == set(text_targets) | set(vision_targets)
