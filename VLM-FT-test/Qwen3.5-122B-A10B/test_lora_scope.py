from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_vlm_contract import (  # noqa: E402
    is_visual_lora,
    required_lora_groups,
    validate_patch_embed_conv3d,
)
from validate_adapter_output import validate_adapter  # noqa: E402


TEXT_KEY = (
    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
)
VISION_KEY = "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight"


def write_adapter(output_dir: Path, keys: tuple[str, ...]) -> None:
    output_dir.mkdir()
    (output_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}), encoding="utf-8"
    )
    save_file(
        {key: torch.ones(2, 2) for key in keys},
        output_dir / "adapter_model.safetensors",
    )


@pytest.mark.parametrize(
    ("scope", "keys", "text_count", "vision_count"),
    (
        ("text", (TEXT_KEY,), 1, 0),
        ("vision", (VISION_KEY,), 0, 1),
        ("all", (TEXT_KEY, VISION_KEY), 1, 1),
    ),
)
def test_validate_adapter_accepts_exact_scope(
    tmp_path, scope, keys, text_count, vision_count
):
    output_dir = tmp_path / scope
    write_adapter(output_dir, keys)

    summary = validate_adapter(output_dir, scope)

    assert summary["scope"] == scope
    assert summary["text_lora"] == text_count
    assert summary["visual_lora"] == vision_count


@pytest.mark.parametrize(
    ("scope", "keys"),
    (
        ("text", (VISION_KEY,)),
        ("vision", (TEXT_KEY,)),
        ("all", (TEXT_KEY,)),
    ),
)
def test_validate_adapter_rejects_scope_mismatch(tmp_path, scope, keys):
    output_dir = tmp_path / scope
    write_adapter(output_dir, keys)

    with pytest.raises(RuntimeError, match="scope"):
        validate_adapter(output_dir, scope)


def test_scope_helpers_distinguish_modalities():
    assert is_visual_lora(TEXT_KEY) is False
    assert is_visual_lora(VISION_KEY) is True
    assert required_lora_groups("text") == ("text",)
    assert required_lora_groups("vision") == ("vision",)
    assert required_lora_groups("all") == ("text", "vision")


def make_visual(*, wrapped: bool) -> torch.nn.Module:
    visual = torch.nn.Module()
    visual.patch_embed = torch.nn.Module()
    if not wrapped:
        visual.patch_embed.proj = torch.nn.Conv3d(
            3, 4, kernel_size=2, stride=2
        )
        return visual

    proj = torch.nn.Module()
    proj.base_layer = torch.nn.Conv3d(3, 4, kernel_size=2, stride=2)
    proj.lora_A = torch.nn.ModuleDict(
        {"default": torch.nn.Conv3d(3, 2, kernel_size=2, stride=2, bias=False)}
    )
    proj.lora_B = torch.nn.ModuleDict(
        {"default": torch.nn.Conv3d(2, 4, kernel_size=1, bias=False)}
    )
    visual.patch_embed.proj = proj
    return visual


def test_text_scope_accepts_unwrapped_patch_conv3d():
    conv3d = validate_patch_embed_conv3d(make_visual(wrapped=False), "text")

    assert conv3d == ["patch_embed.proj"]


@pytest.mark.parametrize("scope", ("vision", "all"))
def test_visual_scope_accepts_peft_wrapped_patch_conv3d(scope):
    conv3d = validate_patch_embed_conv3d(make_visual(wrapped=True), scope)

    assert set(conv3d) == {
        "patch_embed.proj.base_layer",
        "patch_embed.proj.lora_A.default",
        "patch_embed.proj.lora_B.default",
    }


def test_patch_conv3d_contract_rejects_wrong_scope_structure():
    with pytest.raises(RuntimeError, match="unwrapped"):
        validate_patch_embed_conv3d(make_visual(wrapped=True), "text")
    with pytest.raises(RuntimeError, match="PEFT-wrapped"):
        validate_patch_embed_conv3d(make_visual(wrapped=False), "all")
