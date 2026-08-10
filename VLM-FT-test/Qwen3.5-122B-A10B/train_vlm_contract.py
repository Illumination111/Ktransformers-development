#!/usr/bin/env python3
"""LLaMA-Factory entrypoint that asserts the VLM/KT/frozen-tower contract."""

from __future__ import annotations

import os
from collections import deque
from typing import Any


def iter_modules(root: Any):
    queue = deque([root])
    seen: set[int] = set()
    while queue:
        current = queue.popleft()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("model", "base_model", "pretrained_model", "module"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                queue.append(child)


def assert_vlm_contract(model: Any) -> None:
    import torch

    if os.getenv("FFT_TEXT_ONLY"):
        raise RuntimeError("FFT_TEXT_ONLY must not be set for the VLM test")
    nodes = list(iter_modules(model))
    conditional = next(
        (node for node in nodes if type(node).__name__ == "Qwen3_5MoeForConditionalGeneration"), None
    )
    if conditional is None:
        raise RuntimeError("full Qwen3_5MoeForConditionalGeneration was not constructed")
    if getattr(conditional.config, "model_type", None) != "qwen3_5_moe":
        raise RuntimeError(f"unexpected model type: {getattr(conditional.config, 'model_type', None)!r}")
    visual = getattr(getattr(conditional, "model", None), "visual", None)
    language = getattr(getattr(conditional, "model", None), "language_model", None)
    if visual is None or language is None:
        raise RuntimeError("full VLM must retain both model.visual and model.language_model")
    conv3d = [name for name, module in visual.named_modules() if isinstance(module, torch.nn.Conv3d)]
    if conv3d != ["patch_embed.proj"]:
        raise RuntimeError(f"unexpected vision Conv3D modules: {conv3d}")
    visual_trainable = [name for name, param in visual.named_parameters() if param.requires_grad]
    if visual_trainable:
        raise RuntimeError(f"vision tower is not frozen: {visual_trainable[:5]}")
    lora_params = [name for name, param in model.named_parameters() if "lora_" in name and param.requires_grad]
    if not lora_params:
        raise RuntimeError("no trainable LoRA parameters were created")
    bad_lora = [name for name in lora_params if ".visual." in name or name.startswith("visual.")]
    if bad_lora:
        raise RuntimeError(f"LoRA leaked into the frozen vision tower: {bad_lora[:5]}")
    wrapper_owners = [node for node in nodes if getattr(node, "_kt_wrappers", None)]
    wrapper_count = max((len(node._kt_wrappers) for node in wrapper_owners), default=0)
    if wrapper_count != 48:
        raise RuntimeError(f"expected 48 KT-wrapped MoE layers, got {wrapper_count}")
    print(
        "[qwen35_vlm_contract] OK "
        f"class={type(conditional).__name__} conv3d={conv3d} visual_trainable=0 "
        f"trainable_lora={len(lora_params)} kt_wrappers={wrapper_count}",
        flush=True,
    )


def install_contract() -> None:
    import llamafactory.model as model_api
    from llamafactory.model import loader

    original = loader.load_model

    def checked_load_model(*args, **kwargs):
        model = original(*args, **kwargs)
        assert_vlm_contract(model)
        return model

    loader.load_model = checked_load_model
    model_api.load_model = checked_load_model


def main() -> None:
    install_contract()
    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
