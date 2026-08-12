#!/usr/bin/env python3
"""LLaMA-Factory entrypoint asserting the Qwen3-VL-MoE LoRA/KT contract."""

from __future__ import annotations

import os
from collections import deque
from typing import Any

from transformers import TrainerCallback

from load_conv3d_compat import load_conv3d_compat
from load_kt_arch_compat import activate_kt_architecture_shim


VALID_LORA_SCOPES = ("text", "vision", "all")
EXPECTED_CLASS = "Qwen3VLMoeForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_vl_moe"


def get_lora_scope() -> str:
    scope = os.getenv("VLM_LORA_SCOPE", "text").lower()
    if scope not in VALID_LORA_SCOPES:
        raise RuntimeError(f"VLM_LORA_SCOPE must be one of {VALID_LORA_SCOPES}, got {scope!r}")
    return scope


def is_visual_lora(name: str) -> bool:
    return ".visual." in name or name.startswith("visual.")


def required_lora_groups(scope: str) -> tuple[str, ...]:
    return ("text", "vision") if scope == "all" else (scope,)


def to_local_tensor(tensor: Any) -> Any:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        pass
    else:
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
    wait = getattr(tensor, "wait", None)
    return wait() if callable(wait) else tensor


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


def find_conditional(model: Any) -> Any:
    conditional = next((node for node in iter_modules(model) if type(node).__name__ == EXPECTED_CLASS), None)
    if conditional is None:
        raise RuntimeError(f"full {EXPECTED_CLASS} was not constructed")
    return conditional


def validate_patch_embed_conv3d(visual: Any, scope: str) -> list[str]:
    import torch

    proj = getattr(getattr(visual, "patch_embed", None), "proj", None)
    conv3d = [name for name, module in visual.named_modules() if isinstance(module, torch.nn.Conv3d)]
    if scope == "text":
        if not isinstance(proj, torch.nn.Conv3d) or conv3d != ["patch_embed.proj"]:
            raise RuntimeError(f"text scope requires one unwrapped patch Conv3D, got {conv3d}")
        return conv3d

    base_layer = getattr(proj, "base_layer", None)
    lora_a = getattr(proj, "lora_A", None)
    lora_b = getattr(proj, "lora_B", None)
    if not isinstance(base_layer, torch.nn.Conv3d):
        raise RuntimeError(f"{scope} scope requires a PEFT-wrapped patch Conv3D")
    if not isinstance(lora_a, torch.nn.ModuleDict) or not isinstance(lora_b, torch.nn.ModuleDict):
        raise RuntimeError(f"{scope} scope requires Conv3D LoRA A/B dictionaries")
    adapters = set(lora_a) & set(lora_b)
    if not adapters or set(lora_a) != set(lora_b):
        raise RuntimeError("vision patch Conv3D LoRA adapter mismatch")
    if any(not isinstance(lora_a[name], torch.nn.Conv3d) or not isinstance(lora_b[name], torch.nn.Conv3d) for name in adapters):
        raise RuntimeError("vision patch LoRA A/B modules must be Conv3D")
    return conv3d


def assert_vlm_contract(model: Any) -> Any:
    import torch

    scope = get_lora_scope()
    conditional = find_conditional(model)
    if getattr(conditional.config, "model_type", None) != EXPECTED_MODEL_TYPE:
        raise RuntimeError(f"unexpected model type: {getattr(conditional.config, 'model_type', None)!r}")
    visual = getattr(getattr(conditional, "model", None), "visual", None)
    language = getattr(getattr(conditional, "model", None), "language_model", None)
    if visual is None or language is None:
        raise RuntimeError("full VLM must retain model.visual and model.language_model")
    conv3d = validate_patch_embed_conv3d(visual, scope)

    compat = load_conv3d_compat()
    torch_29 = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]) == (2, 9)
    active = compat.is_swift_conv3d_patch_active()
    compat.validate_swift_conv3d_modules(visual)
    if torch_29 and not active:
        raise RuntimeError("LLaMA-Factory did not activate KT/ms-swift Conv3D compatibility")

    visual_base_trainable = [name for name, param in visual.named_parameters() if param.requires_grad and "lora_" not in name]
    if visual_base_trainable:
        raise RuntimeError(f"visual base parameters are trainable: {visual_base_trainable[:5]}")
    lora = [(name, param) for name, param in model.named_parameters() if param.requires_grad and "lora_" in name]
    visual_lora = [name for name, _ in lora if is_visual_lora(name)]
    text_lora = [name for name, _ in lora if not is_visual_lora(name)]
    if scope == "text" and (not text_lora or visual_lora):
        raise RuntimeError(f"text LoRA scope mismatch: text={len(text_lora)}, vision={len(visual_lora)}")
    if scope == "vision" and (not visual_lora or text_lora):
        raise RuntimeError(f"vision LoRA scope mismatch: text={len(text_lora)}, vision={len(visual_lora)}")
    if scope == "all" and (not text_lora or not visual_lora):
        raise RuntimeError(f"all LoRA scope requires both modalities: text={len(text_lora)}, vision={len(visual_lora)}")

    wrapper_count = max((len(node._kt_wrappers) for node in iter_modules(model) if getattr(node, "_kt_wrappers", None)), default=0)
    expected = int(os.getenv("VLM_EXPECTED_KT_WRAPPERS", "48"))
    if wrapper_count != expected:
        raise RuntimeError(f"expected {expected} KT-wrapped MoE layers, got {wrapper_count}")

    conditional._qwen3vl_vision_forward_count = 0

    def record_vision_forward(_module, _inputs, _output):
        conditional._qwen3vl_vision_forward_count += 1

    visual.patch_embed.proj.register_forward_hook(record_vision_forward)
    print(
        "[qwen3vl_contract] OK "
        f"class={type(conditional).__name__} scope={scope} conv3d={conv3d} "
        f"text_lora={len(text_lora)} visual_lora={len(visual_lora)} "
        f"visual_base_trainable=0 kt_wrappers={wrapper_count}",
        flush=True,
    )
    return conditional


class VLMFunctionalCallback(TrainerCallback):
    """Require a real image, non-zero LoRA gradients, and optimizer updates."""

    def __init__(self) -> None:
        self.scope = get_lora_scope()
        self.last_vision_forward_count = 0
        self.snapshots: dict[str, tuple[str, Any]] = {}
        self.gradient_checked = False
        self.optimizer_steps = 0

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        import torch

        model = kwargs["model"]
        conditional = find_conditional(model)
        count = getattr(conditional, "_qwen3vl_vision_forward_count", 0)
        if count <= self.last_vision_forward_count:
            raise RuntimeError("optimizer step received no new image at model.visual.patch_embed.proj")
        self.last_vision_forward_count = count

        sampled: dict[str, tuple[str, Any]] = {}
        inspected = 0
        for name, parameter in model.named_parameters():
            if "lora_" not in name or not parameter.requires_grad or parameter.grad is None:
                continue
            inspected += 1
            grad = to_local_tensor(parameter.grad).detach()
            if grad.numel() == 0:
                continue
            if not torch.isfinite(grad).all().item():
                raise RuntimeError(f"non-finite LoRA gradient: {name}")
            if grad.ne(0).any().item():
                sampled.setdefault("vision" if is_visual_lora(name) else "text", (name, parameter))
        missing = [group for group in required_lora_groups(self.scope) if group not in sampled]
        if missing:
            raise RuntimeError(f"LoRA groups received no non-zero gradient: {missing}")
        visual_base_grads = [name for name, parameter in conditional.model.visual.named_parameters() if "lora_" not in name and parameter.grad is not None]
        if visual_base_grads:
            raise RuntimeError(f"visual base parameters received gradients: {visual_base_grads[:5]}")
        self.snapshots = {
            group: (name, to_local_tensor(parameter).detach().float().cpu().clone())
            for group, (name, parameter) in sampled.items()
            if group in required_lora_groups(self.scope)
        }
        self.gradient_checked = True
        print(f"[qwen3vl_functional] GRADIENT_OK scope={self.scope} inspected={inspected}", flush=True)

    def on_optimizer_step(self, args, state, control, **kwargs):
        import torch

        parameters = dict(kwargs["model"].named_parameters())
        updates = []
        for group, (name, before) in self.snapshots.items():
            current = parameters.get(name)
            if current is None:
                raise RuntimeError(f"sampled LoRA parameter disappeared: {name}")
            after = to_local_tensor(current).detach().float().cpu()
            delta = float((after - before).abs().max().item())
            if not torch.isfinite(torch.tensor(delta)) or delta == 0.0:
                raise RuntimeError(f"LoRA optimizer step did not update {name}: {delta}")
            updates.append(f"{group}:{name}:{delta:.6e}")
        self.optimizer_steps += 1
        print(f"[qwen3vl_functional] OPTIMIZER_OK updates={','.join(updates)}", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if not self.gradient_checked or self.optimizer_steps < 1 or state.global_step < 1:
            raise RuntimeError("incomplete Qwen3-VL functional validation")
        print(f"[qwen3vl_functional] PASS optimizer_steps={self.optimizer_steps}", flush=True)


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
    arch_source = activate_kt_architecture_shim()
    load_conv3d_compat(register_as_kt_module=True)
    print(f"[qwen3vl_kt_arch_shim] source={arch_source}", flush=True)
    install_contract()
    from llamafactory.train.tuner import run_exp

    run_exp(callbacks=[VLMFunctionalCallback()])


if __name__ == "__main__":
    main()
