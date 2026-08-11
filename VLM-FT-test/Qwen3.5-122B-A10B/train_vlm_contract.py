#!/usr/bin/env python3
"""LLaMA-Factory entrypoint that asserts the scoped VLM LoRA/KT contract."""

from __future__ import annotations

import os
from collections import deque
from typing import Any

from transformers import TrainerCallback

from load_conv3d_compat import load_conv3d_compat


VALID_LORA_SCOPES = ("text", "vision", "all")


def get_lora_scope() -> str:
    scope = os.getenv("VLM_LORA_SCOPE", "text").lower()
    if scope not in VALID_LORA_SCOPES:
        raise RuntimeError(
            f"VLM_LORA_SCOPE must be one of {VALID_LORA_SCOPES}, got {scope!r}"
        )
    return scope


def is_visual_lora(name: str) -> bool:
    return ".visual." in name or name.startswith("visual.")


def required_lora_groups(scope: str) -> tuple[str, ...]:
    return ("text", "vision") if scope == "all" else (scope,)


def to_local_tensor(tensor: Any) -> Any:
    """Return a regular tensor or the current-rank shard of an FSDP2 DTensor."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        pass
    else:
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
    wait = getattr(tensor, "wait", None)
    if callable(wait):
        tensor = wait()
    return tensor


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


def validate_patch_embed_conv3d(visual: Any, scope: str) -> list[str]:
    """Validate the bare or PEFT-wrapped vision patch Conv3D for a LoRA scope."""
    import torch

    proj = getattr(getattr(visual, "patch_embed", None), "proj", None)
    conv3d = [
        name
        for name, module in visual.named_modules()
        if isinstance(module, torch.nn.Conv3d)
    ]
    if scope == "text":
        if not isinstance(proj, torch.nn.Conv3d) or conv3d != ["patch_embed.proj"]:
            raise RuntimeError(
                f"text scope requires one unwrapped vision patch Conv3D, got {conv3d}"
            )
        return conv3d

    if scope not in ("vision", "all"):
        raise RuntimeError(f"cannot validate vision patch Conv3D for scope {scope!r}")

    base_layer = getattr(proj, "base_layer", None)
    lora_a = getattr(proj, "lora_A", None)
    lora_b = getattr(proj, "lora_B", None)
    if not isinstance(base_layer, torch.nn.Conv3d):
        raise RuntimeError(
            f"{scope} scope requires a PEFT-wrapped vision patch Conv3D base layer"
        )
    if not isinstance(lora_a, torch.nn.ModuleDict) or not isinstance(
        lora_b, torch.nn.ModuleDict
    ):
        raise RuntimeError(
            f"{scope} scope requires Conv3D LoRA A/B module dictionaries"
        )

    adapter_names = set(lora_a) & set(lora_b)
    if not adapter_names or set(lora_a) != set(lora_b):
        raise RuntimeError(
            f"vision patch Conv3D LoRA adapter mismatch: "
            f"lora_A={sorted(lora_a)}, lora_B={sorted(lora_b)}"
        )
    if any(
        not isinstance(lora_a[name], torch.nn.Conv3d)
        or not isinstance(lora_b[name], torch.nn.Conv3d)
        for name in adapter_names
    ):
        raise RuntimeError("vision patch LoRA A/B modules must all be Conv3D")

    expected = {"patch_embed.proj.base_layer"}
    expected.update(
        f"patch_embed.proj.lora_A.{name}" for name in adapter_names
    )
    expected.update(
        f"patch_embed.proj.lora_B.{name}" for name in adapter_names
    )
    if set(conv3d) != expected:
        raise RuntimeError(
            f"unexpected PEFT-wrapped vision Conv3D modules: {conv3d}; "
            f"expected={sorted(expected)}"
        )
    return conv3d


def assert_vlm_contract(model: Any) -> Any:
    import torch

    scope = get_lora_scope()
    if os.getenv("FFT_TEXT_ONLY"):
        raise RuntimeError("FFT_TEXT_ONLY must not be set for the VLM test")
    nodes = list(iter_modules(model))
    conditional = next(
        (
            node
            for node in nodes
            if type(node).__name__ == "Qwen3_5MoeForConditionalGeneration"
        ),
        None,
    )
    if conditional is None:
        raise RuntimeError(
            "full Qwen3_5MoeForConditionalGeneration was not constructed"
        )
    if getattr(conditional.config, "model_type", None) != "qwen3_5_moe":
        raise RuntimeError(
            f"unexpected model type: {getattr(conditional.config, 'model_type', None)!r}"
        )
    visual = getattr(getattr(conditional, "model", None), "visual", None)
    language = getattr(getattr(conditional, "model", None), "language_model", None)
    if visual is None or language is None:
        raise RuntimeError(
            "full VLM must retain both model.visual and model.language_model"
        )
    conv3d = validate_patch_embed_conv3d(visual, scope)
    compatibility_api = load_conv3d_compat()
    torch_version = torch.__version__
    torch_base_version = tuple(
        int(part) for part in torch_version.split("+", 1)[0].split(".")[:2]
    )
    compatibility_required = torch_base_version == (2, 9)
    compatibility_active = compatibility_api.is_swift_conv3d_patch_active()
    compatibility_api.validate_swift_conv3d_modules(visual)
    if compatibility_required and not compatibility_active:
        raise RuntimeError(
            "LLaMA-Factory did not automatically activate the KT/ms-swift "
            "Conv3D compatibility layer in this training rank"
        )
    visual_trainable = [
        name for name, param in visual.named_parameters() if param.requires_grad
    ]
    visual_base_trainable = [name for name in visual_trainable if "lora_" not in name]
    if visual_base_trainable:
        raise RuntimeError(
            f"visual base parameters are trainable: {visual_base_trainable[:5]}"
        )
    lora_params = [
        name
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    ]
    if not lora_params:
        raise RuntimeError("no trainable LoRA parameters were created")
    visual_lora = [name for name in lora_params if is_visual_lora(name)]
    text_lora = [name for name in lora_params if not is_visual_lora(name)]
    if scope == "text" and (not text_lora or visual_lora):
        raise RuntimeError(
            f"text scope mismatch: text_lora={len(text_lora)}, visual_lora={len(visual_lora)}"
        )
    if scope == "vision" and (not visual_lora or text_lora):
        raise RuntimeError(
            f"vision scope mismatch: text_lora={len(text_lora)}, visual_lora={len(visual_lora)}"
        )
    if scope == "all" and (not text_lora or not visual_lora):
        raise RuntimeError(
            f"all scope requires both modalities: text={len(text_lora)}, vision={len(visual_lora)}"
        )
    wrapper_owners = [node for node in nodes if getattr(node, "_kt_wrappers", None)]
    wrapper_count = max((len(node._kt_wrappers) for node in wrapper_owners), default=0)
    expected_wrappers = int(os.getenv("VLM_EXPECTED_KT_WRAPPERS", "48"))
    if wrapper_count != expected_wrappers:
        raise RuntimeError(
            f"expected {expected_wrappers} KT-wrapped MoE layers, got {wrapper_count}"
        )
    conditional._qwen35_vlm_vision_forward_count = 0

    def record_vision_forward(_module, _inputs, _output):
        conditional._qwen35_vlm_vision_forward_count += 1

    getattr(visual.patch_embed, "proj").register_forward_hook(record_vision_forward)
    print(
        "[qwen35_vlm_contract] OK "
        f"class={type(conditional).__name__} scope={scope} conv3d={conv3d} "
        f"text_lora={len(text_lora)} visual_lora={len(visual_lora)} "
        f"visual_base_trainable=0 kt_wrappers={wrapper_count} "
        f"swift_conv3d_patch={'active' if compatibility_required else 'not_required'}",
        flush=True,
    )
    print(
        "[qwen35_vlm_conv3d] "
        f"required={compatibility_required} active={compatibility_active} "
        f"torch={torch_version}",
        flush=True,
    )
    return conditional


class VLMFunctionalCallback(TrainerCallback):
    """Fail unless a real image batch drives LoRA gradient and optimizer paths."""

    def __init__(self) -> None:
        self.lora_scope = get_lora_scope()
        self.gradient_checked = False
        self.optimizer_steps = 0
        self.last_vision_forward_count = 0
        self.parameter_snapshots: dict[str, tuple[str, Any]] = {}

    @staticmethod
    def _named_parameters(model: Any) -> dict[str, Any]:
        return dict(model.named_parameters())

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        import torch

        model = kwargs["model"]
        nodes = list(iter_modules(model))
        conditional = next(
            (
                node
                for node in nodes
                if type(node).__name__ == "Qwen3_5MoeForConditionalGeneration"
            ),
            None,
        )
        if conditional is None:
            raise RuntimeError(
                "functional callback cannot find the Qwen3.5 conditional-generation model"
            )
        vision_forward_count = getattr(
            conditional, "_qwen35_vlm_vision_forward_count", 0
        )
        if vision_forward_count <= self.last_vision_forward_count:
            raise RuntimeError(
                "this optimizer step received no new image batch at "
                "model.visual.patch_embed.proj"
            )
        self.last_vision_forward_count = vision_forward_count

        params = self._named_parameters(model)
        sampled_lora: dict[str, tuple[str, Any]] = {}
        inspected_lora_grads = 0
        for name, parameter in params.items():
            if (
                "lora_" not in name
                or not parameter.requires_grad
                or parameter.grad is None
            ):
                continue
            inspected_lora_grads += 1
            local_grad = to_local_tensor(parameter.grad).detach()
            if local_grad.numel() == 0:
                continue
            if not torch.isfinite(local_grad).all().item():
                raise RuntimeError(f"non-finite LoRA gradient: {name}")
            if local_grad.ne(0).any().item():
                group = "vision" if is_visual_lora(name) else "text"
                sampled_lora.setdefault(group, (name, parameter))
        missing_groups = [
            group
            for group in required_lora_groups(self.lora_scope)
            if group not in sampled_lora
        ]
        if missing_groups:
            raise RuntimeError(
                f"LoRA groups received no non-zero gradient: {missing_groups}"
            )

        visual = conditional.model.visual
        visual_base_grads = [
            name
            for name, parameter in visual.named_parameters()
            if "lora_" not in name and parameter.grad is not None
        ]
        if visual_base_grads:
            raise RuntimeError(
                f"visual base parameters received gradients: {visual_base_grads[:5]}"
            )

        self.parameter_snapshots = {
            group: (name, to_local_tensor(parameter).detach().float().cpu().clone())
            for group, (name, parameter) in sampled_lora.items()
            if group in required_lora_groups(self.lora_scope)
        }
        self.gradient_checked = True
        samples = ",".join(
            f"{group}:{name}" for group, (name, _) in self.parameter_snapshots.items()
        )
        print(
            "[qwen35_vlm_functional] GRADIENT_OK "
            f"scope={self.lora_scope} "
            f"vision_forwards={vision_forward_count} "
            f"inspected_lora_grads={inspected_lora_grads} samples={samples}",
            flush=True,
        )

    def on_optimizer_step(self, args, state, control, **kwargs):
        import torch

        self.optimizer_steps += 1
        if not self.parameter_snapshots:
            raise RuntimeError(
                "optimizer step occurred before the LoRA gradient contract was checked"
            )
        parameters = self._named_parameters(kwargs["model"])
        deltas = []
        for group, (name, before) in self.parameter_snapshots.items():
            current = parameters.get(name)
            if current is None:
                raise RuntimeError(
                    f"sampled {group} LoRA parameter disappeared before optimizer step: {name}"
                )
            after = to_local_tensor(current).detach().float().cpu()
            if after.shape != before.shape:
                raise RuntimeError(
                    f"LoRA local shard changed shape for {name}: "
                    f"before={tuple(before.shape)}, after={tuple(after.shape)}"
                )
            if after.numel() == 0:
                raise RuntimeError(
                    f"sampled LoRA parameter has an empty local shard: {name}"
                )
            delta = float((after - before).abs().max().item())
            if not torch.isfinite(torch.tensor(delta)) or delta == 0.0:
                raise RuntimeError(
                    f"LoRA optimizer step did not update {name}: max_abs_delta={delta}"
                )
            deltas.append(f"{group}:{name}:{delta:.6e}")
        print(
            f"[qwen35_vlm_functional] OPTIMIZER_OK scope={self.lora_scope} updates={','.join(deltas)}",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        if (
            not self.gradient_checked
            or self.optimizer_steps < 1
            or state.global_step < 1
        ):
            raise RuntimeError(
                "incomplete VLM functional validation: "
                f"gradient_checked={self.gradient_checked}, optimizer_steps={self.optimizer_steps}, "
                f"global_step={state.global_step}"
            )
        print(
            "[qwen35_vlm_functional] PASS "
            f"optimizer_steps={self.optimizer_steps} global_step={state.global_step}",
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
    # Development shim only: make the new additive KT API visible while the
    # environment keeps its released kt-kernel binary. LLaMA-Factory remains
    # responsible for detecting torch/VLM/Conv3D and activating ms-swift.
    load_conv3d_compat(register_as_kt_module=True)
    install_contract()
    from llamafactory.train.tuner import run_exp

    run_exp(callbacks=[VLMFunctionalCallback()])


if __name__ == "__main__":
    main()
