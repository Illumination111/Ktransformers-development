"""Enable owner-sharded Torch MoE full-FT without changing the xmy venv.

Vendor ``torch_moe`` stores expert weights as plain tensors and only trains
LoRA.  This module keeps those files untouched and, inside the FFTtest process:

* turns owner-local expert weights into optimizer Parameters
* replays backward through those weights
* injects them after FSDP2 prepare
* skips the LoRA-only ``kt_adapt_peft_lora`` path
"""

from __future__ import annotations

import os
from typing import Any


_INSTALLED = False
_EXPERT_ATTRS = (
    "expert_gate_up_proj",
    "expert_gate_proj",
    "expert_up_proj",
    "expert_down_proj",
)


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def collect_expert_weight_parameters(model: Any) -> list[Any]:
    parameters: list[Any] = []
    seen: set[int] = set()
    wrappers = _iter_kt_wrappers(model)
    for wrapper in wrappers:
        for attr in _EXPERT_ATTRS:
            value = getattr(wrapper, attr, None)
            if value is None or not getattr(value, "requires_grad", False):
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            parameters.append(value)
    return parameters


def enable_full_weight_grad(model: Any) -> dict[str, int]:
    import torch.distributed as dist
    import torch.nn as nn

    wrappers = list(getattr(model, "_kt_wrappers", None) or [])
    if not wrappers:
        raise RuntimeError("pytorch_torch full-FT requires TorchMoE wrappers on the model")

    current = dist.get_rank() if dist.is_initialized() else 0
    owned_layers = 0
    expert_numel = 0
    for wrapper in wrappers:
        wrapper._full_weight_grad = True
        wrapper._skip_lora = True
        wrapper._has_expert_lora = False
        wrapper._peft_lora_modules = {}
        if int(getattr(wrapper, "owner_rank", -1)) != current:
            continue
        if not bool(getattr(wrapper, "_has_expert_weights", False)):
            raise RuntimeError(
                f"rank {current} owns layer {getattr(wrapper, 'layer_idx', None)} "
                "but has no CPU expert weights"
            )
        for attr in _EXPERT_ATTRS:
            value = getattr(wrapper, attr, None)
            if value is None:
                continue
            # Keep these off Module.named_parameters().  Vendor torch_moe
            # stores expert weights as plain tensors so FSDP2 will not move
            # them.  A normal setattr(nn.Parameter) would register rank-local
            # keys, make state_dict() diverge, and deadlock
            # fsdp2_load_full_state_dict's rank-0 broadcasts.
            if not isinstance(value, nn.Parameter):
                value = nn.Parameter(value.detach().contiguous())
            value.requires_grad_(True)
            object.__setattr__(wrapper, attr, value)
            expert_numel += int(value.numel())
        owned_layers += 1

    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if ".experts." in f".{lowered}." or lowered.endswith("experts.gate_up_proj") or lowered.endswith("experts.down_proj"):
            if "shared_expert" in lowered:
                continue
            parameter.requires_grad_(False)

    leaked = [
        name
        for name, _ in model.named_parameters()
        if any(attr in name for attr in _EXPERT_ATTRS)
    ]
    if leaked:
        raise RuntimeError(
            "owner expert weights leaked into named_parameters(); "
            f"FSDP2 would deadlock: {leaked[:8]}"
        )

    model._kt_full_weight_grad = True
    model._kt_train_mode = "full"
    model._kt_owned_expert_numel = expert_numel
    print(
        f"[pytorch_torch] rank={current} owned_moe_layers={owned_layers} "
        f"owned_expert_numel={expert_numel}",
        flush=True,
    )
    return {"owned_layers": owned_layers, "owned_expert_numel": expert_numel}


def _iter_kt_wrappers(model: Any) -> list[Any]:
    wrappers = list(getattr(model, "_kt_wrappers", None) or [])
    if wrappers:
        return wrappers
    base = model
    for attr in ("base_model", "model", "module"):
        base = getattr(base, attr, None)
        if base is None:
            break
        wrappers = list(getattr(base, "_kt_wrappers", None) or [])
        if wrappers:
            return wrappers
    return []


def _sync_router_buffers(model: Any) -> int:
    """Move GLM router buffers off CPU/meta after FSDP2 prepare.

    ``e_score_correction_bias`` is a persistent buffer. FSDP2 only broadcasts
    DTensors, so rank 0 keeps the CPU copy and other ranks keep meta. The
    bound ``route_tokens_to_experts`` then adds that buffer to CUDA logits.
    """
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        return 0
    device = torch.device("cuda", torch.cuda.current_device())
    moved = 0
    for wrapper in _iter_kt_wrappers(model):
        router_attr = getattr(wrapper, "_router_attr", "gate")
        router = getattr(wrapper, router_attr, None)
        if router is None:
            continue
        route_fn = getattr(wrapper, "_route_tokens_fn", None)
        owner = getattr(route_fn, "__self__", None)
        if owner is not None and getattr(owner, router_attr, None) is not router:
            setattr(owner, router_attr, router)
        for name, buffer in list(router.named_buffers(recurse=True)):
            if buffer is None or getattr(buffer, "device_mesh", None) is not None:
                continue
            if buffer.device == device and buffer.device.type != "meta":
                continue
            tensor = torch.empty(tuple(buffer.shape), dtype=buffer.dtype, device=device)
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    tensor.copy_(buffer.detach().to(device=device, dtype=buffer.dtype))
                dist.broadcast(tensor, src=0)
            else:
                tensor.copy_(buffer.detach().to(device=device, dtype=buffer.dtype))
            if "." in name:
                parent_name, local_name = name.rsplit(".", 1)
                parent = router.get_submodule(parent_name)
            else:
                parent, local_name = router, name
            parent.register_buffer(local_name, tensor, persistent=True)
            moved += 1
    return moved


def _inject_optimizer_params(optimizer: Any, model: Any) -> int:
    extra = collect_expert_weight_parameters(model)
    if optimizer is None or not extra:
        return 0
    existing = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    new_params = [parameter for parameter in extra if id(parameter) not in existing]
    if not new_params:
        return 0
    optimizer.param_groups[-1]["params"].extend(new_params)
    return len(new_params)


def _patch_compute_routing() -> None:
    """GLM ``route_tokens_to_experts`` expects 2D [tokens, experts] logits."""
    from accelerate.utils.torch_moe import TorchMoELayerWrapper

    original = TorchMoELayerWrapper._compute_routing

    def _compute_routing(self, hidden_states):  # type: ignore[no-untyped-def]
        if self.router_type == "deepseek_gate" and self._route_tokens_fn is not None:
            import torch

            router = getattr(self, self._router_attr)
            was_training = router.training
            if was_training:
                router.eval()
            router_output = router(hidden_states.view(-1, self.hidden_size))
            if was_training:
                router.train()
            if isinstance(router_output, torch.Tensor) and router_output.dim() > 2:
                router_output = router_output.reshape(-1, router_output.shape[-1])
            topk_ids, topk_weights = self._route_tokens_fn(router_output)
            if topk_ids.dtype not in (torch.int32, torch.int64):
                raise RuntimeError(f"GLM router returned non-integral expert ids: {topk_ids.dtype}")
            if topk_ids.numel() and (
                int(topk_ids.min()) < 0 or int(topk_ids.max()) >= self.num_experts
            ):
                raise RuntimeError("GLM router emitted an out-of-range expert id")
            return topk_ids, topk_weights.to(torch.bfloat16)
        return original(self, hidden_states)

    TorchMoELayerWrapper._compute_routing = _compute_routing


def _patch_owner_backward() -> None:
    import torch
    from accelerate.utils.torch_moe import TorchMoELayerWrapper

    original = TorchMoELayerWrapper._owner_backward

    def _owner_backward(self, hs, topk_ids, topk_weights, grad_output):  # type: ignore[no-untyped-def]
        if not bool(getattr(self, "_full_weight_grad", False)):
            return original(self, hs, topk_ids, topk_weights, grad_output)

        import torch.distributed as dist
        from accelerate.utils.kt_moe import (
            _all_gather_qlens,
            _dist_gather_varlen_to_owner,
            _dist_scatter_varlen_from_owner,
        )

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        all_qlens = (
            _all_gather_qlens(hs.shape[0], hs.device, world_size) if world_size > 1 else [hs.shape[0]]
        )

        def gather(tensor):  # type: ignore[no-untyped-def]
            if world_size == 1:
                return [tensor]
            return _dist_gather_varlen_to_owner(
                tensor,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
                owner_rank=self.owner_rank,
            )

        grad_chunks = gather(grad_output)
        hs_chunks = gather(hs)
        ids_chunks = gather(topk_ids)
        weight_chunks = gather(topk_weights)
        if rank == self.owner_rank:
            replay_dtype_name = os.environ.get("TORCH_MOE_BACKWARD_DTYPE", "bfloat16").lower()
            if replay_dtype_name in ("float32", "fp32"):
                replay_dtype = torch.float32
            elif replay_dtype_name in ("bfloat16", "bf16"):
                replay_dtype = torch.bfloat16
            else:
                raise RuntimeError(f"unsupported TORCH_MOE_BACKWARD_DTYPE={replay_dtype_name!r}")
            cpu_hs = torch.cat(hs_chunks).to("cpu", dtype=replay_dtype)
            cpu_weights = torch.cat(weight_chunks).to("cpu", dtype=replay_dtype)
            cpu_ids = torch.cat(ids_chunks).to("cpu")
            cpu_grad = torch.cat(grad_chunks).to("cpu", dtype=replay_dtype)
            grad_hs = torch.zeros_like(cpu_hs)
            grad_weights = torch.zeros_like(cpu_weights)
            for expert_idx in torch.unique(cpu_ids).tolist():
                expert_idx = int(expert_idx)
                pair_positions = torch.nonzero(cpu_ids == expert_idx, as_tuple=False)
                if pair_positions.numel() == 0:
                    continue
                token_indices = pair_positions[:, 0]
                slot_indices = pair_positions[:, 1]
                x = cpu_hs.index_select(0, token_indices).detach().requires_grad_(True)
                route_weight = cpu_weights[token_indices, slot_indices].detach().requires_grad_(True)
                weight_masters: list[Any] = []
                weight_leaves: list[Any] = []
                if self.expert_layout == "fused_gate_up":
                    gate_up_leaf = (
                        self.expert_gate_up_proj[expert_idx].detach().to(replay_dtype).requires_grad_(True)
                    )
                    gate_weight, up_weight = gate_up_leaf.chunk(2, dim=0)
                    down_leaf = self.expert_down_proj[expert_idx].detach().to(replay_dtype).requires_grad_(True)
                    weight_masters = [self.expert_gate_up_proj, self.expert_down_proj]
                    weight_leaves = [gate_up_leaf, down_leaf]
                    down_weight = down_leaf
                else:
                    gate_leaf = self.expert_gate_proj[expert_idx].detach().to(replay_dtype).requires_grad_(True)
                    up_leaf = self.expert_up_proj[expert_idx].detach().to(replay_dtype).requires_grad_(True)
                    down_leaf = self.expert_down_proj[expert_idx].detach().to(replay_dtype).requires_grad_(True)
                    gate_weight, up_weight, down_weight = gate_leaf, up_leaf, down_leaf
                    weight_masters = [self.expert_gate_proj, self.expert_up_proj, self.expert_down_proj]
                    weight_leaves = [gate_leaf, up_leaf, down_leaf]
                with torch.enable_grad():
                    gate_out = torch.nn.functional.linear(x, gate_weight)
                    up_out = torch.nn.functional.linear(x, up_weight)
                    activation = torch.nn.functional.silu(gate_out) * up_out
                    down_out = torch.nn.functional.linear(activation, down_weight)
                    weighted_output = route_weight.unsqueeze(-1) * down_out
                    grads = torch.autograd.grad(
                        weighted_output,
                        [x, route_weight, *weight_leaves],
                        cpu_grad.index_select(0, token_indices),
                        allow_unused=True,
                    )
                grad_x, grad_route = grads[:2]
                weight_grads = grads[2:]
                grad_hs.index_add_(0, token_indices, grad_x)
                grad_weights[token_indices, slot_indices] += grad_route
                self._append_indexed_grads(
                    expert_idx,
                    weight_masters,
                    weight_leaves,
                    tuple(weight_grads),
                    scale=1.0 / world_size,
                )
            h_chunks = list(grad_hs.to(hs.device, dtype=hs.dtype).split([int(q) for q in all_qlens], dim=0))
            w_chunks = list(
                grad_weights.to(topk_weights.device, dtype=topk_weights.dtype).split(
                    [int(q) for q in all_qlens], dim=0
                )
            )
        else:
            h_chunks = w_chunks = None

        def scatter(chunks, feature_shape, dtype):  # type: ignore[no-untyped-def]
            if world_size == 1:
                return chunks[0]
            return _dist_scatter_varlen_from_owner(
                owner_chunks=chunks,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
                owner_rank=self.owner_rank,
                feature_shape=feature_shape,
                device=hs.device,
                dtype=dtype,
            )

        return (
            scatter(h_chunks, (self.hidden_size,), hs.dtype),
            scatter(w_chunks, (topk_weights.shape[-1],), topk_weights.dtype),
        )

    TorchMoELayerWrapper._owner_backward = _owner_backward


def _patch_glm4_moe_arch() -> None:
    """Vendor get_moe_arch_config does not know Glm4Moe; keep that file untouched."""
    import accelerate.utils.kt_moe as kt_moe

    original = kt_moe.get_moe_arch_config

    def get_moe_arch_config(config):  # type: ignore[no-untyped-def]
        text = kt_moe._moe_text_config(config)
        architectures = list(getattr(text, "architectures", None) or getattr(config, "architectures", None) or [])
        arch = architectures[0] if architectures else ""
        model_type = str(getattr(text, "model_type", "") or getattr(config, "model_type", ""))
        if "Glm4Moe" in arch or model_type == "glm4_moe":
            return kt_moe.MOEArchConfig(
                moe_layer_attr="mlp",
                router_attr="gate",
                experts_attr="experts",
                weight_names=("gate_proj", "up_proj", "down_proj"),
                expert_num=int(text.n_routed_experts),
                intermediate_size=int(text.moe_intermediate_size),
                num_experts_per_tok=int(text.num_experts_per_tok),
                has_shared_experts=int(getattr(text, "n_shared_experts", 0) or 0) > 0,
                router_type="deepseek_gate",
                expert_layout="separate",
                layers_prefix="model.layers",
            )
        return original(config)

    kt_moe.get_moe_arch_config = get_moe_arch_config


def _patch_kt_adapt() -> None:
    import accelerate.utils.kt_moe as kt_moe

    original = kt_moe.kt_adapt_peft_lora

    def kt_adapt_peft_lora(model):  # type: ignore[no-untyped-def]
        if os.environ.get("ACCELERATE_KT_TRAIN_MODE", "").strip().lower() == "full":
            print("[pytorch_torch] skipping LoRA PEFT adapt; full-weight expert grads are enabled", flush=True)
            return None
        return original(model)

    kt_moe.kt_adapt_peft_lora = kt_adapt_peft_lora


def _patch_trainer_prepare() -> None:
    from transformers import Trainer

    original = Trainer._prepare_for_training

    def _prepare_for_training(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if model is None:
            return result
        unwrapped = model
        unwrap = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
        if callable(unwrap):
            unwrapped = unwrap(model, keep_torch_compile=False)
        synced = _sync_router_buffers(unwrapped)
        injected = _inject_optimizer_params(getattr(self, "optimizer", None), unwrapped)
        if _rank() == 0:
            if synced:
                print(
                    f"[pytorch_torch] synchronized {synced} router buffers onto CUDA",
                    flush=True,
                )
            if injected:
                print(
                    f"[pytorch_torch] injected {injected} owner expert parameters into the optimizer",
                    flush=True,
                )
        return result

    Trainer._prepare_for_training = _prepare_for_training


def install_torch_moe_full_ft() -> None:
    """Install process-local full-FT patches. Safe to call more than once."""
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("FFT_TRAINING_BACKEND", "").strip().lower() != "pytorch_torch":
        raise RuntimeError("install_torch_moe_full_ft requires FFT_TRAINING_BACKEND=pytorch_torch")
    if os.environ.get("ACCELERATE_KT_BACKEND", "").strip().upper() != "TORCH":
        raise RuntimeError("install_torch_moe_full_ft requires ACCELERATE_KT_BACKEND=TORCH")
    _patch_glm4_moe_arch()
    _patch_compute_routing()
    _patch_owner_backward()
    _patch_kt_adapt()
    _patch_trainer_prepare()
    _INSTALLED = True
    if _rank() == 0:
        print("[pytorch_torch] installed owner-sharded full-FT patches", flush=True)
