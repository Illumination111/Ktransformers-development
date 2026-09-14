#!/usr/bin/env python3
"""PyTorch oneDNN hybrid full-FT benchmark: GPU trunk plus CPU experts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

MODEL_DEFAULTS = {
    "qwen35_122b": "/mnt/data2/models/Qwen3.5-122B-A10B",
    "glm45_air": "/mnt/data2/models/GLM-4.5-Air",
}
EXPECTED_MODELS = {
    "qwen35_122b": ("qwen3_5_moe_text", "Qwen3_5MoeForCausalLM"),
    "glm45_air": ("glm4_moe", "Glm4MoeForCausalLM"),
}


class _HybridMoE:
    """GPU router/shared path with the routed expert weights kept on CPU.

    The class is installed as a real ``nn.Module`` at runtime.  Keeping this
    small wrapper local to the benchmark avoids changing Transformers model
    implementations used by the KT and APTMoE paths.
    """

    def __new__(cls, block, model_kind: str):  # type: ignore[no-untyped-def]
        import torch.nn as nn

        class HybridMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.model_kind = model_kind
                self.gate = block.gate
                self.experts = block.experts
                if model_kind == "qwen35_122b":
                    self.shared_expert = block.shared_expert
                    self.shared_expert_gate = block.shared_expert_gate
                else:
                    self.shared_experts = block.shared_experts

            def forward(self, hidden_states):  # type: ignore[no-untyped-def]
                import torch

                original_shape = hidden_states.shape
                flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                if self.model_kind == "qwen35_122b":
                    shared = self.shared_expert(flat)
                    _, routing_weights, selected_experts = self.gate(flat)
                else:
                    router_logits = self.gate(flat)
                    selected_experts, routing_weights = self.route_tokens_to_experts(
                        router_logits
                    )
                    shared = self.shared_experts(flat)
                    shared = shared.reshape(-1, shared.shape[-1])

                # The copy operations retain autograd edges, so gradients for
                # CPU expert parameters and the GPU router weights both flow.
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    expert = self.experts(
                        flat.to(device="cpu"),
                        selected_experts.to(device="cpu"),
                        routing_weights.to(device="cpu"),
                    )
                expert = expert.to(device=flat.device, dtype=flat.dtype)
                if self.model_kind == "qwen35_122b":
                    shared = torch.sigmoid(self.shared_expert_gate(flat)) * shared
                return (expert + shared).reshape(original_shape)

            def route_tokens_to_experts(self, router_logits):  # type: ignore[no-untyped-def]
                import torch

                router_logits = router_logits.sigmoid()
                correction = self.gate.e_score_correction_bias
                group_scores = (
                    (router_logits + correction)
                    .view(-1, self.gate.n_group, self.gate.n_routed_experts // self.gate.n_group)
                    .topk(2, dim=-1)[0]
                    .sum(dim=-1)
                )
                group_idx = torch.topk(
                    group_scores, k=self.gate.topk_group, dim=-1, sorted=False
                )[1]
                group_mask = torch.zeros_like(group_scores)
                group_mask.scatter_(1, group_idx, 1)
                score_mask = (
                    group_mask.unsqueeze(-1)
                    .expand(-1, self.gate.n_group, self.gate.n_routed_experts // self.gate.n_group)
                    .reshape(-1, self.gate.n_routed_experts)
                )
                scores = (router_logits + correction).masked_fill(
                    ~score_mask.bool(), float("-inf")
                )
                indices = torch.topk(scores, k=self.gate.top_k, dim=-1, sorted=False)[1]
                weights = router_logits.gather(1, indices)
                if self.gate.norm_topk_prob:
                    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
                return indices, weights * self.gate.routed_scaling_factor

        return HybridMoE()


def _install_hybrid_moe(model, model_kind: str):  # type: ignore[no-untyped-def]
    """Replace native MoE blocks with a GPU-trunk/CPU-expert implementation."""
    layer_names = {
        "qwen35_122b": "Qwen3_5MoeDecoderLayer",
        "glm45_air": "Glm4MoeDecoderLayer",
    }
    expected = layer_names[model_kind]
    accepted = {expected}
    if model_kind == "qwen35_122b":
        accepted.add("Qwen3_5DecoderLayer")
    replaced = 0
    for module in list(model.modules()):
        if type(module).__name__ in accepted and hasattr(module.mlp, "experts"):
            module.mlp = _HybridMoE(module.mlp, model_kind)
            replaced += 1
    if replaced <= 0:
        raise RuntimeError(f"no {expected} MoE layers found for hybrid oneDNN")
    return replaced


def _move_hybrid_model(model, device):  # type: ignore[no-untyped-def]
    """Move every non-expert module to CUDA while leaving ``*.experts`` on CPU."""
    import torch

    def visit(module):  # type: ignore[no-untyped-def]
        children = list(module.named_children())
        for name, child in children:
            if name != "experts":
                visit(child)
        for parameter in module.parameters(recurse=False):
            parameter.data = parameter.data.to(device)
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                module._buffers[name] = buffer.to(device)

    visit(model)
    cpu_experts = 0
    cuda_parameters = 0
    for name, parameter in model.named_parameters():
        if ".experts." in f".{name}.":
            if parameter.device.type != "cpu":
                raise RuntimeError(f"expert parameter moved off CPU: {name} -> {parameter.device}")
            cpu_experts += parameter.numel()
        else:
            if parameter.device != device:
                raise RuntimeError(f"trunk parameter not on {device}: {name} -> {parameter.device}")
            cuda_parameters += parameter.numel()
    if not cpu_experts or not cuda_parameters:
        raise RuntimeError(
            f"invalid hybrid residency: cpu_experts={cpu_experts}, cuda_trunk={cuda_parameters}"
        )
    return {"cpu_expert_parameters": cpu_experts, "cuda_trunk_parameters": cuda_parameters}


class CPUStateAdamW:
    """AdamW whose persistent moments live on CPU for every parameter."""

    def __init__(self, parameters, lr: float, weight_decay: float):  # type: ignore[no-untyped-def]
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.step_index = 0
        self.state: dict[Any, dict[str, Any]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    def step(self) -> None:
        import torch

        self.step_index += 1
        beta1, beta2 = 0.9, 0.999
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            # CPU experts already take this path without a device transfer;
            # CUDA trunk tensors use temporary CPU copies while moments remain
            # persistently CPU-resident.
            value = parameter.detach().to(device="cpu", dtype=torch.float32)
            gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
            state = self.state.setdefault(
                parameter,
                {
                    "step": 0,
                    "exp_avg": torch.zeros_like(value, dtype=torch.bfloat16),
                    "exp_avg_sq": torch.zeros_like(value, dtype=torch.bfloat16),
                },
            )
            state["step"] += 1
            gradient_bf16 = gradient.to(dtype=torch.bfloat16)
            state["exp_avg"].mul_(beta1).add_(gradient_bf16, alpha=1.0 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(gradient_bf16, gradient_bf16, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            step_size = self.lr / bias_correction1
            exp_avg = state["exp_avg"].to(dtype=torch.float32)
            denom = state["exp_avg_sq"].to(dtype=torch.float32).sqrt().div_(bias_correction2**0.5).add_(1e-8)
            if self.weight_decay:
                value.mul_(1.0 - self.lr * self.weight_decay)
            value.addcdiv_(exp_avg, denom, value=-step_size)
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="fft_real_100")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-stage microbatch size (KT uses 1)")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0,
                        help="Override KT thread roles on every rank")
    parser.add_argument("--owner-threads", type=int, default=80,
                        help="Rank 0 CPU expert/optimizer threads")
    parser.add_argument("--non-owner-threads", type=int, default=2,
                        help="CPU threads for ranks 1..N-1")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--pipeline-parallel-size", type=int, default=8,
                        help="Pipeline ranks per data-parallel replica")
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="Number of replicated pipeline groups")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel degree (currently only 1 is supported)")
    parser.add_argument("--device", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Kept for launcher compatibility; stage pipeline retains microbatch activations")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _rss_gb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / (1024**2)


def _load_records(dataset_dir: Path, dataset_name: str) -> list[dict[str, Any]]:
    info_path = dataset_dir / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if dataset_name not in info:
        raise ValueError(f"dataset {dataset_name!r} is missing from {info_path}")
    data_path = dataset_dir / info[dataset_name]["file_name"]
    records = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"dataset is empty or not a list: {data_path}")
    required = {"instruction", "input", "output"}
    if not required.issubset(records[0]):
        raise ValueError(f"dataset record must contain {sorted(required)}")
    return records


def _materialize_module(module: Any, device: Any, prefix: str = "") -> None:
    """Allocate only this module, keeping ``*.experts`` on host memory."""
    import torch
    import torch.nn as nn

    for name, parameter in list(module._parameters.items()):
        if parameter is None:
            continue
        target = torch.device("cpu") if ".experts" in f".{prefix}." else device
        if parameter.is_meta:
            module._parameters[name] = nn.Parameter(
                torch.empty(parameter.shape, dtype=parameter.dtype, device=target),
                requires_grad=parameter.requires_grad,
            )
    for name, buffer in list(module._buffers.items()):
        if buffer is not None and buffer.is_meta:
            module._buffers[name] = torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        _materialize_module(child, device, child_prefix)


def _checkpoint_name_candidates(name: str, model_kind: str) -> list[str]:
    if model_kind != "qwen35_122b":
        return [name]
    if name.startswith("model."):
        return [name, "model.language_model." + name[len("model."):]]
    return [name]


def _load_stage_model(args: argparse.Namespace, dtype: Any, rank: int, pipeline_size: int, device: Any):
    """Build a meta HF model, materialize this rank's stages, and load their shards."""
    import torch
    import torch.nn as nn
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model_path or MODEL_DEFAULTS[args.model_kind]).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    source_config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    expected_type, expected_arch = EXPECTED_MODELS[args.model_kind]
    if args.model_kind == "qwen35_122b":
        helper_dir = Path(__file__).resolve().parents[1] / "Qwen3.5-122B-A10B"
        sys.path.insert(0, str(helper_dir))
        from qwen35_text_only import _extract_text_config, assert_text_only_model
        config = _extract_text_config(source_config)
    else:
        if getattr(source_config, "model_type", None) != expected_type:
            raise RuntimeError(
                f"unexpected GLM model_type={getattr(source_config, 'model_type', None)!r}"
            )
        config = source_config

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.to(dtype=dtype)
    # GLM keeps the router correction bias in FP32.  ``Module.to(dtype=...)``
    # also changes meta buffers, so restore that contract before materializing.
    for module_name, module in model.named_modules():
        for buffer_name, buffer in list(module._buffers.items()):
            if buffer is not None and "e_score_correction_bias" in buffer_name:
                module._buffers[buffer_name] = torch.empty(
                    buffer.shape, dtype=torch.float32, device="meta"
                )
    if args.model_kind == "qwen35_122b":
        assert_text_only_model(model, "full")
    architectures = list(getattr(model.config, "architectures", None) or [])
    if args.model_kind == "glm45_air" and type(model).__name__ != expected_arch and expected_arch not in architectures:
        raise RuntimeError(f"unexpected GLM architecture class={type(model).__name__!r}, config={architectures!r}")
    model.config.use_cache = False
    replaced = _install_hybrid_moe(model, args.model_kind)

    num_layers = int(model.config.num_hidden_layers)
    if num_layers < pipeline_size:
        raise ValueError(f"model has {num_layers} layers, fewer than pipeline size {pipeline_size}")
    pipeline_rank = rank % pipeline_size
    owned_ids = [layer_id for layer_id in range(num_layers) if layer_id % pipeline_size == pipeline_rank]
    last_owner = (num_layers - 1) % pipeline_size
    base = model.model
    roots: list[tuple[str, Any]] = [(f"model.layers.{i}", base.layers[i]) for i in owned_ids]
    if pipeline_rank == 0:
        roots.append(("model.embed_tokens", base.embed_tokens))
    if pipeline_rank == last_owner:
        roots.extend([("model.norm", base.norm), ("lm_head", model.lm_head)])
    for root_name, root_module in roots:
        _materialize_module(root_module, device, root_name)

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    tensors: dict[str, Any] = {}
    tensors.update(model.named_parameters())
    tensors.update(model.named_buffers())
    selected_prefixes = [f"model.layers.{i}." for i in owned_ids]
    if pipeline_rank == 0:
        selected_prefixes.append("model.embed_tokens.")
    if pipeline_rank == last_owner:
        selected_prefixes.extend(["model.norm.", "lm_head."])
    selected = {
        name: tensor for name, tensor in tensors.items()
        if any(name.startswith(prefix) for prefix in selected_prefixes) and not tensor.is_meta
    }
    loaded = 0
    missing: list[str] = []
    # Build a shard-indexed request list first.  Opening one safetensors shard
    # per parameter causes thousands of repeated mmap/open/close operations on
    # the 122B checkpoints and leaves the GPUs idle for a very long time.
    requests: dict[str, list[tuple[str, Any, str, int | None]]] = {}
    for name, tensor in selected.items():
        source_name = next((candidate for candidate in _checkpoint_name_candidates(name, args.model_kind)
                            if candidate in weight_map), None)
        if source_name is not None:
            requests.setdefault(weight_map[source_name], []).append((source_name, tensor, "direct", None))
            continue
        if args.model_kind == "glm45_air" and name.endswith("mlp.experts.gate_up_proj"):
            for expert_id in range(int(tensor.shape[0])):
                prefix = name.rsplit('.', 1)[0]
                for part, offset in (("gate_proj.weight", 0), ("up_proj.weight", int(tensor.shape[1]) // 2)):
                    source_name = f"{prefix}.{expert_id}.{part}"
                    if source_name not in weight_map:
                        missing.append(name)
                        break
                    requests.setdefault(weight_map[source_name], []).append((source_name, tensor, "gate_up", expert_id * 2 + (offset != 0)))
            continue
        if args.model_kind == "glm45_air" and name.endswith("mlp.experts.down_proj"):
            prefix = name.rsplit(".", 1)[0]
            for expert_id in range(int(tensor.shape[0])):
                source_name = f"{prefix}.{expert_id}.down_proj.weight"
                if source_name not in weight_map:
                    missing.append(name)
                    break
                requests.setdefault(weight_map[source_name], []).append((source_name, tensor, "down", expert_id))
            continue
        missing.append(name)

    request_total = sum(len(items) for items in requests.values())
    if rank == 0:
        print(f"[pytorch_onednn] loading {len(requests)} safetensors shards ({request_total} tensors) for pipeline rank {pipeline_rank}", flush=True)
    for shard_index, (shard_name, shard_requests) in enumerate(requests.items(), start=1):
        with safe_open(str(model_path / shard_name), framework="pt", device="cpu") as handle:
            for source_name, tensor, kind, slot in shard_requests:
                value = handle.get_tensor(source_name)
                if kind == "direct":
                    tensor.data.copy_(value.to(device=tensor.device, dtype=tensor.dtype))
                elif kind == "down":
                    tensor.data[slot].copy_(value.to(device=tensor.device, dtype=tensor.dtype))
                else:
                    # gate/up requests are paired by expert slot; concatenate
                    # after both halves have been read from their shard.
                    half = tensor.shape[1] // 2
                    target = slot // 2
                    start = 0 if slot % 2 == 0 else half
                    tensor.data[target, start:start + value.shape[0]].copy_(value.to(device=tensor.device, dtype=tensor.dtype))
                del value
                loaded += 1
        if rank == 0:
            print(f"[pytorch_onednn] loaded shard {shard_index}/{len(requests)} {shard_name} ({loaded}/{request_total} tensors)", flush=True)
    if missing:
        generated = [name for name in missing if "rotary_emb" in name]
        missing = [name for name in missing if name not in generated]
    if missing:
        raise RuntimeError(f"missing checkpoint tensors on rank {rank}: {missing[:8]}")

    class StageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_kind = args.model_kind
            self.config = model.config
            self.layer_ids = owned_ids
            self.layers = nn.ModuleList([base.layers[i] for i in owned_ids])
            # RoPE inverse-frequency buffers are generated from config and are
            # intentionally absent from the safetensors index.
            self.rotary_emb = type(base.rotary_emb)(config=self.config).to(device=device)
            if pipeline_rank == 0:
                self.embed_tokens = base.embed_tokens
            if pipeline_rank == last_owner:
                self.norm = base.norm
                self.lm_head = model.lm_head

        def forward_layer(self, layer_position: int, hidden_states: Any, position_ids: Any) -> Any:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            layer = self.layers[layer_position]
            return layer(hidden_states, position_embeddings=position_embeddings,
                         attention_mask=None, position_ids=position_ids,
                         past_key_values=None, use_cache=False)

    # All selected roots were materialized explicitly above.  Do not call
    # ``StageModel.to(cuda)`` here: that would incorrectly move CPU experts.
    stage = StageModel()
    for parameter in stage.parameters():
        parameter.requires_grad_(True)
    stage.train()
    local_total = sum(parameter.numel() for parameter in stage.parameters())
    local_cpu_experts = sum(parameter.numel() for name, parameter in stage.named_parameters()
                             if ".experts." in f".{name}.")
    local_cuda = sum(parameter.numel() for name, parameter in stage.named_parameters()
                     if ".experts." not in f".{name}.")
    if not local_cpu_experts or not local_cuda:
        raise RuntimeError(f"invalid rank {rank} residency: cpu_experts={local_cpu_experts}, cuda_trunk={local_cuda}")
    residency = {"rank": rank, "pipeline_rank": pipeline_rank, "owned_layers": owned_ids, "num_layers": num_layers,
                 "last_owner_rank": last_owner, "loaded_tensors": loaded,
                 "cpu_expert_parameters": local_cpu_experts,
                 "cuda_trunk_parameters": local_cuda, "device": str(device),
                 "moe_layers_replaced_global": replaced}
    return stage, tokenizer, model_path, local_total, residency, num_layers, last_owner, pipeline_rank


def _make_batches(tokenizer: Any, records: list[dict[str, Any]], sequence_length: int, batch_size: int):
    import torch
    samples: list[dict[str, Any]] = []
    for record in records:
        text = str(record["instruction"])
        if record.get("input"):
            text += "\n" + str(record["input"])
        text += "\n" + str(record["output"])
        encoded = tokenizer(text, max_length=sequence_length, truncation=True,
                            padding="max_length", return_tensors="pt")
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        samples.append({"input_ids": encoded["input_ids"].squeeze(0),
                        "attention_mask": encoded["attention_mask"].squeeze(0),
                        "labels": labels.squeeze(0)})
    batches = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        if len(chunk) < batch_size:
            chunk = chunk + [samples[j % len(chunk)] for j in range(batch_size - len(chunk))]
        batches.append({key: torch.stack([item[key] for item in chunk]) for key in chunk[0]})
    return batches


def _stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {"count": 0, "mean_sec": None, "p50_sec": None, "p95_sec": None}
    ordered = sorted(values)
    return {"count": len(values), "mean_sec": statistics.fmean(values),
            "p50_sec": ordered[(len(ordered) - 1) // 2],
            "p95_sec": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]}


def _init_distributed(args: argparse.Namespace):
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not args.dry_run and world_size != args.num_gpus:
        raise RuntimeError(
            f"KT-compatible run requires torchrun with {args.num_gpus} ranks; got WORLD_SIZE={world_size}"
        )
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        dist.init_process_group(backend="nccl", init_method="env://")
    if not torch.cuda.is_available():
        raise RuntimeError("8-card oneDNN stage pipeline requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device(f"cuda:{local_rank}"), dist


def _set_threads(args: argparse.Namespace, rank: int) -> int:
    import torch

    count = args.threads if args.threads > 0 else (args.owner_threads if rank == 0 else args.non_owner_threads)
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(max(1, min(count, 4)))
    except RuntimeError:
        pass
    return count


def _max_time(value: float, device: Any, dist: Any) -> float:
    import torch

    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _pipeline_microbatch(stage: Any, input_ids: Any, labels: Any, sequence_length: int,
                         rank: int, pipeline_rank: int, pipeline_size: int, dp_rank: int,
                         num_layers: int, last_owner: int, device: Any, dist: Any,
                         pipeline_group: Any, loss_scale: float):
    import torch
    import torch.nn.functional as F

    batch_size = labels.shape[0]
    hidden_size = int(stage.config.hidden_size)
    position_ids = torch.arange(sequence_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    saved: dict[int, tuple[Any, Any]] = {}
    loss = None
    forward_start = time.perf_counter()
    group_base = dp_rank * pipeline_size
    for layer_id in range(num_layers):
        owner = layer_id % pipeline_size
        if pipeline_rank != owner:
            continue
        if layer_id == 0:
            hidden = stage.embed_tokens(input_ids)
        else:
            previous_owner = group_base + (layer_id - 1) % pipeline_size
            if (layer_id - 1) % pipeline_size != owner:
                hidden = torch.empty((batch_size, sequence_length, hidden_size), dtype=torch.bfloat16, device=device)
                dist.recv(hidden, src=previous_owner, group=pipeline_group)
            else:
                hidden = saved[layer_id - 1][1]
        if not hidden.is_leaf:
            hidden.retain_grad()
        else:
            hidden.requires_grad_(True)
            hidden.retain_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = stage.forward_layer(stage.layer_ids.index(layer_id), hidden, position_ids)
        output.retain_grad()
        saved[layer_id] = (hidden, output)
        if layer_id == num_layers - 1:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = stage.lm_head(stage.norm(output))
                loss = F.cross_entropy(
                    logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1), ignore_index=-100,
                )
        else:
            next_owner_local = (layer_id + 1) % pipeline_size
            if next_owner_local != owner:
                dist.send(output.detach(), dst=group_base + next_owner_local, group=pipeline_group)

    forward_sec = time.perf_counter() - forward_start
    backward_start = time.perf_counter()
    if pipeline_rank == last_owner:
        if loss is None:
            raise RuntimeError("last pipeline stage did not compute loss")
        (loss * loss_scale).backward()

    for layer_id in range(num_layers - 1, -1, -1):
        owner = layer_id % pipeline_size
        if pipeline_rank != owner:
            continue
        hidden, output = saved[layer_id]
        if layer_id != num_layers - 1:
            next_owner_local = (layer_id + 1) % pipeline_size
            if next_owner_local != owner:
                grad_output = torch.empty_like(output)
                dist.recv(grad_output, src=group_base + next_owner_local, group=pipeline_group)
            else:
                grad_output = saved[layer_id + 1][0].grad
            if grad_output is None:
                raise RuntimeError(f"missing pipeline gradient for layer {layer_id}")
            output.backward(grad_output)
        if layer_id > 0:
            previous_owner_local = (layer_id - 1) % pipeline_size
            if previous_owner_local != owner:
                if hidden.grad is None:
                    raise RuntimeError(f"missing input gradient for layer {layer_id}")
                dist.send(hidden.grad.detach(), dst=group_base + previous_owner_local, group=pipeline_group)
    return loss, forward_sec, time.perf_counter() - backward_start


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("steps must be positive and warmup-steps must be smaller than steps")
    if args.sequence_length <= 0 or args.batch_size <= 0 or args.global_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("sequence length, batch size, and gradient accumulation must be positive")
    if args.threads < 0 or args.owner_threads <= 0 or args.non_owner_threads <= 0:
        raise ValueError("thread counts must be positive")
    if args.global_batch_size % args.batch_size:
        raise ValueError("global-batch-size must be divisible by batch-size")
    if args.tensor_parallel_size != 1:
        raise ValueError("tensor-parallel-size > 1 is not implemented; use 1")
    if args.pipeline_parallel_size <= 0 or args.data_parallel_size <= 0:
        raise ValueError("pipeline/data parallel sizes must be positive")
    if args.num_gpus != args.pipeline_parallel_size * args.data_parallel_size:
        raise ValueError("num-gpus must equal pipeline-parallel-size * data-parallel-size")
    if args.global_batch_size % args.data_parallel_size:
        raise ValueError("global-batch-size must be divisible by data-parallel-size")
    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rank, world_size, device, dist, cpu_threads = 0, args.num_gpus, "cuda:0", None, 0
    else:
        rank, world_size, device, dist = _init_distributed(args)
        cpu_threads = _set_threads(args, rank)
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            dist.barrier()
    config_payload = vars(args).copy()
    config_payload["model_path"] = str(args.model_path or MODEL_DEFAULTS[args.model_kind])
    config_payload["dataset_dir"] = str(args.dataset_dir.resolve())
    config_payload["output_dir"] = str(args.output_dir.resolve())
    config_payload.update({
        "backend": "pytorch_onednn",
        "execution_layout": "fsdp_style_data_parallel_gpu_trunk_cpu_experts",
        "parallelism": {"pipeline_parallel": args.pipeline_parallel_size,
                        "tensor_parallel": args.tensor_parallel_size,
                        "data_parallel": args.data_parallel_size,
                        "virtual_stages": "one decoder layer",
                        "fsdp_parameter_sharding": "gpu trunk replicated stage-local; CPU experts replicated and gradient-reduced"},
        "num_gpus": args.num_gpus,
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "expert_device": "cpu",
        "global_batch_size": args.global_batch_size,
        "per_device_batch_size": args.batch_size,
        "cpu_threads_per_rank": cpu_threads,
        "kt_owner_rank": 0,
        "kt_owner_threads": args.owner_threads,
        "kt_non_owner_threads": args.non_owner_threads,
        "precision": "bf16",
    })
    if args.dry_run:
        (args.output_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(config_payload, indent=2, default=str))
        return 0
    if rank == 0:
        (args.output_dir / "run_config.json").write_text(
            json.dumps(config_payload, indent=2, default=str) + "\n", encoding="utf-8"
        )

    import torch
    if not torch.backends.mkldnn.is_available():
        raise RuntimeError("this PyTorch build does not expose oneDNN (torch.backends.mkldnn)")
    torch.backends.mkldnn.enabled = True
    torch.manual_seed(args.seed)
    pipeline_rank = rank % args.pipeline_parallel_size
    dp_rank = rank // args.pipeline_parallel_size
    stage, tokenizer, model_path, local_parameter_count, residency, num_layers, last_owner, loaded_pipeline_rank = _load_stage_model(
        args, torch.bfloat16, rank, args.pipeline_parallel_size, device
    )
    if loaded_pipeline_rank != pipeline_rank:
        raise RuntimeError("stage rank mapping mismatch")
    pipeline_groups = []
    dp_groups = []
    dp_cpu_groups = []
    if world_size > 1:
        for group_dp in range(args.data_parallel_size):
            pipeline_groups.append(dist.new_group(
                list(range(group_dp * args.pipeline_parallel_size,
                           (group_dp + 1) * args.pipeline_parallel_size)), backend="nccl"))
        for group_pipe in range(args.pipeline_parallel_size):
            ranks = [group_dp * args.pipeline_parallel_size + group_pipe
                     for group_dp in range(args.data_parallel_size)]
            dp_groups.append(dist.new_group(ranks, backend="nccl"))
            dp_cpu_groups.append(dist.new_group(ranks, backend="gloo"))
    pipeline_group = pipeline_groups[dp_rank] if world_size > 1 else None
    dp_group = dp_groups[pipeline_rank] if world_size > 1 else None
    dp_cpu_group = dp_cpu_groups[pipeline_rank] if world_size > 1 else None
    records = _load_records(args.dataset_dir, args.dataset_name)
    local_batch_size = args.global_batch_size // args.data_parallel_size
    # Every pipeline group's input rank builds the same global batch and takes
    # a distinct DP slice, matching KT's global_batch/per_device_batch contract.
    batches = _make_batches(tokenizer, records, args.sequence_length, args.global_batch_size) if pipeline_rank == 0 else []
    if pipeline_rank == 0 and not batches:
        raise RuntimeError("dataset did not produce a global batch")
    optimizer = CPUStateAdamW(
        stage.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rows: list[dict[str, Any]] = []
    batch_index = 0
    microbatches = local_batch_size // args.batch_size
    tokens_per_step = args.global_batch_size * args.gradient_accumulation_steps * args.sequence_length
    completed = False
    try:
        for step in range(1, args.steps + 1):
            forward_sec = backward_sec = 0.0
            step_loss_sum = 0.0
            step_start = time.perf_counter()
            for _ in range(args.gradient_accumulation_steps):
                batch = batches[batch_index % len(batches)] if pipeline_rank == 0 else None
                batch_index += 1
                if batch is not None and args.data_parallel_size > 1:
                    start = dp_rank * local_batch_size
                    end = start + local_batch_size
                    batch = {key: value[start:end] for key, value in batch.items()}
                for microbatch in range(microbatches):
                    if pipeline_rank == 0:
                        start = microbatch * args.batch_size
                        end = start + args.batch_size
                        input_ids = batch["input_ids"][start:end].to(device)
                        labels = batch["labels"][start:end].to(device)
                    else:
                        input_ids = None
                        labels = torch.empty((args.batch_size, args.sequence_length), dtype=torch.long, device=device)
                    if world_size > 1:
                        dist.broadcast(labels, src=dp_rank * args.pipeline_parallel_size,
                                       group=pipeline_group)
                    loss, micro_forward, micro_backward = _pipeline_microbatch(
                        stage, input_ids, labels, args.sequence_length, rank, pipeline_rank,
                        args.pipeline_parallel_size, dp_rank, num_layers, last_owner, device, dist,
                        pipeline_group, 1.0 / (microbatches * args.gradient_accumulation_steps),
                    )
                    torch.cuda.synchronize(device)
                    forward_sec += micro_forward
                    if pipeline_rank == last_owner:
                        loss_value = torch.tensor([float(loss.detach())], dtype=torch.float64, device=device)
                    else:
                        loss_value = torch.zeros(1, dtype=torch.float64, device=device)
                    if world_size > 1:
                        dist.broadcast(loss_value, src=dp_rank * args.pipeline_parallel_size + last_owner,
                                       group=pipeline_group)
                    backward_sec += micro_backward
                    if pipeline_rank == last_owner:
                        step_loss_sum += float(loss.detach())
            if world_size > 1:
                for parameter in stage.parameters():
                    if parameter.grad is None:
                        continue
                    if parameter.grad.device.type == "cuda":
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=dp_group)
                        parameter.grad.div_(args.data_parallel_size)
                    else:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=dp_cpu_group)
                        parameter.grad.div_(args.data_parallel_size)
            optimizer_start = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            optimizer_sec = time.perf_counter() - optimizer_start
            forward_sec = _max_time(forward_sec, device, dist)
            optimizer_sec = _max_time(optimizer_sec, device, dist)
            total_sec = _max_time(time.perf_counter() - step_start, device, dist)
            if pipeline_rank == last_owner:
                step_loss = step_loss_sum / microbatches / args.gradient_accumulation_steps
            else:
                step_loss = 0.0
            if world_size > 1:
                loss_box = torch.tensor([step_loss], dtype=torch.float64, device=device)
                dist.broadcast(loss_box, src=dp_rank * args.pipeline_parallel_size + last_owner,
                               group=pipeline_group)
                step_loss = float(loss_box.item())
            row = {"global_step": step, "forward_sec": forward_sec, "backward_sec": backward_sec,
                   "optimizer_sec": optimizer_sec, "step_total_sec": total_sec,
                   "step_tps": tokens_per_step / total_sec,
                   "loss": step_loss, "rss_max_gb": _rss_gb(), "rank": rank}
            if rank == 0:
                rows.append(row)
                print(f"[pytorch_onednn] model={args.model_kind} seq={args.sequence_length} "
                      f"step={step}/{args.steps} pp={args.pipeline_parallel_size} "
                      f"dp={args.data_parallel_size} loss={row['loss']:.6f} "
                      f"step_sec={total_sec:.3f} tps={row['step_tps']:.3f} "
                      f"rss_max_gb={row['rss_max_gb']:.2f}", flush=True)
        completed = True
    finally:
        if rank != 0:
            if world_size > 1:
                dist.barrier()
            return 0
        stable = [row for row in rows if row["global_step"] > args.warmup_steps]
        stable_step = statistics.fmean([row["step_total_sec"] for row in stable]) if stable else None
        summary = {"schema_version": 2, "backend": "pytorch_onednn",
        "execution_layout": "fsdp_style_data_parallel_gpu_trunk_cpu_experts",
                   "parallelism": config_payload["parallelism"], "num_gpus": world_size,
                   "global_batch_size": args.global_batch_size, "per_device_batch_size": args.batch_size,
                   "device": str(device),
                   "expert_device": "cpu", "precision": "bf16",
                   "onednn_available": bool(torch.backends.mkldnn.is_available()),
                   "onednn_enabled": bool(torch.backends.mkldnn.enabled), "model_kind": args.model_kind,
                   "model_path": str(model_path) if 'model_path' in locals() else str(args.model_path or MODEL_DEFAULTS[args.model_kind]),
                   "parameter_count_local": local_parameter_count if 'local_parameter_count' in locals() else None,
                   "residency": residency if 'residency' in locals() else None,
                   "optimizer_state_device": "cpu",
                   "optimizer_state_dtype": "torch.bfloat16",
                   "tokens_per_step": tokens_per_step, "warmup_steps": args.warmup_steps,
                   "num_steps": len(rows), "num_stable_steps": len(stable), "steps": rows,
                   "aggregate_stable": {key: _stats(stable, key) for key in
                                        ("forward_sec", "backward_sec", "optimizer_sec", "step_total_sec")},
                   "stable_tps": tokens_per_step / stable_step if stable_step else None}
        (args.output_dir / "step_timing.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        with (args.output_dir / "step_timing.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["global_step"])
            writer.writeheader()
            writer.writerows(rows)
        (args.output_dir / "summary.md").write_text("# PyTorch oneDNN hybrid full fine-tuning\n\n"
            f"- Model: `{args.model_kind}`\n- Layout: PP={args.pipeline_parallel_size}, TP={args.tensor_parallel_size}, DP={args.data_parallel_size}; GPU trunk + CPU routed experts\n"
            f"- Global batch: `{args.global_batch_size}`; per-stage microbatch: `{args.batch_size}`\n"
            f"- Device: `{device}`\n- Expert device: `cpu`\n- Precision: BF16\n"
            f"- oneDNN available/enabled: `{torch.backends.mkldnn.is_available()}`/`{torch.backends.mkldnn.enabled}`\n"
            f"- Stable steps: {len(stable)} (warm-up excluded: {args.warmup_steps})\n"
            f"- Stable TPS: `{summary['stable_tps']}`\n", encoding="utf-8")
        (args.output_dir / "exit_code.txt").write_text(
            "0\n" if completed else "1\n", encoding="utf-8"
        )
        if world_size > 1:
            dist.barrier()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[pytorch_onednn] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
