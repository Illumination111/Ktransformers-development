"""GLM-4.5-Air proxy stages wired to APTMoE's expert offload bridge."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeMLP,
    Glm4MoeRMSNorm,
)

from model.transformer_lm import OffloadInputBegin, OffloadInputEnd
from Runtime.OffloadRuntime.offload import ModelShard

from glm45_air_aptmoe_proxy_components import (
    Glm45RoutedExpert,
    Glm45TokenMixer,
)
from .placement import ProxyPlacementSolver
from .routes import RouteController

LOSS_TOKEN_CHUNK_SIZE = 1024


def _chunked_causal_lm_loss(
    head: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_items_in_batch: int | None,
) -> torch.Tensor:
    flat_hidden = hidden[..., :-1, :].reshape(-1, hidden.shape[-1])
    flat_labels = labels[..., 1:].reshape(-1)
    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for start in range(0, flat_labels.numel(), LOSS_TOKEN_CHUNK_SIZE):
        stop = min(start + LOSS_TOKEN_CHUNK_SIZE, flat_labels.numel())
        total = total + F.cross_entropy(
            head(flat_hidden[start:stop]).float(),
            flat_labels[start:stop],
            ignore_index=-100,
            reduction="sum",
        )
    denominator = (
        max(1, num_items_in_batch)
        if num_items_in_batch is not None
        else torch.count_nonzero(flat_labels != -100)
    )
    return total / denominator


class APTGlm45RoutedExpert(Glm45RoutedExpert):
    """Differentiable CPU/GPU bridge used by the real APTMoE runtime."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.on_GPU:
            return super().forward(hidden_states)
        gpu_device = hidden_states.device
        cpu_states = OffloadInputBegin.apply(
            hidden_states, gpu_device, self.layer_id, self.expert_id
        )
        cpu_output = super().forward(cpu_states)
        return OffloadInputEnd.apply(
            cpu_output, gpu_device, self.layer_id, self.expert_id
        )


class APTGlm45Router(nn.Module):
    def __init__(
        self,
        config: Glm4MoeConfig,
        layer_idx: int,
        routes: RouteController,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        nn.init.normal_(self.weight, std=config.initializer_range)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
        )
        self.layer_idx = layer_idx
        self.routes = routes
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.on_GPU = False
        self.last_scores: torch.Tensor | None = None
        self.last_indices: torch.Tensor | None = None
        self.last_counts: list[int] | None = None

    def forward(self, hidden_states: torch.Tensor) -> list[int]:
        logits = F.linear(
            hidden_states.float(),
            self.weight.float(),
        )
        scores, indices, counts = self.routes.select(
            layer_idx=self.layer_idx,
            logits=logits,
            correction_bias=self.e_score_correction_bias,
            norm_topk_prob=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        self.last_scores = scores
        self.last_indices = indices
        self.last_counts = counts
        return counts


class APTGlm45MoE(nn.Module):
    def __init__(
        self,
        *,
        config: Glm4MoeConfig,
        layer_idx: int,
        gate: APTGlm45Router,
        comm_scheduler: Any,
        placement_solver: ProxyPlacementSolver,
        comp_stream: torch.cuda.Stream,
        expert_events: list[torch.cuda.Event],
    ) -> None:
        super().__init__()
        self.layer_id = layer_idx
        self.gate = gate
        self.CommScheduler = comm_scheduler
        self.R_solver = placement_solver
        self.comp_stream = comp_stream
        self.expert_events = expert_events
        self.experts = nn.ModuleList(
            [
                APTGlm45RoutedExpert(
                    config.hidden_size,
                    config.moe_intermediate_size,
                    layer_id=layer_idx,
                    expert_id=expert_id,
                    device="cpu",
                    dtype=torch.bfloat16,
                )
                for expert_id in range(config.n_routed_experts)
            ]
        )
        self.shared_experts = Glm4MoeMLP(
            config,
            intermediate_size=(
                config.moe_intermediate_size * config.n_shared_experts
            ),
        )
        self.assigned_tokens_list = [0] * config.n_routed_experts
        self.historical_assigned_tokens_list = [0] * config.n_routed_experts

    def _queue_missing_hot_experts(self, counts: list[int]) -> None:
        hot = self.R_solver.solve(counts)
        self.CommScheduler.clear_priority(0)
        queued = 0
        for expert_id in hot:
            expert = self.experts[expert_id]
            if not expert.on_GPU:
                self.CommScheduler.add_model_to_queue(
                    expert, self.expert_events[expert_id], 0
                )
                queued += 1
        for _ in range(queued):
            self.CommScheduler.load_execute_with_priority()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        counts = self.gate(flat)
        scores = self.gate.last_scores
        indices = self.gate.last_indices
        if scores is None or indices is None:
            raise RuntimeError("router did not return top-k dispatch")
        for expert_id, count in enumerate(counts):
            self.assigned_tokens_list[expert_id] += count
        self._queue_missing_hot_experts(self.assigned_tokens_list)
        output = torch.zeros_like(flat)
        dispatch: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for expert_id in range(len(self.experts)):
            positions = torch.nonzero(indices == expert_id, as_tuple=False)
            if positions.numel():
                dispatch.append((expert_id, positions[:, 0], positions[:, 1]))
        for execute_on_gpu in (True, False):
            for expert_id, token_ids, slots in dispatch:
                expert = self.experts[expert_id]
                if expert.on_GPU is not execute_on_gpu:
                    continue
                if execute_on_gpu:
                    self.comp_stream.wait_event(self.expert_events[expert_id])
                value = expert(flat.index_select(0, token_ids))
                value = value * scores[token_ids, slots].unsqueeze(-1)
                output.index_add_(0, token_ids, value.to(output.dtype))
        output = output + self.shared_experts(flat)
        return output.reshape_as(hidden_states)


class APTGlm45DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        config: Glm4MoeConfig,
        layer_idx: int,
        routes: RouteController,
        comm_scheduler: Any,
        placement_solver: ProxyPlacementSolver,
    ) -> None:
        super().__init__()
        self.layer_id = layer_idx
        self.comp_stream = torch.cuda.Stream()
        self.self_attn = Glm45TokenMixer(config, layer_idx)
        self.norm1 = Glm4MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm2 = Glm4MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.dropout = nn.Identity()
        self.is_dense = layer_idx < config.first_k_dense_replace
        self.gate: APTGlm45Router | None = None
        self.expert_events: list[torch.cuda.Event] = []
        if self.is_dense:
            self.dense_mlp: nn.Module | None = Glm4MoeMLP(config)
            self.moe_layer: APTGlm45MoE | None = None
        else:
            self.dense_mlp = None
            self.gate = APTGlm45Router(config, layer_idx, routes)
            self.expert_events = [
                torch.cuda.Event() for _ in range(config.n_routed_experts)
            ]
            self.moe_layer = APTGlm45MoE(
                config=config,
                layer_idx=layer_idx,
                gate=self.gate,
                comm_scheduler=comm_scheduler,
                placement_solver=placement_solver,
                comp_stream=self.comp_stream,
                expert_events=self.expert_events,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn(self.norm1(hidden_states))
        residual = hidden_states
        normalized = self.norm2(hidden_states)
        transformed = (
            self.dense_mlp(normalized)
            if self.is_dense
            else self.moe_layer(normalized)
        )
        return residual + transformed


class APTGlm45Stage(nn.Sequential):
    def __init__(
        self,
        *,
        config: Glm4MoeConfig,
        stage_id: int,
        routes: RouteController,
        comm_scheduler: Any,
        placement_solver: ProxyPlacementSolver,
    ) -> None:
        layer = APTGlm45DecoderLayer(
            config=config,
            layer_idx=stage_id,
            routes=routes,
            comm_scheduler=comm_scheduler,
            placement_solver=placement_solver,
        )
        super().__init__(layer)
        self.comp_stream = layer.comp_stream
        self.comm_scheduler = comm_scheduler
        self.R_solver = placement_solver
        self.stage_id = stage_id
        self.embed_tokens: nn.Embedding | None = None
        self.final_norm: nn.Module | None = None
        self.lm_head: nn.Linear | None = None
        if stage_id == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if stage_id == config.num_hidden_layers - 1:
            self.final_norm = Glm4MoeRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
        self.to(device="cpu", dtype=torch.bfloat16)

    def __iter__(self) -> Iterator[APTGlm45DecoderLayer]:
        return iter([self._modules["0"]])

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if self.embed_tokens is not None and hidden_states.dtype == torch.long:
            hidden_states = self.embed_tokens(hidden_states)
        hidden_states = self._modules["0"](hidden_states)
        if self.lm_head is None:
            return hidden_states
        assert self.final_norm is not None
        hidden_states = self.final_norm(hidden_states)
        if labels is None:
            return self.lm_head(hidden_states)
        return _chunked_causal_lm_loss(
            self.lm_head,
            hidden_states,
            labels,
            num_items_in_batch=num_items_in_batch,
        )


class Glm45ModelShard(ModelShard):
    def _load_modules(
        self,
        use_historical: bool,
    ) -> tuple[list[nn.Module], dict[int, list[int]]]:
        modules = list(self._get_extra_modules())
        hot_by_layer: dict[int, list[int]] = {}
        for layer in self.model_shard:
            modules.extend((layer.self_attn, layer.norm1, layer.norm2))
            if layer.is_dense:
                assert layer.dense_mlp is not None
                modules.append(layer.dense_mlp)
                continue
            assert layer.gate is not None and layer.moe_layer is not None
            modules.extend((layer.gate, layer.moe_layer.shared_experts))
            counts = (
                layer.moe_layer.historical_assigned_tokens_list
                if use_historical
                else layer.moe_layer.assigned_tokens_list
            )
            hot = self.R_solver.solve(list(counts))
            hot_by_layer[layer.layer_id] = hot
            modules.extend(layer.moe_layer.experts[index] for index in hot)
        return modules, hot_by_layer

    def _drop_loaded_modules(self) -> None:
        """Drop every module that is actually resident on this rank's GPU.

        Forward preloads experts from the previous route and may then load a
        different set after the current router runs.  Recomputing only the
        current hot set here leaves mispredicted preload experts on CUDA.  The
        optimizer subsequently creates or moves Adam state beside those
        leaked parameters.  Base modules are always included, while experts
        are selected from their authoritative ``on_GPU`` residency flag.
        """
        modules = list(self._get_extra_modules())
        for layer in self.model_shard:
            modules.extend((layer.self_attn, layer.norm1, layer.norm2))
            if layer.is_dense:
                assert layer.dense_mlp is not None
                modules.append(layer.dense_mlp)
                continue
            assert layer.gate is not None and layer.moe_layer is not None
            modules.extend((layer.gate, layer.moe_layer.shared_experts))
            modules.extend(
                expert
                for expert in layer.moe_layer.experts
                if expert.on_GPU
            )
        with torch.cuda.stream(self.CommScheduler.drop_stream):
            torch.cuda.current_stream().wait_event(self.StageCompEvent)
            for module in modules:
                self.CommScheduler.drop(module)
            self._StageDropEvent.record(torch.cuda.current_stream())

    def FwdStageLoad(self, chunk_id=None, num_chunks=None, sft_mode=False):
        del num_chunks, sft_mode
        if chunk_id != 0:
            return
        modules, hot_by_layer = self._load_modules(use_historical=True)
        self.CommScheduler.load_execute(
            modules,
            waitEvent=self._StageDropEvent,
            recordEvent=self.StageLoadEvent,
        )
        with torch.cuda.stream(self.CommScheduler.load_stream):
            for layer in self.model_shard:
                for expert_id in hot_by_layer.get(layer.layer_id, []):
                    layer.expert_events[expert_id].record()

    def FwdStageDrop(
        self, chunk_id=None, num_chunks=None, fwd_only=False, sft_mode=False
    ):
        del sft_mode
        if chunk_id != (num_chunks or 1) - 1:
            return
        for layer in self.model_shard:
            if layer.moe_layer is not None:
                layer.moe_layer.historical_assigned_tokens_list = list(
                    layer.moe_layer.assigned_tokens_list
                )
        self._drop_loaded_modules()
        if fwd_only:
            self._clear_counts()

    def BwdStageLoad(self, chunk_id=None, num_chunks=None, sft_mode=False):
        del num_chunks, sft_mode
        if chunk_id != 0:
            return
        modules, _ = self._load_modules(use_historical=False)
        self.CommScheduler.load_execute(
            modules,
            waitEvent=self._StageDropEvent,
            recordEvent=self.StageLoadEvent,
        )

    def BwdStageDrop(self, chunk_id=None, num_chunks=None, sft_mode=False):
        del sft_mode
        if chunk_id != (num_chunks or 1) - 1:
            return
        self._drop_loaded_modules()
        self._clear_counts()

    def _clear_counts(self) -> None:
        for layer in self.model_shard:
            if layer.moe_layer is not None:
                layer.moe_layer.assigned_tokens_list[:] = [0] * len(
                    layer.moe_layer.assigned_tokens_list
                )


def parameter_category(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "final_norm" in name or ".norm1." in name or ".norm2." in name:
        return "norm"
    if ".self_attn." in name:
        return "token_mixer"
    if ".dense_mlp." in name:
        return "dense_mlp"
    if ".moe_layer.experts." in name:
        return "routed_experts"
    if ".moe_layer.shared_experts." in name:
        return "shared_expert"
    if ".gate.weight" in name:
        return "router"
    return "other"


def categorized_parameter_counts(module: nn.Module) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, parameter in module.named_parameters():
        category = parameter_category(name)
        result[category] = result.get(category, 0) + parameter.numel()
    return result
