"""Real Transformers GLM shapes used by the APTMoE deployment proxy."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeMLP,
    Glm4MoeRMSNorm,
    Glm4MoeRotaryEmbedding,
)


def load_glm_config(model_path: str | Path) -> Glm4MoeConfig:
    config = Glm4MoeConfig.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    if config.model_type != "glm4_moe":
        raise ValueError(f"not a GLM-4-MoE config: {model_path}")
    config._attn_implementation = "sdpa"
    return config


class Glm45TokenMixer(nn.Module):
    """One unmodified Transformers GLM attention/rotary pair."""

    def __init__(self, config: Glm4MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = "full_attention"
        self.impl = Glm4MoeAttention(config, layer_idx)
        self.rotary = Glm4MoeRotaryEmbedding(config)
        self.on_GPU = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        position_ids = (
            torch.arange(sequence, device=hidden_states.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        position_embeddings = self.rotary(hidden_states, position_ids)
        output, _ = self.impl(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        return output


class Glm45RoutedExpert(nn.Module):
    """One independently movable GLM fused gate/up routed expert."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        layer_id: int,
        expert_id: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.gate_up_proj = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            **kwargs,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            **kwargs,
        )
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.on_GPU = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Glm45ProxyLayerComponents(nn.Module):
    """Meta-device component audit without allocating the 100B proxy."""

    def __init__(
        self,
        config: Glm4MoeConfig,
        layer_idx: int,
        *,
        device: str | torch.device = "meta",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        with torch.device(device):
            self.token_mixer = Glm45TokenMixer(config, layer_idx)
            self.input_layernorm = Glm4MoeRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = Glm4MoeRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            if layer_idx < config.first_k_dense_replace:
                self.dense_mlp = Glm4MoeMLP(config)
                self.router = None
                self.experts = None
                self.shared_expert = None
            else:
                self.dense_mlp = None
                self.router = nn.Linear(
                    config.hidden_size,
                    config.n_routed_experts,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                self.experts = nn.ModuleList(
                    [
                        Glm45RoutedExpert(
                            config.hidden_size,
                            config.moe_intermediate_size,
                            layer_id=layer_idx,
                            expert_id=expert_id,
                            device=device,
                            dtype=dtype,
                        )
                        for expert_id in range(config.n_routed_experts)
                    ]
                )
                self.shared_expert = Glm4MoeMLP(
                    config,
                    intermediate_size=(
                        config.moe_intermediate_size * config.n_shared_experts
                    ),
                )
        if str(device) != "meta":
            self.to(device=device, dtype=dtype)


def component_parameter_counts(
    config: Glm4MoeConfig,
    layer_idx: int,
) -> dict[str, int]:
    layer = Glm45ProxyLayerComponents(config, layer_idx, device="meta")

    def count(module: nn.Module | None) -> int:
        return (
            0
            if module is None
            else sum(parameter.numel() for parameter in module.parameters())
        )

    result = {
        "token_mixer": count(layer.token_mixer),
        "norms": count(layer.input_layernorm)
        + count(layer.post_attention_layernorm),
        "dense_mlp": count(layer.dense_mlp),
        "router": count(layer.router),
        "routed_experts": count(layer.experts),
        "shared_expert": count(layer.shared_expert),
    }
    result["total"] = sum(result.values())
    return result
