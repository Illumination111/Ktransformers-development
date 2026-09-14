#!/usr/bin/env python3
"""GLM-4.5-Air APTMoE component-isomorphic deployment-proxy contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPECTED_PARAMETERS = 106_852_245_504
EXPECTED_MODEL_TYPE = "glm4_moe"


@dataclass(frozen=True)
class Component:
    parameters: int
    bf16_bytes: int

    @classmethod
    def from_parameters(cls, count: int) -> "Component":
        return cls(count, count * 2)


def load_raw_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / "config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"expected model_type={EXPECTED_MODEL_TYPE!r}, got {raw.get('model_type')!r}"
        )
    return raw


def _positive_int(raw: dict[str, Any], name: str) -> int:
    value = raw.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config.{name} must be a positive integer, got {value!r}")
    return value


def component_counts(raw: dict[str, Any]) -> dict[str, int]:
    hidden = _positive_int(raw, "hidden_size")
    layers = _positive_int(raw, "num_hidden_layers")
    dense_layers = _positive_int(raw, "first_k_dense_replace")
    experts = _positive_int(raw, "n_routed_experts")
    expert_intermediate = _positive_int(raw, "moe_intermediate_size")
    dense_intermediate = _positive_int(raw, "intermediate_size")
    shared_experts = _positive_int(raw, "n_shared_experts")
    vocab = _positive_int(raw, "vocab_size")
    heads = _positive_int(raw, "num_attention_heads")
    kv_heads = _positive_int(raw, "num_key_value_heads")
    head_dim = _positive_int(raw, "head_dim")
    moe_layers = layers - dense_layers
    if moe_layers <= 0:
        raise ValueError("config must contain at least one MoE layer")

    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    attention_each = (
        hidden * (q_dim + 2 * kv_dim)
        + q_dim * hidden
        + (q_dim + 2 * kv_dim if raw.get("attention_bias", False) else 0)
    )
    return {
        "embedding": vocab * hidden,
        "lm_head": vocab * hidden,
        "norm": (2 * layers + 1) * hidden,
        "token_mixer": layers * attention_each,
        "dense_mlp": dense_layers * 3 * hidden * dense_intermediate,
        "router": moe_layers * hidden * experts,
        "routed_experts": moe_layers
        * experts
        * 3
        * hidden
        * expert_intermediate,
        "shared_expert": moe_layers
        * shared_experts
        * 3
        * hidden
        * expert_intermediate,
    }


def build_manifest(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).resolve()
    raw = load_raw_config(path)
    counts = component_counts(raw)
    total = sum(counts.values())
    if total != EXPECTED_PARAMETERS:
        raise ValueError(
            f"config-derived parameters={total:,}, expected={EXPECTED_PARAMETERS:,}"
        )
    layers = _positive_int(raw, "num_hidden_layers")
    dense_layers = _positive_int(raw, "first_k_dense_replace")
    expert_parameters = (
        3
        * _positive_int(raw, "hidden_size")
        * _positive_int(raw, "moe_intermediate_size")
    )
    components = {
        name: asdict(Component.from_parameters(count))
        for name, count in counts.items()
    }
    components["model_total"] = asdict(Component.from_parameters(total))
    return {
        "schema_version": 1,
        "benchmark_class": "deployment_proxy",
        "proxy_architecture": "glm45_air_component_isomorphic",
        "target": {
            "model_path": str(path),
            "architecture": "Glm4MoeForCausalLM",
            "model_type": raw["model_type"],
            "precision": "bf16",
            "num_hidden_layers": layers,
            "dense_layers": dense_layers,
            "dense_layer_indices": list(range(dense_layers)),
            "moe_layers": layers - dense_layers,
            "moe_layer_indices": list(range(dense_layers, layers)),
            "hidden_size": raw["hidden_size"],
            "intermediate_size": raw["intermediate_size"],
            "moe_intermediate_size": raw["moe_intermediate_size"],
            "n_routed_experts": raw["n_routed_experts"],
            "num_experts_per_tok": raw["num_experts_per_tok"],
            "n_shared_experts": raw["n_shared_experts"],
            "components": components,
        },
        "required_proxy_contract": {
            "weight_source": "deterministic_random_bf16_initialization",
            "checkpoint_compatible": False,
            "exact_model_claim_allowed": False,
            "model_quality_claim_allowed": False,
            "real_forward_backward_optimizer_update_required": True,
            "token_mixer": "Transformers Glm4MoeAttention with Glm4MoeRotaryEmbedding",
            "normalization": "Transformers Glm4MoeRMSNorm",
            "dense_layer": "Transformers Glm4MoeMLP",
            "routed_expert_home_device": "cpu",
            "routed_expert_transfer_granularity": "one expert",
            "routed_expert": {
                "count_per_moe_layer": raw["n_routed_experts"],
                "total_count": (layers - dense_layers) * raw["n_routed_experts"],
                "parameters_each": expert_parameters,
                "bf16_bytes_each": expert_parameters * 2,
            },
            "routing": {
                "activation": "sigmoid",
                "top_k": raw["num_experts_per_tok"],
                "norm_topk_prob": raw["norm_topk_prob"],
                "synthetic_fallback_validity": "SMOKE_ONLY",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/data2/models/GLM-4.5-Air"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.model_path)
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
