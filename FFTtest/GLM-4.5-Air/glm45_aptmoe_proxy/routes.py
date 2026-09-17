"""Replay or synthesize GLM-4.5-Air top-8 routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class RouteController:
    def __init__(
        self,
        *,
        num_layers: int,
        first_moe_layer: int,
        num_experts: int,
        top_k: int,
        sequence_length: int,
        tokens_per_microbatch: int,
        microbatches_per_step: int = 1,
        expected_patterns: int | None = None,
        trace_path: str | Path | None,
        allow_synthetic: bool,
    ) -> None:
        self.num_layers = num_layers
        self.first_moe_layer = first_moe_layer
        self.num_moe_layers = num_layers - first_moe_layer
        self.num_experts = num_experts
        self.top_k = top_k
        self.sequence_length = sequence_length
        self.tokens_per_microbatch = tokens_per_microbatch
        self.microbatches_per_step = microbatches_per_step
        self.expected_patterns = expected_patterns
        self.step = 0
        self.microbatch = 0
        self._cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self._last_counts: dict[int, list[int]] = {}
        self.trace_path = self._resolve_trace(trace_path)
        self.trace: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.trace_sha256: str | None = None
        if self.trace_path:
            self._load(self.trace_path)
            if self.metadata.get("source") == "merged_exact_glm45_air_router_trace":
                self.mode = "replayed_glm45_air_topk_indices"
            elif allow_synthetic:
                self.mode = "synthetic_trace_smoke_only"
            else:
                raise ValueError("formal proxy requires an exact GLM route trace")
        elif allow_synthetic:
            self.mode = "synthetic_router_smoke_only"
        else:
            raise ValueError(
                "formal proxy requires a route trace; "
                "use --allow-synthetic-routing only for SMOKE_ONLY"
            )

    def _resolve_trace(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            path = path / f"seq_{self.sequence_length}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _load(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            trace = np.asarray(data["topk_indices"])
            self.metadata = json.loads(str(data["metadata_json"].item()))
        expected_tail = (
            self.num_moe_layers,
            self.tokens_per_microbatch,
            self.top_k,
        )
        if trace.ndim == 3:
            valid_shape = trace.shape == expected_tail
        elif trace.ndim == 4:
            valid_shape = trace.shape[1:] == expected_tail
        else:
            valid_shape = False
        if not valid_shape:
            raise ValueError(
                f"route trace shape={trace.shape}, expected {expected_tail} "
                "or [patterns, ...]"
            )
        if trace.min() < 0 or trace.max() >= self.num_experts:
            raise ValueError("route trace contains out-of-range expert ids")
        if np.any(np.diff(np.sort(trace, axis=-1), axis=-1) == 0):
            raise ValueError("route trace repeats an expert for one token")
        expected_metadata = {
            "schema_version": 1,
            "sequence_length": self.sequence_length,
            "layers": self.num_moe_layers,
            "tokens": self.tokens_per_microbatch,
            "top_k": self.top_k,
            "global_batch_size": (
                self.tokens_per_microbatch // self.sequence_length
            ),
        }
        if trace.ndim == 4:
            expected_metadata["patterns"] = trace.shape[0]
            if (
                self.expected_patterns is not None
                and trace.shape[0] != self.expected_patterns
            ):
                raise ValueError(
                    f"route patterns={trace.shape[0]}, "
                    f"expected={self.expected_patterns}"
                )
        for key, expected in expected_metadata.items():
            if self.metadata.get(key) != expected:
                raise ValueError(
                    f"route metadata {key}={self.metadata.get(key)!r}, "
                    f"expected={expected!r}"
                )
        self.trace = trace.astype(np.int16, copy=False)
        self.trace_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def set_position(self, step: int, microbatch: int) -> None:
        self.step = step
        self.microbatch = microbatch

    def _pattern(self) -> int:
        if self.trace is None or self.trace.ndim == 3:
            return 0
        return (
            self.step * self.microbatches_per_step + self.microbatch
        ) % self.trace.shape[0]

    def select(
        self,
        *,
        layer_idx: int,
        logits: torch.Tensor,
        correction_bias: torch.Tensor,
        norm_topk_prob: bool,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        probabilities = torch.sigmoid(logits.float())
        choice = probabilities + correction_bias.float()
        computed_indices = torch.topk(
            choice, self.top_k, dim=-1, sorted=False
        ).indices
        if self.trace is None:
            indices = computed_indices
        else:
            moe_index = layer_idx - self.first_moe_layer
            key = (self._pattern(), moe_index, str(logits.device))
            indices = self._cache.get(key)
            if indices is None:
                source = (
                    self.trace[moe_index]
                    if self.trace.ndim == 3
                    else self.trace[self._pattern(), moe_index]
                )
                indices = torch.as_tensor(
                    source.astype(np.int64, copy=False),
                    device=logits.device,
                )
                self._cache[key] = indices
        weights = probabilities.gather(1, indices)
        if norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * routed_scaling_factor
        counts = torch.bincount(
            indices.reshape(-1), minlength=self.num_experts
        ).cpu().tolist()
        self._last_counts[layer_idx] = counts
        return weights.to(logits.dtype), indices, counts

    def predicted_counts(self, layer_idx: int) -> list[int]:
        return list(
            self._last_counts.get(
                min(layer_idx + 1, self.num_layers - 1),
                [0] * self.num_experts,
            )
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "trace_sha256": self.trace_sha256,
            "trace_metadata": self.metadata,
            "num_layers": self.num_layers,
            "num_moe_layers": self.num_moe_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "sequence_length": self.sequence_length,
            "tokens_per_microbatch": self.tokens_per_microbatch,
        }
