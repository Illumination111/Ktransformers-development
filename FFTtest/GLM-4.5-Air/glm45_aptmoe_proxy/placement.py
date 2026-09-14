"""Host-profiled APTMoE placement for GLM-4.5-Air experts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_EXPERT_BF16_BYTES = 34_603_008


class ProxyPlacementSolver:
    def __init__(
        self,
        num_experts: int,
        num_chunks: int,
        *,
        lookup_path: str | Path | None,
        prefetch_portion: float,
        allow_unprofiled: bool,
        expected_profile: str,
        required_max_tokens: int,
    ) -> None:
        if not 0 < prefetch_portion <= 1:
            raise ValueError("prefetch_portion must be in (0, 1]")
        self.num_experts = num_experts
        self.num_chunks = num_chunks
        self.prefetch_portion = prefetch_portion
        self.expected_profile = expected_profile
        self.required_max_tokens = required_max_tokens
        self.lookup_path = Path(lookup_path).resolve() if lookup_path else None
        self.lookup: dict[str, Any] | None = None
        self.lookup_sha256: str | None = None
        self.mode = "unprofiled_fraction_smoke_only"
        if self.lookup_path:
            payload = self.lookup_path.read_bytes()
            self.lookup = json.loads(payload)
            self.lookup_sha256 = hashlib.sha256(payload).hexdigest()
            self._validate()
            self.mode = "profiled_compute_load"
        elif not allow_unprofiled:
            raise ValueError(
                "formal proxy requires a host lookup table; "
                "use --allow-unprofiled-placement only for SMOKE_ONLY"
            )

    def _positive(self, mapping: dict[str, Any], name: str) -> float:
        value = mapping.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"lookup {name} must be positive")
        return float(value)

    def _validate(self) -> None:
        assert self.lookup is not None
        if self.lookup.get("schema_version") != 1:
            raise ValueError("lookup schema_version must be 1")
        if self.lookup.get("benchmark_class") != "aptmoe_glm45_air_proxy_lookup":
            raise ValueError("lookup benchmark_class mismatch")
        if self.lookup.get("deployment_profile") != self.expected_profile:
            raise ValueError("lookup deployment_profile mismatch")
        expert = self.lookup.get("expert") or {}
        if expert.get("bf16_bytes") != EXPECTED_EXPERT_BF16_BYTES:
            raise ValueError("lookup routed-expert BF16 size mismatch")
        if expert.get("num_experts") != self.num_experts:
            raise ValueError("lookup routed-expert count mismatch")
        self._positive(expert, "h2d_seconds")
        control = self.lookup.get("control_plane") or {}
        self._positive(control, "non_expert_h2d_seconds")
        cpu = self.lookup.get("cpu_expert") or {}
        max_tokens = cpu.get("max_tokens")
        curve = cpu.get("forward_seconds_by_tokens")
        if (
            not isinstance(max_tokens, int)
            or max_tokens < self.required_max_tokens
            or not isinstance(curve, list)
            or len(curve) != max_tokens + 1
            or any(
                not isinstance(value, (int, float)) or value < 0
                for value in curve
            )
        ):
            raise ValueError("lookup CPU expert curve does not cover this run")

    def solve(
        self,
        assigned_tokens_list: list[int],
        **_: Any,
    ) -> list[int]:
        if len(assigned_tokens_list) != self.num_experts:
            raise ValueError("expert count vector length mismatch")
        ordered = sorted(
            range(self.num_experts),
            key=lambda index: assigned_tokens_list[index],
        )
        if self.lookup is None:
            count = max(1, math.ceil(self.num_experts * self.prefetch_portion))
            return ordered[-count:]
        curve = self.lookup["cpu_expert"]["forward_seconds_by_tokens"]
        load = float(self.lookup["control_plane"]["non_expert_h2d_seconds"])
        load_expert = float(self.lookup["expert"]["h2d_seconds"])
        cpu_seconds = 0.0
        cpu_experts: list[int] = []
        for expert_id in ordered:
            tokens = min(assigned_tokens_list[expert_id], len(curve) - 1)
            cpu_seconds += float(curve[tokens])
            load += load_expert
            if cpu_seconds / load < 1.0:
                cpu_experts.append(expert_id)
            else:
                break
        cpu_set = set(cpu_experts)
        return [expert_id for expert_id in ordered if expert_id not in cpu_set]

    def manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "lookup_path": str(self.lookup_path) if self.lookup_path else None,
            "lookup_sha256": self.lookup_sha256,
            "deployment_profile": self.expected_profile,
            "num_experts": self.num_experts,
            "prefetch_portion": self.prefetch_portion,
            "required_max_tokens": self.required_max_tokens,
        }
