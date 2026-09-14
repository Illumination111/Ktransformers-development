#!/usr/bin/env python3
"""Lightweight meta/audit tests for the GLM-4.5-Air proxy."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from aggregate_sweep_results import collect_case
from glm45_air_aptmoe_proxy_components import (
    Glm45RoutedExpert,
    component_parameter_counts,
    load_glm_config,
)
from glm45_air_proxy_spec import (
    EXPECTED_PARAMETERS,
    build_manifest,
    component_counts,
    load_raw_config,
)
from glm45_aptmoe_proxy import ProxyPlacementSolver, RouteController
from glm45_aptmoe_proxy.placement import EXPECTED_EXPERT_BF16_BYTES

MODEL_PATH = Path("/mnt/data2/models/GLM-4.5-Air")


@unittest.skipUnless(MODEL_PATH.is_dir(), "GLM-4.5-Air config unavailable")
class ConfigAndComponentAuditTest(unittest.TestCase):
    def test_config_maps_required_topology_and_total(self) -> None:
        raw = load_raw_config(MODEL_PATH)
        self.assertEqual(raw["num_hidden_layers"], 46)
        self.assertEqual(raw["first_k_dense_replace"], 1)
        self.assertEqual(raw["n_routed_experts"], 128)
        self.assertEqual(raw["num_experts_per_tok"], 8)
        self.assertEqual(raw["hidden_size"], 4096)
        self.assertEqual(raw["moe_intermediate_size"], 1408)
        self.assertEqual(raw["n_shared_experts"], 1)
        self.assertEqual(sum(component_counts(raw).values()), EXPECTED_PARAMETERS)
        manifest = build_manifest(MODEL_PATH)
        self.assertEqual(manifest["target"]["dense_layer_indices"], [0])
        self.assertEqual(len(manifest["target"]["moe_layer_indices"]), 45)
        self.assertFalse(
            manifest["required_proxy_contract"]["exact_model_claim_allowed"]
        )

    def test_dense_and_moe_meta_shapes_use_transformers_components(self) -> None:
        config = load_glm_config(MODEL_PATH)
        dense = component_parameter_counts(config, 0)
        moe = component_parameter_counts(config, 1)
        self.assertEqual(dense["dense_mlp"], 134_479_872)
        self.assertEqual(dense["routed_experts"], 0)
        self.assertEqual(moe["dense_mlp"], 0)
        self.assertEqual(moe["router"], 524_288)
        self.assertEqual(moe["routed_experts"], 2_214_592_512)
        self.assertEqual(moe["shared_expert"], 17_301_504)
        self.assertEqual(dense["token_mixer"], 109_066_240)
        self.assertEqual(moe["token_mixer"], 109_066_240)
        self.assertEqual(dense["norms"], 8192)

    def test_expert_fused_layout_and_bf16_transfer_size(self) -> None:
        expert = Glm45RoutedExpert(
            4096,
            1408,
            layer_id=1,
            expert_id=0,
            device="meta",
            dtype=torch.bfloat16,
        )
        shapes = {
            name: tuple(parameter.shape)
            for name, parameter in expert.named_parameters()
        }
        self.assertEqual(
            shapes,
            {
                "gate_up_proj.weight": (2816, 4096),
                "down_proj.weight": (4096, 1408),
            },
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in expert.parameters()) * 2,
            EXPECTED_EXPERT_BF16_BYTES,
        )


class SmokeGuardTest(unittest.TestCase):
    def test_synthetic_routing_requires_explicit_flag(self) -> None:
        kwargs = {
            "num_layers": 46,
            "first_moe_layer": 1,
            "num_experts": 128,
            "top_k": 8,
            "sequence_length": 32,
            "tokens_per_microbatch": 256,
            "trace_path": None,
        }
        with self.assertRaisesRegex(ValueError, "SMOKE_ONLY"):
            RouteController(**kwargs, allow_synthetic=False)
        routes = RouteController(**kwargs, allow_synthetic=True)
        self.assertEqual(routes.mode, "synthetic_router_smoke_only")

    def test_unprofiled_placement_is_smoke_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "SMOKE_ONLY"):
            ProxyPlacementSolver(
                128,
                1,
                lookup_path=None,
                prefetch_portion=0.1,
                allow_unprofiled=False,
                expected_profile="server",
                required_max_tokens=256,
            )
        solver = ProxyPlacementSolver(
            128,
            1,
            lookup_path=None,
            prefetch_portion=0.1,
            allow_unprofiled=True,
            expected_profile="server",
            required_max_tokens=256,
        )
        self.assertEqual(solver.manifest()["mode"], "unprofiled_fraction_smoke_only")
        self.assertEqual(len(solver.solve([0] * 128)), 13)

    def test_aggregator_preserves_smoke_only_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "server_8gpu_batch8" / "seq_32"
            timing = run / "step_timing"
            timing.mkdir(parents=True)
            config = {
                "backend": "aptmoe",
                "profile": "server",
                "benchmark_class": "deployment_proxy",
                "result_validity": "SMOKE_ONLY",
                "weight_source": "deterministic_random_bf16_initialization",
                "checkpoint_compatible": False,
                "exact_model_claim_allowed": False,
                "model_load_architecture": (
                    "Glm45AirComponentIsomorphicAPTMoEProxy"
                ),
                "precision": "bf16",
                "sequence_length": 32,
                "steps": 2,
                "warmup_steps": 1,
            }
            (run / "run_config.json").write_text(json.dumps(config))
            (run / "exit_code.txt").write_text("0\n")
            (timing / "step_timing.json").write_text(
                json.dumps(
                    {
                        "num_stable_steps": 1,
                        "aggregate_stable": {
                            key: {"mean_sec": 1.0}
                            for key in (
                                "step_total_sec",
                                "forward_sec",
                                "backward_sec",
                                "optimizer_sec",
                            )
                        },
                        "tps_attribution": {
                            "mean_stable_step_sec": 1.0,
                            "stable_tps": 256.0,
                        },
                    }
                )
            )
            (run / "proxy_manifest.json").write_text(
                json.dumps(
                    {
                        "benchmark_class": "deployment_proxy",
                        "proxy_architecture": "glm45_air_component_isomorphic",
                        "parameter_count": EXPECTED_PARAMETERS,
                        "checkpoint_compatible": False,
                    }
                )
            )
            (run / "full_update_verification.json").write_text(
                json.dumps({"valid_full_update": True})
            )
            row = collect_case(run / "run_config.json")
            self.assertEqual(row["status"], "SMOKE_ONLY")
            self.assertTrue(row["full_update_verified"])


if __name__ == "__main__":
    unittest.main()
