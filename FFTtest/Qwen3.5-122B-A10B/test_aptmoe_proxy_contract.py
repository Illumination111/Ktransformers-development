#!/usr/bin/env python3
"""CPU-only contract tests for the Qwen3.5-122B APTMoE proxy."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "Qwen3.5-35B-A3B"
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_sweep_results import (
    EXPECTED_PROXY_SHAPE,
    collect_case,
)
from aptmoe_proxy.placement import ProxyPlacementSolver
from aptmoe_proxy.routes import RouteController
from qwen35_proxy_spec import build_manifest


MODEL_PATH = Path("/mnt/data2/models/Qwen3.5-122B-A10B")
EXPECTED_PARAMETERS = 122_111_526_912
EXPECTED_EXPERT_BF16_BYTES = 18 * (1 << 20)


@unittest.skipUnless(MODEL_PATH.is_dir(), "local 122B config unavailable")
class ModelConfigAuditTest(unittest.TestCase):
    def test_shape_and_parameters_are_derived_from_config(self) -> None:
        manifest = build_manifest(
            MODEL_PATH,
            None,
            proxy_tag="qwen35_122b",
        )
        target = manifest["target"]
        self.assertEqual(
            {key: target[key] for key in EXPECTED_PROXY_SHAPE},
            EXPECTED_PROXY_SHAPE,
        )
        self.assertEqual(
            target["components"]["model_total"]["parameters"],
            EXPECTED_PARAMETERS,
        )
        expert = manifest["required_proxy_contract"]["routed_expert"]
        self.assertEqual(expert["bf16_bytes_each"], EXPECTED_EXPERT_BF16_BYTES)
        self.assertEqual(expert["total_count"], 48 * 256)


class RoutingAndPlacementAuditTest(unittest.TestCase):
    def test_synthetic_routing_has_smoke_only_mode(self) -> None:
        controller = RouteController(
            num_layers=48,
            num_experts=256,
            top_k=8,
            sequence_length=32,
            tokens_per_microbatch=256,
            trace_path=None,
            allow_synthetic=True,
            formal_trace_source=(
                "merged_exact_qwen35_122b_router_trace"
            ),
            formal_replay_mode="replayed_qwen35_122b_topk_indices",
        )
        self.assertEqual(controller.mode, "random_router_synthetic")

    def test_lookup_validates_dynamic_18_mib_expert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lookup_path = Path(directory) / "lookup.json"
            lookup_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_class": (
                            "aptmoe_qwen35_122b_proxy_lookup"
                        ),
                        "deployment_profile": "server",
                        "expert": {
                            "bf16_bytes": EXPECTED_EXPERT_BF16_BYTES,
                            "num_experts": 256,
                            "h2d_seconds": 1.0,
                        },
                        "control_plane": {
                            "load_seconds": 1.0,
                            "non_mixer_load_seconds": 0.5,
                        },
                        "token_mixers": {
                            "linear_attention": {"h2d_seconds": 0.5},
                            "full_attention": {"h2d_seconds": 0.5},
                        },
                        "extra_modules": {
                            "embedding_h2d_seconds": 1.0,
                            "final_norm_h2d_seconds": 0.1,
                            "lm_head_h2d_seconds": 1.0,
                        },
                        "cpu_expert": {
                            "max_tokens": 1,
                            "forward_seconds_by_tokens": [0.0, 0.5],
                        },
                    }
                ),
                encoding="utf-8",
            )
            solver = ProxyPlacementSolver(
                256,
                1,
                lookup_path=lookup_path,
                prefetch_portion=0.6,
                allow_unprofiled=False,
                expected_profile="server",
                expected_expert_bf16_bytes=EXPECTED_EXPERT_BF16_BYTES,
                expected_benchmark_class=(
                    "aptmoe_qwen35_122b_proxy_lookup"
                ),
            )
            self.assertEqual(solver.mode, "profiled_compute_load")


class AggregationGuardTest(unittest.TestCase):
    def test_synthetic_unprofiled_result_is_smoke_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "server_8gpu_batch8" / "seq_32"
            timing_dir = run_dir / "step_timing"
            timing_dir.mkdir(parents=True)
            config = {
                "backend": "aptmoe",
                "profile": "server",
                "benchmark_class": "deployment_proxy",
                "result_validity": "smoke_only",
                "result_marker": "SMOKE_ONLY",
                "formal_claim_allowed": False,
                "target_model": "Qwen3.5-122B-A10B",
                "proxy_architecture": (
                    "qwen35_122b_component_isomorphic"
                ),
                "model_load_architecture": (
                    "Qwen35_122BComponentIsomorphicAPTMoEProxy"
                ),
                "expected_text_parameters": EXPECTED_PARAMETERS,
                "model_shape": EXPECTED_PROXY_SHAPE,
                "weight_source": "deterministic_random_initialization",
                "checkpoint_compatible": False,
                "llamafactory_backend": False,
                "precision": "bf16",
                "steps": 2,
                "warmup_steps": 1,
            }
            manifest = {
                "benchmark_class": "deployment_proxy",
                "result_validity": "smoke_only",
                "result_marker": "SMOKE_ONLY",
                "formal_claim_allowed": False,
                "proxy_architecture": (
                    "qwen35_122b_component_isomorphic"
                ),
                "parameter_count": EXPECTED_PARAMETERS,
                "model_shape": {
                    **EXPECTED_PROXY_SHAPE,
                    "expert_bf16_bytes": EXPECTED_EXPERT_BF16_BYTES,
                },
                "checkpoint_compatible": False,
                "real_forward_backward_optimizer_update": True,
                "route": {"mode": "random_router_synthetic"},
                "placement": {"mode": "unprofiled_fraction"},
            }
            timing = {
                "timing_mode": "coarse_host_wall_no_cuda_sync",
                "num_stable_steps": 1,
                "aggregate_stable": {
                    key: {"mean_sec": 1.0}
                    for key in (
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
            verification = {
                "valid_full_update": True,
                "optimizer_parameter_count": EXPECTED_PARAMETERS,
            }
            for name, value in (
                ("run_config.json", config),
                ("proxy_manifest.json", manifest),
                ("full_update_verification.json", verification),
            ):
                (run_dir / name).write_text(
                    json.dumps(value),
                    encoding="utf-8",
                )
            (timing_dir / "step_timing.json").write_text(
                json.dumps(timing),
                encoding="utf-8",
            )
            (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
            row = collect_case(run_dir / "run_config.json")
            self.assertEqual(row["status"], "SMOKE_ONLY")
            self.assertFalse(row["formal_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
