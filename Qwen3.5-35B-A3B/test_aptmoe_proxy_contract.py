#!/usr/bin/env python3
"""CPU-only contract tests for the Qwen3.5 APTMoE deployment proxy."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from aggregate_sweep_results import aggregate_run
from aptmoe_qwen35_proxy_train import _configure_distributed
from aptmoe_proxy.model import (
    _chunked_causal_lm_loss,
    _solve_stage_hot_experts,
)
from aptmoe_proxy.placement import (
    EXPECTED_EXPERT_BF16_BYTES,
    ProxyPlacementSolver,
)
from aptmoe_proxy.routes import RouteController
from aptmoe_proxy.storage import require_within_simulation_root
from qwen35_aptmoe_proxy_components import (
    Qwen35RoutedExpert,
    component_parameter_counts,
    load_text_config,
)
from merge_qwen35_route_traces import (
    group_route_patterns_by_optimizer_step,
)
from qwen35_route_capture import RouteTraceCapture, _install_kt_route_hooks


MODEL_PATH = Path("/mnt/data3/models/Qwen3.5-35B-A3B")


class PersistentDistributedContractTest(unittest.TestCase):
    def test_existing_process_group_is_not_reinitialized(self) -> None:
        args = Namespace(num_gpus=4, seed=42)
        with (
            patch.dict(os.environ, {"LOCAL_RANK": "2"}),
            patch(
                "aptmoe_qwen35_proxy_train.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "aptmoe_qwen35_proxy_train.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "aptmoe_qwen35_proxy_train.dist.init_process_group"
            ) as init_process_group,
            patch(
                "aptmoe_qwen35_proxy_train.dist.get_rank",
                return_value=2,
            ),
            patch(
                "aptmoe_qwen35_proxy_train.dist.get_world_size",
                return_value=4,
            ),
            patch("aptmoe_qwen35_proxy_train.torch.cuda.set_device"),
            patch("aptmoe_qwen35_proxy_train.torch.set_num_threads"),
        ):
            rank, world_size, local_rank, initialized_here = (
                _configure_distributed(args)
            )
        self.assertEqual((rank, world_size, local_rank), (2, 4, 2))
        self.assertFalse(initialized_here)
        init_process_group.assert_not_called()


class ComponentContractTest(unittest.TestCase):
    @unittest.skipUnless(MODEL_PATH.is_dir(), "local Qwen3.5 config unavailable")
    def test_representative_layer_parameter_counts(self) -> None:
        config = load_text_config(MODEL_PATH)
        linear = component_parameter_counts(config, 0)
        full = component_parameter_counts(config, 3)
        self.assertEqual(linear["token_mixer"], 33_718_464)
        self.assertEqual(full["token_mixer"], 27_263_488)
        for counts in (linear, full):
            self.assertEqual(counts["router"], 524_288)
            self.assertEqual(counts["routed_experts"], 805_306_368)
            self.assertEqual(counts["shared_expert_and_gate"], 3_147_776)
            self.assertEqual(counts["norms"], 4_096)

    def test_routed_expert_uses_target_fused_tensor_layout(self) -> None:
        expert = Qwen35RoutedExpert(
            2048,
            512,
            layer_id=0,
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
                "gate_up_proj.weight": (1024, 2048),
                "down_proj.weight": (2048, 512),
            },
        )


class RouteContractTest(unittest.TestCase):
    def _write_trace(self, path: Path, source: str) -> None:
        routes = np.empty((2, 3, 2), dtype=np.int16)
        routes[0] = [[0, 1], [1, 2], [2, 3]]
        routes[1] = [[3, 2], [2, 1], [1, 0]]
        metadata = {
            "schema_version": 1,
            "source": source,
            "source_backend": "kt",
            "sequence_length": 3,
            "global_batch_size": 1,
            "patterns": 1,
            "layers": 2,
            "tokens": 3,
            "top_k": 2,
        }
        np.savez_compressed(
            path,
            topk_indices=routes[None, ...],
            metadata_json=np.asarray(json.dumps(metadata)),
        )

    def test_exact_trace_replay_preserves_router_gradient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.npz"
            self._write_trace(trace, "merged_exact_qwen35_router_trace")
            controller = RouteController(
                num_layers=2,
                num_experts=4,
                top_k=2,
                sequence_length=3,
                tokens_per_microbatch=3,
                trace_path=trace,
                allow_synthetic=False,
            )
            logits = torch.randn(3, 4, requires_grad=True)
            scores, indices, counts = controller.select(
                layer_idx=0,
                logits=logits,
            )
            (scores[:, 0] * torch.arange(1, 4)).sum().backward()
            self.assertEqual(indices.tolist(), [[0, 1], [1, 2], [2, 3]])
            self.assertEqual(counts, [1, 2, 2, 1])
            self.assertIsNotNone(logits.grad)
            self.assertGreater(int(torch.count_nonzero(logits.grad)), 0)

    def test_synthetic_trace_requires_explicit_smoke_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.npz"
            self._write_trace(trace, "synthetic_zipf_smoke_only")
            arguments = {
                "num_layers": 2,
                "num_experts": 4,
                "top_k": 2,
                "sequence_length": 3,
                "tokens_per_microbatch": 3,
                "trace_path": trace,
            }
            with self.assertRaisesRegex(ValueError, "formal APTMoE"):
                RouteController(**arguments, allow_synthetic=False)
            controller = RouteController(
                **arguments,
                allow_synthetic=True,
            )
            self.assertEqual(controller.mode, "synthetic_trace_smoke_only")

    def test_multi_pattern_trace_cycles_by_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.npz"
            first = np.empty((2, 3, 2), dtype=np.int16)
            first[0] = [[0, 1], [1, 2], [2, 3]]
            first[1] = [[3, 2], [2, 1], [1, 0]]
            second = np.flip(first, axis=1).copy()
            metadata = {
                "schema_version": 1,
                "source": "merged_exact_qwen35_router_trace",
                "source_backend": "deepspeed",
                "sequence_length": 3,
                "global_batch_size": 1,
                "patterns": 2,
                "layers": 2,
                "tokens": 3,
                "top_k": 2,
            }
            np.savez_compressed(
                trace,
                topk_indices=np.stack((first, second), axis=0),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            controller = RouteController(
                num_layers=2,
                num_experts=4,
                top_k=2,
                sequence_length=3,
                tokens_per_microbatch=3,
                microbatches_per_step=2,
                trace_path=trace,
                allow_synthetic=False,
            )
            logits = torch.randn(3, 4)
            controller.set_position(step=0, microbatch=0)
            _, step_zero, _ = controller.select(layer_idx=0, logits=logits)
            controller.set_position(step=0, microbatch=1)
            _, step_one, _ = controller.select(layer_idx=0, logits=logits)
            controller.set_position(step=1, microbatch=0)
            _, step_two, _ = controller.select(layer_idx=0, logits=logits)
            self.assertEqual(step_zero.tolist(), first[0].tolist())
            self.assertEqual(step_one.tolist(), second[0].tolist())
            self.assertEqual(step_two.tolist(), first[0].tolist())
            with self.assertRaisesRegex(
                ValueError,
                r"expected warmup_steps\*GAS=1",
            ):
                RouteController(
                    num_layers=2,
                    num_experts=4,
                    top_k=2,
                    sequence_length=3,
                    tokens_per_microbatch=3,
                    expected_patterns=1,
                    trace_path=trace,
                    allow_synthetic=False,
                )

    def test_capture_writes_all_warmup_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ",
                {"FFT_APTMOE_SIMULATION_ROOT": str(root)},
            ):
                capture = RouteTraceCapture(
                    output_dir=root / "routes",
                    sequence_length=3,
                    max_patterns=2,
                    expected_layers=2,
                    top_k=2,
                )
            for offset in (0, 1):
                capture.begin_model_forward(None, ())
                for layer_idx in range(2):
                    selected = torch.tensor(
                        [
                            [offset, 2],
                            [1, 3],
                            [2, offset],
                        ],
                        dtype=torch.long,
                    )
                    capture.hook(layer_idx)(
                        None,
                        (),
                        (None, None, selected),
                    )
                capture.end_model_forward(None, (), None)
            capture.write()
            with np.load(
                root / "routes" / "rank_00.npz",
                allow_pickle=False,
            ) as data:
                routes = np.asarray(data["topk_indices"])
                metadata = json.loads(str(data["metadata_json"].item()))
            self.assertEqual(routes.shape, (2, 2, 3, 2))
            self.assertEqual(metadata["patterns"], 2)

    def test_route_merge_groups_microbatches_by_optimizer_step(self) -> None:
        routes = np.arange(8 * 2 * 3 * 2, dtype=np.int16).reshape(
            8,
            2,
            3,
            2,
        )
        grouped = group_route_patterns_by_optimizer_step(routes, 4)
        self.assertEqual(grouped.shape, (2, 2, 12, 2))
        np.testing.assert_array_equal(
            grouped[0, 0],
            np.concatenate([routes[index, 0] for index in range(4)]),
        )
        np.testing.assert_array_equal(
            grouped[1, 1],
            np.concatenate([routes[index, 1] for index in range(4, 8)]),
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            group_route_patterns_by_optimizer_step(routes[:7], 4)

    def test_kt_capture_observes_actual_wrapper_routing(self) -> None:
        class FakeKTMoEWrapper(torch.nn.Module):
            def __init__(self, offset: int) -> None:
                super().__init__()
                self._is_kt_moe_wrapper = True
                self.offset = offset

            def _compute_routing(
                self,
                hidden_states: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                tokens = hidden_states.shape[0]
                ids = torch.tensor(
                    [[self.offset, self.offset + 1]],
                    dtype=torch.long,
                ).expand(tokens, -1)
                weights = torch.full((tokens, 2), 0.5)
                return ids, weights

        class FakeLayer(torch.nn.Module):
            def __init__(self, offset: int) -> None:
                super().__init__()
                self.mlp = FakeKTMoEWrapper(offset)

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [FakeLayer(0), FakeLayer(2)]
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ",
                {"FFT_APTMOE_SIMULATION_ROOT": str(root)},
            ):
                capture = RouteTraceCapture(
                    output_dir=root / "routes",
                    sequence_length=3,
                    max_patterns=1,
                    expected_layers=2,
                    top_k=2,
                )
            model = FakeModel()
            handles = _install_kt_route_hooks(model, capture)
            capture.set_hook_handles(handles)
            capture.begin_model_forward(model, ())
            hidden = torch.zeros(3, 4)
            for layer in model.layers:
                layer.mlp._compute_routing(hidden)
            capture.end_model_forward(model, (), None)
            self.assertEqual(
                capture.patterns[0][0].tolist(),
                [[0, 1], [0, 1], [0, 1]],
            )
            self.assertEqual(
                capture.patterns[0][1].tolist(),
                [[2, 3], [2, 3], [2, 3]],
            )
            for layer in model.layers:
                self.assertNotIn("_compute_routing", layer.mlp.__dict__)


class PlacementAndStorageTest(unittest.TestCase):
    def test_profiled_empty_placement_is_not_randomly_replaced(self) -> None:
        class EmptyPlacement:
            def solve(self, *_args, **_kwargs):
                return []

        self.assertEqual(
            _solve_stage_hot_experts(
                EmptyPlacement(),
                [0, 0, 0, 0],
                layer_type="linear_attention",
                is_first_stage=False,
                is_last_stage=False,
            ),
            [],
        )

    def test_missing_placement_decision_fails_instead_of_prefetching(self) -> None:
        class MissingPlacement:
            def solve(self, *_args, **_kwargs):
                return None

        with self.assertRaisesRegex(RuntimeError, "no decision"):
            _solve_stage_hot_experts(
                MissingPlacement(),
                [0, 0, 0, 0],
                layer_type="linear_attention",
                is_first_stage=False,
                is_last_stage=False,
            )

    def test_profiled_solver_uses_aptmoe_compute_load_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lookup = Path(directory) / "lookup.json"
            lookup.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_class": "aptmoe_qwen35_proxy_lookup",
                        "expert": {
                            "bf16_bytes": EXPECTED_EXPERT_BF16_BYTES,
                            "num_experts": 4,
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
                            "max_tokens": 2,
                            "forward_seconds_by_tokens": [
                                0.0,
                                0.5,
                                10.0,
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            solver = ProxyPlacementSolver(
                4,
                1,
                lookup_path=lookup,
                prefetch_portion=0.25,
                allow_unprofiled=False,
            )
            # The formal solver must not apply the 25% smoke-only cap.
            self.assertGreater(
                len(
                    solver.solve(
                        [0, 1, 2, 2],
                        layer_type="linear_attention",
                    )
                ),
                1,
            )
            with self.assertRaisesRegex(ValueError, "required at least 3"):
                ProxyPlacementSolver(
                    4,
                    1,
                    lookup_path=lookup,
                    prefetch_portion=0.25,
                    allow_unprofiled=False,
                    required_max_tokens=3,
                )

    def test_chunked_causal_lm_loss_matches_full_logits(self) -> None:
        torch.manual_seed(7)
        full_head = torch.nn.Linear(8, 13, bias=False)
        chunked_head = torch.nn.Linear(8, 13, bias=False)
        chunked_head.load_state_dict(full_head.state_dict())
        full_hidden = torch.randn(2, 5, 8, requires_grad=True)
        chunked_hidden = full_hidden.detach().clone().requires_grad_(True)
        labels = torch.tensor(
            [
                [1, 2, 3, -100, 5],
                [6, 7, 8, 9, 10],
            ]
        )

        full_logits = full_head(full_hidden)
        expected = torch.nn.functional.cross_entropy(
            full_logits[..., :-1, :].contiguous().view(-1, 13),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        actual = _chunked_causal_lm_loss(
            chunked_head,
            chunked_hidden,
            labels,
            num_items_in_batch=None,
            chunk_size=3,
        )
        torch.testing.assert_close(actual, expected)

        expected.backward()
        actual.backward()
        torch.testing.assert_close(chunked_hidden.grad, full_hidden.grad)
        torch.testing.assert_close(
            chunked_head.weight.grad,
            full_head.weight.grad,
        )

    def test_chunked_causal_lm_loss_preserves_num_items_scaling(self) -> None:
        torch.manual_seed(8)
        head = torch.nn.Linear(4, 7, bias=False)
        hidden = torch.randn(1, 4, 4)
        labels = torch.tensor([[1, 2, -100, 4]])
        logits = head(hidden)
        expected = torch.nn.functional.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, 7),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
            reduction="sum",
        ) / 9
        actual = _chunked_causal_lm_loss(
            head,
            hidden,
            labels,
            num_items_in_batch=9,
            chunk_size=2,
        )
        torch.testing.assert_close(actual, expected)

    def test_large_artifact_path_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sim"
            inside = require_within_simulation_root(root / "weights", root)
            self.assertEqual(inside, (root / "weights").resolve())
            with self.assertRaises(ValueError):
                require_within_simulation_root(
                    Path(directory) / "outside",
                    root,
                )


class ResultIsolationTest(unittest.TestCase):
    def test_formal_proxy_is_never_labeled_exact_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "seq_32"
            timing_dir = run_dir / "step_timing"
            timing_dir.mkdir(parents=True)
            config = {
                "backend": "aptmoe",
                "profile": "server",
                "benchmark_class": "deployment_proxy",
                "result_validity": "formal_deployment_proxy",
                "weight_source": "deterministic_random_initialization",
                "checkpoint_compatible": False,
                "llamafactory_backend": False,
                "allow_end_to_end_qwen35_tps_claim": False,
                "model_load_architecture": (
                    "Qwen35ComponentIsomorphicAPTMoEProxy"
                ),
                "precision": "bf16",
                "steps": 2,
                "warmup_steps": 1,
            }
            (run_dir / "run_config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (timing_dir / "step_timing.json").write_text(
                json.dumps(
                    {
                        "timing_mode": "coarse_host_wall_no_cuda_sync",
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
                        "tps_attribution": {"stable_tps": 32.0},
                        "instrumentation": {
                            "forced_cuda_synchronize": False,
                            "backend_internal_probes": False,
                            "system_resource_monitor": False,
                            "per_step_file_io": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "proxy_manifest.json").write_text(
                json.dumps(
                    {
                        "benchmark_class": "deployment_proxy",
                        "result_validity": "formal_deployment_proxy",
                        "proxy_architecture": "qwen35_component_isomorphic",
                        "parameter_count": 34_660_610_688,
                        "checkpoint_compatible": False,
                        "real_forward_backward_optimizer_update": True,
                        "route": {
                            "mode": "replayed_qwen35_topk_indices",
                            "trace_sha256": "route-hash",
                        },
                        "placement": {
                            "mode": "profiled_compute_load",
                            "deployment_profile": "server",
                            "lookup_sha256": "lookup-hash",
                        },
                        "runtime_versions": {
                            "qwen35_linear_attention_fastpath": True
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "full_update_verification.json").write_text(
                json.dumps({"valid_full_update": True}),
                encoding="utf-8",
            )
            row = aggregate_run(run_dir / "run_config.json")
            self.assertEqual(row["status"], "OK_PROXY")
            self.assertEqual(row["benchmark_class"], "deployment_proxy")


if __name__ == "__main__":
    unittest.main()
