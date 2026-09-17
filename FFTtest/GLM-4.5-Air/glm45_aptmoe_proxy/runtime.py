"""APTMoE pipeline construction, timing, and full-update audit."""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from Runtime.OffloadRuntime.comm_scheduler import CommScheduler
from Runtime.PipelineRuntime.generate_action_list import (
    generate_action_Mobius_APTMoE,
)
from Runtime.PipelineRuntime.pipeline_runtime import PipelineRuntime

from .model import (
    APTGlm45Stage,
    Glm45ModelShard,
    categorized_parameter_counts,
    parameter_category,
)
from .placement import ProxyPlacementSolver
from .routes import RouteController

REQUIRED_CATEGORIES = {
    "embedding",
    "lm_head",
    "norm",
    "token_mixer",
    "dense_mlp",
    "router",
    "routed_experts",
    "shared_expert",
}


class ProxyPipelineRuntime(PipelineRuntime):
    def forward_pass(
        self,
        mod_rank: int,
        source_tensor: torch.Tensor,
        chunk_id: int,
    ) -> None:
        target = self.module_list[mod_rank]
        following = (
            self.module_list[mod_rank + 1]
            if mod_rank != self.num_stages - 1
            else None
        )
        is_last = self.sft_mode and self._is_last_stage_module(mod_rank)
        labels = self._get_labels_for_chunk(chunk_id) if is_last else None
        target.FwdStageLoad(chunk_id, self.num_chunks, sft_mode=True)
        if mod_rank + self.world_size < len(self.module_list):
            self.module_list[mod_rank + self.world_size].FwdStageLoad(
                chunk_id, self.num_chunks, sft_mode=True
            )
        self.input_batch_list[mod_rank].append(source_tensor)
        result = (
            target(
                source_tensor,
                chunk_id,
                following,
                labels=labels,
                num_items_in_batch=self.step_num_items_in_batch,
            )
            if is_last and labels is not None
            else target(source_tensor, chunk_id, following)
        )
        torch.cuda.current_stream().wait_event(target.StageCompEvent)
        target.FwdStageDrop(
            chunk_id, self.num_chunks, self.fwd_only, sft_mode=True
        )
        self.batch_activation_list[mod_rank].append(result)


def build_proxy_pipeline(
    *,
    config: Any,
    world_size: int,
    local_rank: int,
    global_rank: int,
    routes: RouteController,
    placement_solver: ProxyPlacementSolver,
    seed: int,
) -> tuple[list[Glm45ModelShard | None], CommScheduler]:
    if config.num_hidden_layers != 46:
        raise ValueError("GLM-4.5-Air proxy requires 46 layers")
    scheduler = CommScheduler(device_id=local_rank)
    modules: list[Glm45ModelShard | None] = []
    for stage_id in range(config.num_hidden_layers):
        if stage_id % world_size != global_rank:
            modules.append(None)
            continue
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + stage_id)
            stage = APTGlm45Stage(
                config=config,
                stage_id=stage_id,
                routes=routes,
                comm_scheduler=scheduler,
                placement_solver=placement_solver,
            )
        modules.append(
            Glm45ModelShard(
                model_shard=stage,
                compute_device_id=local_rank,
                offload_device="cpu",
                index=stage_id,
                offload_grained="fine",
                inter_stage_only=True,
                sft_mode=True,
            )
        )
    return modules, scheduler


def local_parameter_counts(
    modules: Iterable[Glm45ModelShard | None],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for shard in modules:
        if shard is None:
            continue
        for key, value in categorized_parameter_counts(
            shard.model_shard
        ).items():
            result[key] = result.get(key, 0) + value
    return result


def global_parameter_counts(local: dict[str, int]) -> dict[str, int]:
    gathered: list[dict[str, int] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(gathered, local)
    result: dict[str, int] = {}
    for values in gathered:
        if values:
            for key, value in values.items():
                result[key] = result.get(key, 0) + value
    return result


class FullUpdateAudit:
    def __init__(
        self,
        modules: list[Glm45ModelShard | None],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.parameters: list[tuple[str, torch.nn.Parameter]] = []
        for shard in modules:
            if shard is not None:
                self.parameters.extend(shard.model_shard.named_parameters())
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != {id(value) for _, value in self.parameters}:
            raise RuntimeError("optimizer scope does not match local proxy parameters")
        self.optimizer = optimizer
        self.samples: dict[
            str,
            list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]],
        ] = {}
        self.sample_counts: dict[str, int] = {}
        self.gradients: dict[str, bool] = {}

    def observe_before_step(self) -> None:
        for name, parameter in self.parameters:
            if parameter.grad is None:
                continue
            category = parameter_category(name)
            gradient = parameter.grad.detach().reshape(-1)
            index = int(gradient.abs().argmax().item())
            if gradient[index].item() == 0:
                continue
            self.gradients[category] = True
            remaining = 4096 - self.sample_counts.get(category, 0)
            if remaining <= 0:
                continue
            sample_size = min(remaining, parameter.numel())
            start = min(
                max(0, index - sample_size // 2),
                parameter.numel() - sample_size,
            )
            indices = torch.arange(
                start,
                start + sample_size,
                device=parameter.device,
            )
            before = (
                parameter.detach()
                .reshape(-1)
                .index_select(0, indices)
                .cpu()
                .clone()
            )
            self.samples.setdefault(category, []).append(
                (
                    parameter,
                    indices.cpu(),
                    before,
                )
            )
            self.sample_counts[category] = (
                self.sample_counts.get(category, 0) + sample_size
            )

    def local_result(self) -> dict[str, Any]:
        changed: dict[str, bool] = {}
        for category, samples in self.samples.items():
            changed[category] = any(
                not torch.equal(
                    before,
                    parameter.detach()
                    .reshape(-1)
                    .index_select(0, indices.to(parameter.device))
                    .cpu(),
                )
                for parameter, indices, before in samples
            )
        state_dtypes: set[str] = set()
        state_devices: set[str] = set()
        for state in self.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = state.get(key)
                if isinstance(tensor, torch.Tensor):
                    state_dtypes.add(str(tensor.dtype))
                    state_devices.add(tensor.device.type)
        return {
            "optimizer_parameter_count": sum(
                parameter.numel() for _, parameter in self.parameters
            ),
            "gradient_seen": self.gradients,
            "weight_changed": changed,
            "optimizer_state_dtypes": sorted(state_dtypes),
            "optimizer_state_devices": sorted(state_devices),
        }

    def gather(self) -> dict[str, Any] | None:
        gathered: list[dict[str, Any] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered, self.local_result())
        if dist.get_rank() != 0:
            return None
        gradient_seen: dict[str, bool] = {}
        weight_changed: dict[str, bool] = {}
        dtypes: set[str] = set()
        devices: set[str] = set()
        count = 0
        for value in gathered:
            assert value is not None
            count += value["optimizer_parameter_count"]
            for key, seen in value["gradient_seen"].items():
                gradient_seen[key] = gradient_seen.get(key, False) or seen
            for key, changed in value["weight_changed"].items():
                weight_changed[key] = weight_changed.get(key, False) or changed
            dtypes.update(value["optimizer_state_dtypes"])
            devices.update(value["optimizer_state_devices"])
        missing_grad = sorted(
            category
            for category in REQUIRED_CATEGORIES
            if not gradient_seen.get(category, False)
        )
        unchanged = sorted(
            category
            for category in REQUIRED_CATEGORIES - {"norm"}
            if not weight_changed.get(category, False)
        )
        valid = (
            not missing_grad
            and not unchanged
            and dtypes == {"torch.bfloat16"}
            and devices == {"cpu"}
        )
        return {
            "schema_version": 1,
            "optimizer_scope": "all_proxy_parameters",
            "optimizer_parameter_count": count,
            "gradient_seen": gradient_seen,
            "weight_changed": weight_changed,
            "missing_gradient_categories": missing_grad,
            "unchanged_weight_categories": unchanged,
            "optimizer_state_dtypes": sorted(dtypes),
            "optimizer_state_devices": sorted(devices),
            "valid_full_update": valid,
        }


def _stats(rows: list[dict[str, float | int]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "count": len(values),
        "mean_sec": statistics.fmean(values) if values else None,
        "min_sec": min(values) if values else None,
        "max_sec": max(values) if values else None,
    }


def _co_locate_optimizer_tensors(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Repair device mismatches while making every repair auditable."""
    moved: dict[str, dict[str, int]] = {}
    non_cpu: dict[str, int] = {}

    def record(kind: str, source: torch.device, target: torch.device) -> None:
        key = f"{source.type}->{target.type}"
        values = moved.setdefault(kind, {})
        values[key] = values.get(key, 0) + 1

    def record_non_cpu(kind: str, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu":
            non_cpu[kind] = non_cpu.get(kind, 0) + 1

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            record_non_cpu("parameter", parameter)
            gradient = parameter.grad
            if gradient is not None:
                record_non_cpu("gradient", gradient)
                if gradient.device != parameter.device:
                    record("gradient", gradient.device, parameter.device)
                    parameter.grad = gradient.to(parameter.device)
            for key, value in optimizer.state.get(parameter, {}).items():
                if isinstance(value, torch.Tensor):
                    record_non_cpu(key, value)
                    if value.device != parameter.device:
                        record(key, value.device, parameter.device)
                        optimizer.state[parameter][key] = value.to(
                            parameter.device
                        )
    return {
        "cpu_only_before": not non_cpu,
        "non_cpu_tensors": non_cpu,
        "move_count": sum(sum(values.values()) for values in moved.values()),
        "moves": moved,
    }


def _optimizer_residency_snapshot(
    module_list: list[Glm45ModelShard | None],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Summarize parameter, gradient, and Adam-state residency by category."""
    categories: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    def add(
        category: str,
        kind: str,
        tensor: torch.Tensor,
    ) -> None:
        device = tensor.device.type
        values = (
            categories.setdefault(category, {})
            .setdefault(kind, {})
            .setdefault(device, {"tensors": 0, "elements": 0, "bytes": 0})
        )
        values["tensors"] += 1
        values["elements"] += tensor.numel()
        values["bytes"] += tensor.numel() * tensor.element_size()

    for shard in module_list:
        if shard is None:
            continue
        for name, parameter in shard.model_shard.named_parameters():
            category = parameter_category(name)
            add(category, "parameter", parameter)
            if parameter.grad is not None:
                add(category, "gradient", parameter.grad)
            for key in ("exp_avg", "exp_avg_sq"):
                value = optimizer.state.get(parameter, {}).get(key)
                if isinstance(value, torch.Tensor):
                    add(category, key, value)
    return {"categories": categories}


def _snapshot_is_cpu_only(snapshot: dict[str, Any]) -> bool:
    return all(
        not (set(devices) - {"cpu"})
        for kinds in snapshot["categories"].values()
        for devices in kinds.values()
    )


def run_full_update_steps(
    *,
    runtime: ProxyPipelineRuntime,
    module_list: list[Glm45ModelShard | None],
    routes: RouteController,
    steps: int,
    warmup_steps: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    tokens_per_step: int,
    timing_output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    actions = generate_action_Mobius_APTMoE(
        world_size=dist.get_world_size(),
        num_stages=runtime.num_stages,
        num_chunks=runtime.num_chunks,
    )[dist.get_rank()]
    split = next(
        index
        for index, action in enumerate(actions)
        if action.startswith(("backward ", "send_grad "))
    )
    forward_actions, backward_actions = actions[:split], actions[split:]
    runtime._checked_stages = set(range(runtime.num_stages))
    audit = FullUpdateAudit(module_list, runtime.optimizer_)
    local_rows: list[dict[str, float | int]] = []
    local_residency: list[dict[str, Any]] = []
    for step in range(steps):
        started = time.perf_counter()
        runtime.optimizer_.zero_grad(set_to_none=True)
        runtime.prefetch_step_data(gradient_accumulation_steps)
        forward_seconds = 0.0
        backward_seconds = 0.0
        for microbatch in range(gradient_accumulation_steps):
            routes.set_position(step, microbatch)
            phase = time.perf_counter()
            runtime.run_pipeline(action_list=forward_actions)
            forward_seconds += time.perf_counter() - phase
            phase = time.perf_counter()
            runtime.run_pipeline(action_list=backward_actions)
            for shard in module_list:
                if shard is not None:
                    shard._StageDropEvent.synchronize()
            backward_seconds += time.perf_counter() - phase
        if step < max(1, warmup_steps):
            audit.observe_before_step()
        phase = time.perf_counter()
        repair = _co_locate_optimizer_tensors(runtime.optimizer_)
        local_residency.append(
            {"global_step": step + 1, **repair}
        )
        if max_grad_norm > 0:
            parameters = [
                parameter
                for group in runtime.optimizer_.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            if parameters:
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        runtime.optimizer_.step()
        runtime.scheduler_step()
        optimizer_seconds = time.perf_counter() - phase
        total_seconds = time.perf_counter() - started
        local_rows.append(
            {
                "global_step": step + 1,
                "microbatches": gradient_accumulation_steps,
                "forward_sec": forward_seconds,
                "backward_sec": backward_seconds,
                "optimizer_sec": optimizer_seconds,
                "step_total_sec": total_seconds,
                "step_tps": tokens_per_step / total_seconds,
            }
        )
    gathered: list[list[dict[str, float | int]] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(gathered, local_rows)
    verification = audit.gather()
    final_residency = _optimizer_residency_snapshot(
        module_list, runtime.optimizer_
    )
    final_residency["cpu_only"] = _snapshot_is_cpu_only(final_residency)
    gathered_residency: list[dict[str, Any] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(
        gathered_residency,
        {"steps": local_residency, "final": final_residency},
    )
    if dist.get_rank() != 0:
        return None, verification
    residency_by_rank = {
        str(rank): values
        for rank, values in enumerate(gathered_residency)
        if values is not None
    }
    after_drop_cpu_only = all(
        step["cpu_only_before"]
        for values in residency_by_rank.values()
        for step in values["steps"]
    )
    repair_count = sum(
        step["move_count"]
        for values in residency_by_rank.values()
        for step in values["steps"]
    )
    final_cpu_only = all(
        values["final"]["cpu_only"]
        for values in residency_by_rank.values()
    )
    assert verification is not None
    verification["after_backward_drop_cpu_only"] = after_drop_cpu_only
    verification["co_location_repair_count"] = repair_count
    verification["final_optimizer_residency_cpu_only"] = final_cpu_only
    verification["valid_full_update"] = bool(
        verification["valid_full_update"]
        and after_drop_cpu_only
        and repair_count == 0
        and final_cpu_only
    )
    rank_rows = [rows for rows in gathered if rows is not None]
    merged: list[dict[str, float | int]] = []
    for index in range(steps):
        values = [rows[index] for rows in rank_rows]
        total = max(float(row["step_total_sec"]) for row in values)
        merged.append(
            {
                "global_step": index + 1,
                "microbatches": gradient_accumulation_steps,
                "forward_sec": max(float(row["forward_sec"]) for row in values),
                "backward_sec": max(float(row["backward_sec"]) for row in values),
                "optimizer_sec": max(float(row["optimizer_sec"]) for row in values),
                "step_total_sec": total,
                "step_tps": tokens_per_step / total,
            }
        )
    stable = merged[warmup_steps:]
    summary = {
        "schema_version": 1,
        "timing_mode": "coarse_host_wall_no_cuda_sync",
        "backend": "aptmoe",
        "precision": "bf16",
        "benchmark_class": "deployment_proxy",
        "instrumentation": {
            "forced_cuda_synchronize": False,
            "backend_internal_probes": False,
            "system_resource_monitor": False,
            "per_step_file_io": False,
        },
        "warmup_steps": warmup_steps,
        "tokens_per_step": tokens_per_step,
        "num_steps": steps,
        "num_stable_steps": len(stable),
        "steps": merged,
        "aggregate_stable": {
            key: _stats(stable, key)
            for key in (
                "forward_sec",
                "backward_sec",
                "optimizer_sec",
                "step_total_sec",
            )
        },
    }
    mean = summary["aggregate_stable"]["step_total_sec"]["mean_sec"]
    summary["tps_attribution"] = {
        "tokens_per_step": tokens_per_step,
        "mean_stable_step_sec": mean,
        "stable_tps": tokens_per_step / mean if mean else None,
    }
    timing_output_dir.mkdir(parents=True, exist_ok=True)
    (timing_output_dir / "optimizer_residency.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "after_backward_drop_cpu_only": after_drop_cpu_only,
                "co_location_repair_count": repair_count,
                "final_optimizer_residency_cpu_only": final_cpu_only,
                "by_rank": residency_by_rank,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (timing_output_dir / "step_timing.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (timing_output_dir / "step_timing.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    return summary, verification
