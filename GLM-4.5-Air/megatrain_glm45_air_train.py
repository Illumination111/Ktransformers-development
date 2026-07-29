#!/usr/bin/env python3
"""MegaTrain GLM-4.5-Air full-finetuning benchmark with canonical timing."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import psutil
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from infinity import CPUMasterModel, ChatDataset, collate_fn
from infinity.config import get_num_workers, get_optimizer_type, load_training_config
from infinity.config.yaml_loader import load_yaml_config


EXPECTED_MODEL_TYPE = "glm4_moe"
EXPECTED_ARCHITECTURE = "Glm4MoeForCausalLM"
EXPECTED_LOGICAL_PARAMETERS = 106_852_245_504
TIMING_MODE = "megatrain_host_wall_with_backend_cuda_sync"
PHASE_KEYS = ("forward_sec", "backward_sec", "optimizer_sec")
STEP_KEYS = (
    "global_step",
    "microbatches",
    *PHASE_KEYS,
    "step_total_sec",
    "step_tps",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("megatrain_glm45_air_benchmark")


def _stats(
    rows: list[dict[str, float | int]],
    key: str,
) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {
            "count": 0,
            "mean_sec": None,
            "min_sec": None,
            "max_sec": None,
        }
    return {
        "count": len(values),
        "mean_sec": statistics.fmean(values),
        "min_sec": min(values),
        "max_sec": max(values),
    }


def write_timing(
    out_dir: Path,
    rows: list[dict[str, float | int]],
    warmup_steps: int,
    tokens_per_step: int,
) -> None:
    stable = [
        row for row in rows if int(row["global_step"]) > warmup_steps
    ]
    aggregate_all = {
        key: _stats(rows, key)
        for key in (*PHASE_KEYS, "step_total_sec")
    }
    aggregate_stable = {
        key: _stats(stable, key)
        for key in (*PHASE_KEYS, "step_total_sec")
    }
    stable_step = aggregate_stable["step_total_sec"]["mean_sec"]
    stable_tps = (
        tokens_per_step / float(stable_step)
        if stable_step is not None and float(stable_step) > 0
        else None
    )
    summary = {
        "schema_version": 1,
        "timing_mode": TIMING_MODE,
        "backend": "megatrain",
        "precision": "bf16",
        "instrumentation": {
            "forced_cuda_synchronize": True,
            "backend_internal_probes": False,
            "system_resource_monitor": False,
            "per_step_file_io": False,
        },
        "phase_attribution": (
            "forward/backward use MegaTrain CUDA-event timings; optimizer "
            "and step total use host wall time"
        ),
        "warmup_steps": warmup_steps,
        "tokens_per_step": tokens_per_step,
        "num_steps": len(rows),
        "num_stable_steps": len(stable),
        "steps": rows,
        "aggregate_all": aggregate_all,
        "aggregate_stable": aggregate_stable,
        "tps_attribution": {
            "tokens_per_step": tokens_per_step,
            "mean_stable_step_sec": stable_step,
            "stable_tps": stable_tps,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "step_timing.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "step_timing.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_KEYS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MegaTrain step phase timing",
        "",
        f"- Mode: `{TIMING_MODE}`",
        "- MegaTrain performs backend CUDA synchronization in forward/backward.",
        "- External CPU/GPU sampling is outside these phase timers.",
        f"- Stable steps: {len(stable)} (excluded warmup: {warmup_steps})",
        (
            f"- Stable TPS: {stable_tps:.3f}"
            if stable_tps is not None
            else "- Stable TPS: unavailable"
        ),
        "",
        "| Step | Microbatches | Forward (s) | Backward (s) | "
        "Optimizer (s) | Total (s) | TPS |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {global_step} | {microbatches} | {forward_sec:.6f} | "
            "{backward_sec:.6f} | {optimizer_sec:.6f} | "
            "{step_total_sec:.6f} | {step_tps:.3f} |".format(**row)
        )
    (out_dir / "step_timing.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def next_batch(data_iter: Any, dataloader: DataLoader) -> tuple[Any, Any]:
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter), data_iter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--timing-output-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--devices", required=True)
    return parser.parse_args()


def validate_source_model(
    model: Any,
) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]], int]:
    model_type = getattr(model.config, "model_type", None)
    architectures = list(getattr(model.config, "architectures", None) or [])
    if model_type != EXPECTED_MODEL_TYPE:
        raise RuntimeError(
            f"Expected model_type={EXPECTED_MODEL_TYPE!r}, got {model_type!r}"
        )
    if EXPECTED_ARCHITECTURE not in architectures:
        raise RuntimeError(
            f"Expected architecture {EXPECTED_ARCHITECTURE!r}, "
            f"got {architectures!r}"
        )
    if type(model).__name__ != EXPECTED_ARCHITECTURE:
        raise RuntimeError(
            f"Expected {EXPECTED_ARCHITECTURE}, constructed "
            f"{type(model).__name__}"
        )

    named_parameters = list(model.named_parameters())
    frozen = [name for name, parameter in named_parameters if not parameter.requires_grad]
    if frozen:
        raise RuntimeError(
            "MegaTrain full fine-tuning contract failed: source parameters "
            f"are frozen: {frozen[:5]}"
        )
    logical_total = sum(parameter.numel() for _, parameter in named_parameters)
    if logical_total != EXPECTED_LOGICAL_PARAMETERS:
        raise RuntimeError(
            "Unexpected GLM-4.5-Air parameter count: "
            f"got={logical_total}, expected={EXPECTED_LOGICAL_PARAMETERS}"
        )
    named_buffers = list(model.named_buffers())
    meta_buffers = [
        name
        for name, buffer in named_buffers
        if getattr(buffer, "device", None) is not None
        and buffer.device.type == "meta"
    ]
    if meta_buffers:
        raise RuntimeError(
            "MegaTrain source model contains unmaterialized buffers: "
            f"{meta_buffers[:5]}"
        )
    correction_buffers = [
        name
        for name, _ in named_buffers
        if name.endswith("e_score_correction_bias")
    ]
    if not correction_buffers:
        raise RuntimeError(
            "GLM-4.5-Air e_score_correction_bias buffers were not found"
        )
    return named_parameters, named_buffers, logical_total


def validate_megatrain_coverage(
    model: CPUMasterModel,
    source_named_parameters: list[tuple[str, Any]],
    source_named_buffers: list[tuple[str, Any]],
    logical_total: int,
) -> None:
    managed = list(model.get_parameters())
    managed_ids = {id(parameter) for parameter in managed}
    missing = [
        name
        for name, parameter in source_named_parameters
        if id(parameter) not in managed_ids
    ]
    if missing:
        raise RuntimeError(
            "MegaTrain full fine-tuning parameter coverage is incomplete: "
            f"{missing[:5]}"
        )
    frozen = [
        index
        for index, parameter in enumerate(managed)
        if not parameter.requires_grad
    ]
    if frozen:
        raise RuntimeError(
            "MegaTrain optimizer parameters are frozen: "
            f"indices={frozen[:8]}"
        )
    managed_total = sum(parameter.numel() for parameter in managed)
    if managed_total != logical_total:
        raise RuntimeError(
            "MegaTrain optimizer parameter count mismatch: "
            f"managed={managed_total}, source={logical_total}"
        )
    managed_modules = [
        model.embedding,
        *model.cpu_layers,
        model.norm,
        model.lm_head,
        model.rotary_emb,
    ]
    managed_buffer_ids = {
        id(buffer)
        for module in managed_modules
        if module is not None
        for buffer in module.buffers()
    }
    missing_buffers = [
        name
        for name, buffer in source_named_buffers
        if id(buffer) not in managed_buffer_ids
    ]
    if missing_buffers:
        raise RuntimeError(
            "MegaTrain model-structure discovery omitted model buffers: "
            f"{missing_buffers[:5]}"
        )
    logger.info(
        "[glm45_model_contract] contract=OK backend=megatrain "
        "class=%s model_type=%s logical_trainable=%d logical_total=%d "
        "buffers=%d",
        EXPECTED_ARCHITECTURE,
        EXPECTED_MODEL_TYPE,
        managed_total,
        logical_total,
        len(source_named_buffers),
    )


def create_optimizer(
    model: CPUMasterModel,
    config: Any,
    optimizer_type: str,
) -> Any:
    if optimizer_type == "deepspeed_adam":
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except ImportError as error:
            raise RuntimeError(
                "MegaTrain comparison requires the prebuilt DeepSpeedCPUAdam"
            ) from error
        return DeepSpeedCPUAdam(
            model.get_parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            adamw_mode=True,
        )
    return torch.optim.AdamW(
        model.get_parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def main() -> None:
    args = parse_args()
    yaml_config = load_yaml_config(args.config)
    config = load_training_config(args.config)
    config.devices = [int(value) for value in args.devices.split(",")]
    config.device = config.devices[0]
    config.world_size = len(config.devices)
    if config.batch_size % config.world_size:
        raise ValueError(
            f"global batch {config.batch_size} is not divisible by "
            f"{config.world_size} devices"
        )
    if args.warmup_steps < 0 or args.warmup_steps >= config.num_steps:
        raise ValueError("warmup steps must be non-negative and smaller than num_steps")
    if config.dtype is not torch.bfloat16:
        raise ValueError(f"MegaTrain comparison is BF16-only, got {config.dtype}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a real MegaTrain benchmark")

    logger.info(
        "MegaTrain GLM-4.5-Air benchmark: model=%s devices=%s "
        "global_batch=%d sequence=%d GAS=%d steps=%d",
        config.model_name,
        config.devices,
        config.batch_size,
        config.max_seq_len,
        config.gradient_accumulation_steps,
        config.num_steps,
    )
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=config.dtype,
        device_map="cpu",
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
        local_files_only=True,
    )
    (
        source_named_parameters,
        source_named_buffers,
        logical_total,
    ) = validate_source_model(hf_model)
    model = CPUMasterModel(hf_model, config)
    validate_megatrain_coverage(
        model,
        source_named_parameters,
        source_named_buffers,
        logical_total,
    )
    del source_named_parameters
    del source_named_buffers
    del hf_model

    optimizer = create_optimizer(
        model,
        config,
        get_optimizer_type(yaml_config),
    )
    dataset = ChatDataset(
        tokenizer,
        config.max_seq_len,
        dataset_name=config.dataset_name or None,
        dataset_dir=config.dataset_dir,
        dataset_path=config.dataset_path or None,
        query_field=config.query_field,
        response_field=config.response_field,
        system_prompt=config.system_prompt or None,
        train_on_prompt=config.train_on_prompt,
        response_preserving_truncation=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=get_num_workers(yaml_config),
        drop_last=True,
    )
    data_iter = iter(dataloader)
    process = psutil.Process()
    rows: list[dict[str, float | int]] = []
    tokens_per_step = (
        config.batch_size
        * config.max_seq_len
        * config.gradient_accumulation_steps
    )
    optimizer.zero_grad()

    try:
        for global_step in range(1, config.num_steps + 1):
            step_started = time.perf_counter()
            forward_sec = 0.0
            backward_sec = 0.0
            losses: list[float] = []
            for _ in range(config.gradient_accumulation_steps):
                batch, data_iter = next_batch(data_iter, dataloader)
                loss_value, _, timing = model.forward_and_backward(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )
                losses.append(float(loss_value))
                forward_sec += float(timing.get("forward", 0.0))
                backward_sec += float(timing.get("backward", 0.0))

            optimizer_started = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(
                model.get_parameters(),
                config.max_grad_norm,
            )
            optimizer.step()
            model._sync_params_to_gpu()
            model.zero_grad()
            optimizer.zero_grad()
            optimizer_sec = time.perf_counter() - optimizer_started
            step_total_sec = time.perf_counter() - step_started
            row: dict[str, float | int] = {
                "global_step": global_step,
                "microbatches": config.gradient_accumulation_steps,
                "forward_sec": forward_sec,
                "backward_sec": backward_sec,
                "optimizer_sec": optimizer_sec,
                "step_total_sec": step_total_sec,
                "step_tps": tokens_per_step / step_total_sec,
            }
            rows.append(row)
            logger.info(
                "Step %d/%d | Loss %.4f | Time %.3fs | Tokens/s %.2f | "
                "FWD %.3fs | BWD %.3fs | OPT %.3fs | CPU RSS %.2f GiB",
                global_step,
                config.num_steps,
                statistics.fmean(losses),
                step_total_sec,
                row["step_tps"],
                forward_sec,
                backward_sec,
                optimizer_sec,
                process.memory_info().rss / (1024**3),
            )
    finally:
        if rows:
            write_timing(
                args.timing_output_dir,
                rows,
                args.warmup_steps,
                tokens_per_step,
            )
        model.cleanup()

    logger.info("TRAINING COMPLETE; timing=%s", args.timing_output_dir)


if __name__ == "__main__":
    main()
