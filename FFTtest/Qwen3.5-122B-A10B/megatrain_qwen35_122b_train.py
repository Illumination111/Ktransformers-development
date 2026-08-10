#!/usr/bin/env python3
"""MegaTrain Qwen3.5 text-only benchmark with canonical sweep timing output.

MegaTrain's public SFT entrypoint detects the multimodal wrapper in the local
Qwen3.5-122B-A10B checkpoint.  The other backends in this benchmark deliberately
load only Qwen3_5MoeForCausalLM, so this entrypoint applies that same text-only
contract before handing the model to MegaTrain's CPUMasterModel.

MegaTrain uses CUDA events and a CUDA synchronize inside its forward/backward
implementation.  The emitted timing metadata records that fact instead of
claiming the probe-free timing mode used by the other exact-model backends.
"""

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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from infinity import CPUMasterModel, ChatDataset, collate_fn
from infinity.config import get_num_workers, get_optimizer_type, load_training_config
from infinity.config.yaml_loader import load_yaml_config
from qwen35_text_only import _extract_text_config, assert_text_only_model


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
logger = logging.getLogger("megatrain_qwen35_122b_benchmark")


def _stats(rows: list[dict[str, float | int]], key: str) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {"count": 0, "mean_sec": None, "min_sec": None, "max_sec": None}
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
    stable = [row for row in rows if int(row["global_step"]) > warmup_steps]
    aggregate_all = {
        key: _stats(rows, key) for key in (*PHASE_KEYS, "step_total_sec")
    }
    aggregate_stable = {
        key: _stats(stable, key) for key in (*PHASE_KEYS, "step_total_sec")
    }
    stable_step = aggregate_stable["step_total_sec"]["mean_sec"]
    stable_tps = (
        tokens_per_step / float(stable_step)
        if stable_step is not None and float(stable_step) > 0
        else None
    )
    summary: dict[str, Any] = {
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
            "forward/backward use MegaTrain CUDA-event timings; optimizer and "
            "step total use host wall time"
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
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_KEYS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MegaTrain step phase timing",
        "",
        f"- Mode: `{TIMING_MODE}`",
        "- MegaTrain performs backend CUDA synchronization in forward/backward.",
        "- External CPU/GPU resource sampling is not included in these phase timers.",
        f"- Stable steps: {len(stable)} (excluded warmup: {warmup_steps})",
        f"- Stable TPS: {stable_tps:.3f}" if stable_tps is not None else "- Stable TPS: unavailable",
        "",
        "| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {global_step} | {microbatches} | {forward_sec:.6f} | "
            "{backward_sec:.6f} | {optimizer_sec:.6f} | "
            "{step_total_sec:.6f} | {step_tps:.3f} |".format(**row)
        )
    (out_dir / "step_timing.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--timing-output-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--devices", required=True)
    return parser.parse_args()


def next_batch(data_iter: Any, dataloader: DataLoader) -> tuple[Any, Any]:
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter), data_iter


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
    optimizer_type = get_optimizer_type(yaml_config)
    num_workers = get_num_workers(yaml_config)
    if args.warmup_steps < 0 or args.warmup_steps >= config.num_steps:
        raise ValueError("warmup steps must be non-negative and smaller than num_steps")
    if config.dtype is not torch.bfloat16:
        raise ValueError(f"MegaTrain comparison is BF16-only, got {config.dtype}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a real MegaTrain benchmark")

    logger.info(
        "MegaTrain text-only benchmark: model=%s devices=%s global_batch=%d "
        "sequence=%d GAS=%d steps=%d",
        config.model_name,
        config.devices,
        config.batch_size,
        config.max_seq_len,
        config.gradient_accumulation_steps,
        config.num_steps,
    )

    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    source_config = AutoConfig.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    text_config = _extract_text_config(source_config)
    logger.info(
        "Text-only load: Qwen3_5MoeForConditionalGeneration -> "
        "Qwen3_5MoeForCausalLM (vision and MTP excluded)"
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        config=text_config,
        dtype=config.dtype,
        device_map="cpu",
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
    )
    assert_text_only_model(hf_model, "full")
    expected = 122_111_526_912
    actual = sum(parameter.numel() for parameter in hf_model.parameters())
    if actual != expected:
        raise RuntimeError(
            "Unexpected Qwen3.5-122B-A10B text-model parameter count: "
            f"got={actual}, expected={expected}"
        )

    model = CPUMasterModel(hf_model, config)
    del hf_model

    if optimizer_type == "deepspeed_adam":
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except ImportError as error:
            raise RuntimeError(
                "MegaTrain comparison requires the prebuilt DeepSpeedCPUAdam"
            ) from error
        optimizer = DeepSpeedCPUAdam(
            model.get_parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            adamw_mode=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.get_parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
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
        num_workers=num_workers,
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
                model.get_parameters(), config.max_grad_norm
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
