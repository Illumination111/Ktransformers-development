#!/usr/bin/env python3
"""Run every MegaTrain sequence for one profile in the same worker set."""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from infinity import CPUMasterModel, ChatDataset, collate_fn
from infinity.config import (
    get_num_workers,
    get_optimizer_type,
    load_training_config,
)
from infinity.config.yaml_loader import load_yaml_config
from megatrain_qwen35_train import next_batch, write_timing
from persistent_sweep import (
    activate_cuda_cache_hold,
    emit_monitor_phase,
    load_manifest,
    write_case_cuda_snapshot,
    write_case_exit,
)
from qwen35_text_only import _extract_text_config, assert_text_only_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("megatrain_qwen35_persistent_sweep")


def load_case_config(path: str, devices: list[int]) -> tuple[Any, Any]:
    yaml_config = load_yaml_config(path)
    config = load_training_config(path)
    config.devices = list(devices)
    config.device = config.devices[0]
    config.world_size = len(config.devices)
    if config.batch_size % config.world_size:
        raise ValueError(
            f"global batch {config.batch_size} is not divisible by "
            f"{config.world_size} devices"
        )
    if config.dtype is not torch.bfloat16:
        raise ValueError(f"MegaTrain comparison is BF16-only, got {config.dtype}")
    return yaml_config, config


def create_optimizer(model: Any, config: Any, optimizer_type: str) -> Any:
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


def run_case(
    *,
    model: Any,
    optimizer: Any,
    tokenizer: Any,
    config: Any,
    yaml_config: Any,
    timing_output_dir: Path,
    warmup_steps: int,
) -> None:
    if not 0 <= warmup_steps < config.num_steps:
        raise ValueError("warmup steps must be in [0, num_steps)")
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
                "seq=%d step=%d/%d loss=%.4f time=%.3fs tokens/s=%.2f "
                "fwd=%.3fs bwd=%.3fs opt=%.3fs CPU_RSS=%.2fGiB",
                config.max_seq_len,
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
                timing_output_dir,
                rows,
                warmup_steps,
                tokens_per_step,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-manifest", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.sweep_manifest, "megatrain")
    devices = [int(value) for value in manifest["local_devices"].split(",")]
    first_case = manifest["cases"][0]
    first_yaml, first_config = load_case_config(
        first_case["training_config"],
        devices,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a real MegaTrain benchmark")

    torch.manual_seed(first_config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        first_config.model_name,
        trust_remote_code=first_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    source_config = AutoConfig.from_pretrained(
        first_config.model_name,
        trust_remote_code=first_config.trust_remote_code,
    )
    text_config = _extract_text_config(source_config)
    hf_model = AutoModelForCausalLM.from_pretrained(
        first_config.model_name,
        config=text_config,
        dtype=first_config.dtype,
        device_map="cpu",
        trust_remote_code=first_config.trust_remote_code,
        attn_implementation=first_config.attn_implementation,
    )
    assert_text_only_model(hf_model, "full")
    model = CPUMasterModel(hf_model, first_config)
    del hf_model
    optimizer = create_optimizer(
        model,
        first_config,
        get_optimizer_type(first_yaml),
    )

    try:
        for case in manifest["cases"]:
            sequence = int(case["sequence_length"])
            yaml_config, config = load_case_config(
                case["training_config"],
                devices,
            )
            if (
                config.model_name != first_config.model_name
                or config.batch_size != first_config.batch_size
                or config.gradient_accumulation_steps
                != first_config.gradient_accumulation_steps
                or config.max_seq_len > first_config.max_seq_len
            ):
                raise ValueError(
                    f"incompatible MegaTrain persistent case seq={sequence}"
                )
            emit_monitor_phase(manifest, f"seq_{sequence}")
            logger.info(
                "BEGIN persistent sequence=%d; longest buffers remain live",
                sequence,
            )
            try:
                run_case(
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    config=config,
                    yaml_config=yaml_config,
                    timing_output_dir=Path(case["timing_output_dir"]),
                    warmup_steps=int(case["warmup_steps"]),
                )
            except BaseException:
                write_case_exit(case, 1)
                emit_monitor_phase(manifest, "profile_abort")
                raise
            activate_cuda_cache_hold(manifest)
            write_case_cuda_snapshot(manifest, case, "megatrain")
            write_case_exit(case, 0)
            logger.info(
                "END sequence=%d; worker set and CUDA buffers retained",
                sequence,
            )
        emit_monitor_phase(manifest, "profile_release")
    finally:
        model.cleanup()


if __name__ == "__main__":
    main()
