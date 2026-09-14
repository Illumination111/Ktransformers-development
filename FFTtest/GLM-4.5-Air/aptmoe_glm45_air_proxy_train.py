#!/usr/bin/env python3
"""Run the GLM-4.5-Air component-isomorphic BF16 APTMoE proxy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from glm45_air_proxy_spec import EXPECTED_PARAMETERS, build_manifest

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aptmoe-root",
        type=Path,
        default=Path("/mnt/data2/wbw/APTMoE-baseline"),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--step-timing-output-dir", type=Path, required=True)
    parser.add_argument("--route-trace", type=Path)
    parser.add_argument("--lookup-table", type=Path)
    parser.add_argument("--deployment-profile", choices=("server",), default="server")
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--prefetch-portion", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--allow-synthetic-routing", action="store_true")
    parser.add_argument("--allow-unprofiled-placement", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "sequence_length",
        "num_gpus",
        "global_batch_size",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.sequence_length < 32 or args.sequence_length > 4096:
        raise ValueError("sequence_length must be in [32, 4096]")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup_steps must be in [0, steps)")
    if not args.audit_only and args.warmup_steps == 0:
        raise ValueError("a measured run requires at least one warmup step")
    if args.global_batch_size != args.num_gpus * args.per_device_batch_size:
        raise ValueError("global batch must equal GPUs * per-device batch")
    if args.deployment_profile == "server" and args.num_gpus != 8:
        raise ValueError("server protocol requires 8 GPUs")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if not args.audit_only and not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _configure_distributed(
    args: argparse.Namespace,
) -> tuple[int, int, int, bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != args.num_gpus:
        raise RuntimeError(
            f"torchrun world_size={world_size}, expected={args.num_gpus}"
        )
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(max(1, int(os.environ.get("FFT_CPU_THREADS", "1"))))
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    return rank, world_size, local_rank, initialized_here


def run(args: argparse.Namespace) -> None:
    aptmoe_root = args.aptmoe_root.resolve()
    sys.path.insert(0, str(aptmoe_root))
    sys.path.insert(0, str(SCRIPT_DIR))
    from data import SFTDataLoader, load_sft_dataset, load_tokenizer
    from data.sft_dataset import tokenize_and_mask
    from Runtime.OffloadRuntime import offload as offload_runtime

    from glm45_air_aptmoe_proxy_components import load_glm_config
    from glm45_aptmoe_proxy import ProxyPlacementSolver, RouteController
    from glm45_aptmoe_proxy.runtime import (
        ProxyPipelineRuntime,
        build_proxy_pipeline,
        global_parameter_counts,
        local_parameter_counts,
        run_full_update_steps,
    )

    rank, world_size, local_rank, initialized_here = _configure_distributed(args)
    config = load_glm_config(args.model_path)
    target_manifest = build_manifest(args.model_path)
    expected = {
        key: value["parameters"]
        for key, value in target_manifest["target"]["components"].items()
        if key != "model_total"
    }
    tokens_per_microbatch = args.global_batch_size * args.sequence_length
    routes = RouteController(
        num_layers=config.num_hidden_layers,
        first_moe_layer=config.first_k_dense_replace,
        num_experts=config.n_routed_experts,
        top_k=config.num_experts_per_tok,
        sequence_length=args.sequence_length,
        tokens_per_microbatch=tokens_per_microbatch,
        microbatches_per_step=args.gradient_accumulation_steps,
        expected_patterns=args.warmup_steps
        * args.gradient_accumulation_steps,
        trace_path=args.route_trace,
        allow_synthetic=args.allow_synthetic_routing,
    )
    placement = ProxyPlacementSolver(
        config.n_routed_experts,
        1,
        lookup_path=args.lookup_table,
        prefetch_portion=args.prefetch_portion,
        allow_unprofiled=args.allow_unprofiled_placement,
        expected_profile=args.deployment_profile,
        required_max_tokens=tokens_per_microbatch,
    )
    offload_runtime.prefetch_portion = args.prefetch_portion
    tokenizer = load_tokenizer(str(args.model_path), trust_remote_code=True)
    dataset = tokenize_and_mask(
        load_sft_dataset(
            args.dataset_name, str(args.dataset_dir), max_samples=-1
        ),
        tokenizer,
        args.sequence_length,
        "qwen",
    )
    data_loader = SFTDataLoader(
        tokenized_examples=dataset,
        batch_size=args.global_batch_size,
        cutoff_len=args.sequence_length,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        seed=args.seed,
        num_workers=0,
    )
    if len(data_loader) == 0:
        raise RuntimeError("dataset has fewer examples than the global batch")
    module_list, _ = build_proxy_pipeline(
        config=config,
        world_size=world_size,
        local_rank=local_rank,
        global_rank=rank,
        routes=routes,
        placement_solver=placement,
        seed=args.seed,
    )
    non_bf16 = [
        name
        for shard in module_list
        if shard is not None
        for name, parameter in shard.model_shard.named_parameters()
        if parameter.dtype != torch.bfloat16
    ]
    if non_bf16:
        raise RuntimeError(f"proxy contains non-BF16 parameters: {non_bf16[:8]}")
    actual = global_parameter_counts(local_parameter_counts(module_list))
    if actual != expected or sum(actual.values()) != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"proxy parameter mismatch: actual={actual}, expected={expected}"
        )
    runtime_config = SimpleNamespace(
        bf16=True,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0e-8,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        warmup_steps=0,
        warmup_ratio=0.0,
        total_training_steps=args.steps,
    )
    runtime = ProxyPipelineRuntime(
        batch_size=args.global_batch_size,
        num_chunks=1,
        seq_length=args.sequence_length,
        model_dim=config.hidden_size,
        hidden_dim=config.moe_intermediate_size,
        module_list=module_list,
        world_size=world_size,
        local_size=world_size,
        global_rank=rank,
        num_stages=config.num_hidden_layers,
        pipeline="APTMoE",
        fwd_only=False,
        lora_mode=False,
        data_loader=data_loader,
        config=runtime_config,
        sft_mode=True,
    )
    if runtime.total_params != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"optimizer scope={runtime.total_params:,}, "
            f"expected={EXPECTED_PARAMETERS:,}"
        )
    run_dir = args.step_timing_output_dir.resolve().parent
    smoke = args.allow_synthetic_routing or args.allow_unprofiled_placement
    if rank == 0:
        _write_json(
            run_dir / "proxy_manifest.json",
            {
                "schema_version": 1,
                "benchmark_class": "deployment_proxy",
                "result_validity": (
                    "SMOKE_ONLY" if smoke else "formal_deployment_proxy"
                ),
                "target_model": "GLM-4.5-Air",
                "proxy_architecture": "glm45_air_component_isomorphic",
                "weight_source": "deterministic_random_bf16_initialization",
                "checkpoint_compatible": False,
                "exact_model_claim_allowed": False,
                "real_forward_backward_optimizer_update": True,
                "parameter_count": sum(actual.values()),
                "parameter_categories": actual,
                "route": routes.manifest(),
                "placement": placement.manifest(),
                "runtime_versions": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "transformers": _package_version("transformers"),
                    "cuda": torch.version.cuda,
                    "attention_implementation": "sdpa",
                },
                "target_contract": target_manifest,
            },
        )
    tokens_per_step = (
        args.global_batch_size
        * args.sequence_length
        * args.gradient_accumulation_steps
    )
    _, verification = run_full_update_steps(
        runtime=runtime,
        module_list=module_list,
        routes=routes,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        tokens_per_step=tokens_per_step,
        timing_output_dir=args.step_timing_output_dir.resolve(),
    )
    if rank == 0:
        _write_json(run_dir / "full_update_verification.json", verification)
    valid = torch.tensor(
        [int(rank != 0 or bool(verification and verification["valid_full_update"]))],
        device="cuda",
    )
    dist.broadcast(valid, src=0)
    if not valid.item():
        raise RuntimeError("full-update audit failed")
    dist.barrier()
    if initialized_here:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.audit_only:
        print(json.dumps(build_manifest(args.model_path), indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
