#!/usr/bin/env python3
"""Isolate FP32 -> BF16 storage rounding from the AMX BF16 kernel error.

The test uses the native AMXBF16_MOE binding with small deterministic tensors.
It compares:

* fp32_ideal: BF16 input/weights converted to FP32, no intermediate BF16 store;
* bf16_store: the same FP32 calculations with an explicit BF16 round-trip at
  every storage boundary used by the BF16 MoE path;
* amx: the actual AMXBF16_MOE output.

This is deliberately a MoE micro-test, rather than a whole-model logprob test.
It therefore does not mix in attention, layernorm, router, or vocabulary-head
differences.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


DEFAULT_KT_PYTHON = Path("/home/wubowen/ktransformers-RL/ktransformers/kt-kernel/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kt-python", type=Path, default=DEFAULT_KT_PYTHON)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "bf16_storage_probe.json",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--expert-num", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--qlens", type=int, nargs="+", default=[1, 8, 32])
    return parser.parse_args()


def load_extension(kt_python: Path):
    sys.path.insert(0, str(kt_python))
    try:
        import kt_kernel_ext  # type: ignore
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            f"Cannot import kt_kernel_ext from {kt_python}. Build/install the RL KT extension first."
        ) from exc
    return kt_kernel_ext


def bf16_store(value: torch.Tensor) -> torch.Tensor:
    """Model a write to a BF16 buffer followed by a FP32 read."""

    return value.to(torch.bfloat16).float()


def stats(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - reference.float()).abs()
    ref_abs = reference.float().abs()
    return {
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff))),
        "relative_l2": float(torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(ref_abs) + 1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(actual.flatten().float(), reference.flatten().float(), dim=0)),
    }


def make_inputs(args: argparse.Namespace, qlen: int):
    generator = torch.Generator(device="cpu").manual_seed(20260821 + qlen)
    weights = []
    for _ in range(3):
        weights.append(
            (torch.randn(
                args.expert_num,
                args.intermediate_size if len(weights) < 2 else args.hidden_size,
                args.hidden_size if len(weights) < 2 else args.intermediate_size,
                generator=generator,
                dtype=torch.float32,
            ) / 8).to(torch.bfloat16).contiguous()
        )
    gate, up, down = weights
    hidden = (torch.randn(qlen, args.hidden_size, generator=generator, dtype=torch.float32) / 2).to(torch.bfloat16)
    expert_ids = torch.empty((qlen, args.top_k), dtype=torch.int64)
    for token in range(qlen):
        expert_ids[token] = torch.randperm(args.expert_num, generator=generator)[: args.top_k]
    routing = torch.softmax(torch.randn(qlen, args.top_k, generator=generator, dtype=torch.float32), dim=-1).contiguous()
    return hidden.contiguous(), expert_ids.contiguous(), routing, gate, up, down


def references(hidden, expert_ids, routing, gate, up, down):
    x = hidden.float()
    ideal = torch.zeros_like(x)
    stored = torch.zeros_like(x)
    for token in range(hidden.shape[0]):
        for slot in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, slot])
            gw = gate[expert].float()
            uw = up[expert].float()
            dw = down[expert].float()

            gate_ideal = x[token] @ gw.T
            up_ideal = x[token] @ uw.T
            intermediate_ideal = torch.nn.functional.silu(gate_ideal) * up_ideal
            expert_ideal = intermediate_ideal @ dw.T

            gate_stored = bf16_store(gate_ideal)
            up_stored = bf16_store(up_ideal)
            intermediate_stored = bf16_store(torch.nn.functional.silu(gate_stored) * up_stored)
            expert_stored = bf16_store(intermediate_stored @ dw.T)

            weight = routing[token, slot]
            ideal[token] += expert_ideal * weight
            stored[token] += expert_stored * weight
    # The native binding writes the final MoE output into a BF16 output buffer.
    # Include that final store so the software model has the same observable
    # dtype as the AMX result.
    return ideal, bf16_store(stored)


def run_amx(kt_kernel_ext, args, cpu_infer, hidden, expert_ids, routing, gate, up, down):
    physical_to_logical = torch.arange(args.expert_num, dtype=torch.int64).contiguous()
    config = kt_kernel_ext.moe.MOEConfig(
        args.expert_num,
        args.top_k,
        args.hidden_size,
        args.intermediate_size,
        0,
    )
    config.max_len = hidden.shape[0]
    config.gate_proj = gate.data_ptr()
    config.up_proj = up.data_ptr()
    config.down_proj = down.data_ptr()
    config.gate_scale = 0
    config.up_scale = 0
    config.down_scale = 0
    config.pool = cpu_infer.backend_
    moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()

    output = torch.empty_like(hidden)
    batch = torch.tensor([hidden.shape[0]], dtype=torch.int32)
    cpu_infer.submit(
        moe.forward_task(
            batch.data_ptr(),
            args.top_k,
            expert_ids.data_ptr(),
            routing.data_ptr(),
            hidden.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    cpu_infer.sync()
    return output.float(), moe


def main() -> int:
    args = parse_args()
    if args.expert_num <= 0 or args.top_k <= 0 or args.top_k > args.expert_num:
        raise ValueError("Require 0 < top-k <= expert-num")
    if args.hidden_size % 32 or args.intermediate_size % 32:
        raise ValueError("hidden-size and intermediate-size must be multiples of 32 for AMX tiles")

    kt_kernel_ext = load_extension(args.kt_python)
    print(f"kt_kernel_ext={getattr(kt_kernel_ext, '__file__', '<unknown>')}")
    print(f"threads={args.threads}, experts={args.expert_num}, top_k={args.top_k}")
    cpu_infer = kt_kernel_ext.CPUInfer(args.threads)
    # Keep the native objects alive until all qlen cases complete. Recreating
    # CPUInfer repeatedly in one process can trigger a double-free in older
    # extension builds during static NUMA-pool teardown.
    native_moes = []
    report: dict[str, object] = {
        "config": {
            "threads": args.threads,
            "expert_num": args.expert_num,
            "top_k": args.top_k,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "qlens": args.qlens,
        },
        "results": [],
    }
    for qlen in args.qlens:
        hidden, expert_ids, routing, gate, up, down = make_inputs(args, qlen)
        ideal, stored = references(hidden, expert_ids, routing, gate, up, down)
        actual, native_moe = run_amx(kt_kernel_ext, args, cpu_infer, hidden, expert_ids, routing, gate, up, down)
        native_moes.append(native_moe)
        row = {
            "qlen": qlen,
            "storage_rounding": stats(stored, ideal),
            "amx_extra": stats(actual, stored),
            "total": stats(actual, ideal),
        }
        report["results"].append(row)  # type: ignore[union-attr]
        print(
            f"qlen={qlen:3d} | storage max={row['storage_rounding']['max_abs']:.6g} "
            f"rel_l2={row['storage_rounding']['relative_l2']:.6g} | "
            f"amx max={row['amx_extra']['max_abs']:.6g} rel_l2={row['amx_extra']['relative_l2']:.6g} | "
            f"total max={row['total']['max_abs']:.6g}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
