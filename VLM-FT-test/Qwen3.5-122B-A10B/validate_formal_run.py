#!/usr/bin/env python3
"""Validate a completed multi-step Qwen3.5 VLM LoRA formal test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from validate_adapter_output import validate_adapter


VALID_LORA_SCOPES = ("text", "vision", "all")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing formal-test result: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def finite_metric(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"missing or invalid {name}: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"non-finite {name}: {number}")
    return number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--lora-scope", choices=VALID_LORA_SCOPES, default="text")
    args = parser.parse_args()
    if args.expected_steps < 10:
        parser.error("formal validation requires at least 10 optimizer steps")

    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "model_output"
    state = load_json(output_dir / "trainer_state.json")
    train_results = load_json(output_dir / "train_results.json")
    eval_results = load_json(output_dir / "eval_results.json")
    global_step = int(state.get("global_step", -1))
    if global_step != args.expected_steps:
        raise RuntimeError(
            f"optimizer-step mismatch: expected={args.expected_steps}, actual={global_step}"
        )

    history = state.get("log_history") or []
    losses = [
        finite_metric(item["loss"], "step loss") for item in history if "loss" in item
    ]
    if len(losses) < args.expected_steps:
        raise RuntimeError(
            f"incomplete loss history: expected at least {args.expected_steps}, got {len(losses)}"
        )
    if any(loss < 0 for loss in losses):
        raise RuntimeError("cross-entropy loss must not be negative")

    train_loss = finite_metric(train_results.get("train_loss"), "train_loss")
    train_runtime = finite_metric(train_results.get("train_runtime"), "train_runtime")
    eval_loss = finite_metric(eval_results.get("eval_loss"), "eval_loss")
    if train_runtime <= 0:
        raise RuntimeError(f"train_runtime must be positive: {train_runtime}")

    log_path = run_dir / "train.log"
    if not log_path.is_file():
        raise RuntimeError(f"missing training log: {log_path}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if f"scope={args.lora_scope}" not in log_text:
        raise RuntimeError(
            f"training log does not confirm requested LoRA scope: {args.lora_scope}"
        )
    conv3d_checks = log_text.count("[qwen35_vlm_conv3d]")
    vlm_contract_checks = log_text.count("[qwen35_vlm_contract] OK")
    gradient_checks = log_text.count("[qwen35_vlm_functional] GRADIENT_OK")
    optimizer_checks = log_text.count("[qwen35_vlm_functional] OPTIMIZER_OK")
    contract_passes = log_text.count("[qwen35_vlm_functional] PASS")
    if conv3d_checks < 1 or vlm_contract_checks < 1:
        raise RuntimeError(
            "missing Conv3D/VLM contract checks: "
            f"conv3d={conv3d_checks}, vlm_contract={vlm_contract_checks}"
        )
    if gradient_checks < args.expected_steps or optimizer_checks < args.expected_steps:
        raise RuntimeError(
            "missing per-step functional checks: "
            f"gradient={gradient_checks}, optimizer={optimizer_checks}, expected>={args.expected_steps}"
        )
    if contract_passes < 1:
        raise RuntimeError("training log has no final VLM functional PASS marker")

    adapter = validate_adapter(output_dir, args.lora_scope)
    summary = {
        "status": "passed",
        "run_dir": str(run_dir),
        "lora_scope": args.lora_scope,
        "global_step": global_step,
        "logged_losses": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "train_runtime": train_runtime,
        "conv3d_checks": conv3d_checks,
        "vlm_contract_checks": vlm_contract_checks,
        "gradient_checks": gradient_checks,
        "optimizer_checks": optimizer_checks,
        "contract_passes": contract_passes,
        "adapter": adapter,
        "quality_claim": "not_applicable_for_six_row_demo",
    }
    summary_path = run_dir / "formal_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "[qwen35_vlm_formal] PASS "
        f"scope={args.lora_scope} steps={global_step} losses={len(losses)} train_loss={train_loss:.6f} "
        f"eval_loss={eval_loss:.6f} adapter_lora_tensors={adapter['lora_tensors']}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
