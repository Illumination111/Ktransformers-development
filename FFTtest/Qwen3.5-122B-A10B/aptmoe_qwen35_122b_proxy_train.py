#!/usr/bin/env python3
"""Qwen3.5-122B-A10B identity wrapper for the shared APTMoE proxy."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


SHARED_DIR = Path(__file__).resolve().parents[1] / "Qwen3.5-35B-A3B"
sys.path.insert(0, str(SHARED_DIR))

import aptmoe_qwen35_proxy_train as shared


PROXY_TAG = "qwen35_122b"
TARGET_MODEL_NAME = "Qwen3.5-122B-A10B-text"
PROXY_ARCHITECTURE = "qwen35_122b_component_isomorphic"
EXPECTED_SHAPE = {
    "num_hidden_layers": 48,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "hidden_size": 3072,
    "moe_intermediate_size": 1024,
}
EXPECTED_PARAMETERS = 122_111_526_912


def _apply_identity(args: Namespace) -> Namespace:
    args.proxy_tag = PROXY_TAG
    args.target_model_name = TARGET_MODEL_NAME
    args.proxy_architecture = PROXY_ARCHITECTURE
    return args


def _validate_target_config(args: Namespace) -> None:
    categories, manifest = shared._expected_category_counts(
        args.model_path,
        PROXY_TAG,
    )
    target = manifest["target"]
    observed_shape = {key: target[key] for key in EXPECTED_SHAPE}
    observed_parameters = sum(categories.values())
    if (
        observed_shape != EXPECTED_SHAPE
        or observed_parameters != EXPECTED_PARAMETERS
    ):
        raise ValueError(
            "model config is not Qwen3.5-122B-A10B text: "
            f"shape={observed_shape}, parameters={observed_parameters}"
        )


def validate_args(args: Namespace) -> None:
    shared.validate_args(_apply_identity(args))
    _validate_target_config(args)


def run(args: Namespace) -> None:
    shared.run(_apply_identity(args))


def main() -> None:
    args = _apply_identity(shared.parse_args())
    validate_args(args)
    if args.audit_only:
        shared._audit_only(args)
        return
    shared.run(args)


if __name__ == "__main__":
    main()
