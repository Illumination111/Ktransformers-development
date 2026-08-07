#!/usr/bin/env python3
"""LLaMA-Factory train entry with Qwen3.5 text-only (no vision tower) loading.

Usage (via accelerate, see run_train_lora.sh):
  MLS_TEXT_ONLY=1 python train_lora_text_only.py <yaml> [overrides...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    os.environ.setdefault("MLS_TEXT_ONLY", "1")

    from qwen35_text_only import install_text_only_loading

    install_text_only_loading()

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
