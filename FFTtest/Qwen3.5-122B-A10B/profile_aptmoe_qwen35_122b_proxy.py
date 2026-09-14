#!/usr/bin/env python3
"""Profile the Qwen3.5-122B APTMoE proxy on the target host."""

from __future__ import annotations

import sys
from pathlib import Path


SHARED_DIR = Path(__file__).resolve().parents[1] / "Qwen3.5-35B-A3B"
sys.path.insert(0, str(SHARED_DIR))

from profile_aptmoe_qwen35_proxy import main


if __name__ == "__main__":
    main(
        default_model_path=Path("/mnt/data2/models/Qwen3.5-122B-A10B"),
        default_proxy_tag="qwen35_122b",
    )
