"""Load the Conv3D compatibility helper from the selected KT source tree."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


MODULE_NAME = "_kt_source_conv3d_compat"
DEFAULT_SOURCE = Path(
    "/mnt/data2/wbw/ktransformers/kt-kernel/python/sft/conv3d_compat.py"
)


def load_conv3d_compat() -> ModuleType:
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing

    source = Path(os.getenv("VLM_KT_CONV3D_COMPAT", str(DEFAULT_SOURCE))).resolve()
    if not source.is_file():
        raise RuntimeError(f"KT Conv3D compatibility source is missing: {source}")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load KT Conv3D compatibility source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
