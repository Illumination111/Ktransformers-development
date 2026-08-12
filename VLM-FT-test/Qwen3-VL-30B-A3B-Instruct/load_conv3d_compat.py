"""Expose the selected KT Conv3D helper to a development test process."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


MODULE_NAME = "_qwen3vl_kt_source_conv3d_compat"
PUBLIC_MODULE_NAME = "kt_kernel.sft.conv3d_compat"
DEFAULT_SOURCE = Path("/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel/python/sft/conv3d_compat.py")


def load_conv3d_compat(*, register_as_kt_module: bool = False) -> ModuleType:
    module = sys.modules.get(MODULE_NAME)
    if module is None:
        source = Path(os.getenv("VLM_KT_CONV3D_COMPAT", str(DEFAULT_SOURCE))).resolve()
        if not source.is_file():
            raise RuntimeError(f"KT Conv3D compatibility source is missing: {source}")
        spec = importlib.util.spec_from_file_location(MODULE_NAME, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load KT Conv3D compatibility source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
    if register_as_kt_module:
        sys.modules[PUBLIC_MODULE_NAME] = module
    return module
