"""Expose the selected KT source helper to a development test environment."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


MODULE_NAME = "_kt_source_conv3d_compat"
PUBLIC_MODULE_NAME = "kt_kernel.sft.conv3d_compat"
DEFAULT_SOURCE = Path(
    "/mnt/data2/wbw/ktransformers/kt-kernel/python/sft/conv3d_compat.py"
)


def load_conv3d_compat(*, register_as_kt_module: bool = False) -> ModuleType:
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        if register_as_kt_module:
            sys.modules[PUBLIC_MODULE_NAME] = existing
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
    if register_as_kt_module:
        # The installed kt-kernel belongs to the released KT package and must
        # not be replaced just to test a newer additive Python API. Register
        # only that API's fully-qualified module name for this process.
        sys.modules[PUBLIC_MODULE_NAME] = module
    return module
