"""Audit or activate the development Qwen3-VL-MoE KT architecture shim."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_SOURCE = Path("/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel/python/sft/arch.py")
MODULE_NAME = "_qwen3vl_kt_source_arch"


def _load_source() -> tuple[ModuleType, Path]:
    source = Path(os.getenv("VLM_KT_ARCH_COMPAT", str(DEFAULT_SOURCE))).resolve()
    if not source.is_file():
        raise RuntimeError(f"KT architecture compatibility source is missing: {source}")
    module = sys.modules.get(MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load KT architecture compatibility source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
    return module, source


def audit_kt_architecture(config: Any) -> dict[str, Any]:
    import kt_kernel.sft.arch as installed

    installed_error = None
    try:
        installed_moe = installed.get_moe_arch_config(config)
        installed_prefix = installed._get_layers_prefix(config)
    except Exception as exc:
        installed_moe = None
        installed_prefix = None
        installed_error = f"{type(exc).__name__}: {exc}"

    source, source_path = _load_source()
    source_moe = source.get_moe_arch_config(config)
    source_prefix = source._get_layers_prefix(config)
    return {
        "installed_supported": installed_moe is not None,
        "installed_error": installed_error,
        "installed_prefix": installed_prefix,
        "source_supported": True,
        "source": str(source_path),
        "source_prefix": source_prefix,
        "source_experts": source_moe.expert_num,
        "source_moe_intermediate_size": source_moe.intermediate_size,
        "source_top_k": source_moe.num_experts_per_tok,
    }


def activate_kt_architecture_shim() -> Path:
    """Patch only Python architecture dispatch used by the installed KT binary."""
    import kt_kernel.sft as sft
    import kt_kernel.sft.arch as installed
    import kt_kernel.sft.wrapper as wrapper

    source, source_path = _load_source()
    for name in ("get_moe_arch_config", "_get_layers_prefix"):
        replacement = getattr(source, name)
        setattr(installed, name, replacement)
        setattr(wrapper, name, replacement)
        if hasattr(sft, name):
            setattr(sft, name, replacement)
    return source_path
