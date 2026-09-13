"""Expose the selected KT Conv3D helper to a development test process."""

from __future__ import annotations

import importlib.util
import copy
import os
import sys
from pathlib import Path
from types import ModuleType


MODULE_NAME = "_qwen3vl_kt_source_conv3d_compat"
PUBLIC_MODULE_NAME = "kt_kernel.sft.conv3d_compat"
DEFAULT_SOURCE = Path(
    "/mnt/data2/wbw/ktransformers/kt-kernel/python/sft/conv3d_compat.py"
)


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


def install_wrapper_hook() -> ModuleType:
    """Make an installed pre-PR kt-kernel exercise the source compatibility path."""
    module = load_conv3d_compat(register_as_kt_module=True)
    import kt_kernel.sft as kt_sft

    original = kt_sft.wrap_moe_layers_with_kt_wrapper
    if getattr(original, "_kt_local_conv3d_hook", False):
        return module

    def wrapped(model, kt_plugin):
        module.patch_vlm_conv3d(model)
        return original(model, kt_plugin)

    wrapped._kt_local_conv3d_hook = True
    kt_sft.wrap_moe_layers_with_kt_wrapper = wrapped
    return module


def self_test_conv3d_compat() -> dict[str, object]:
    module = load_conv3d_compat()
    import torch
    from types import SimpleNamespace

    if not module._requires_conv3d_patch():
        return {"required": False, "self_test": "not_required", "module_names": []}

    reference = torch.nn.Sequential(
        torch.nn.Conv3d(3, 4, kernel_size=(2, 4, 4), stride=(2, 4, 4), bias=True)
    ).double()
    reference.config = SimpleNamespace(
        model_type="qwen3_vl_moe", vision_config=object()
    )
    candidate = copy.deepcopy(reference)
    reference_input = torch.randn(
        1, 3, 4, 8, 8, dtype=torch.float64, requires_grad=True
    )
    candidate_input = reference_input.detach().clone().requires_grad_(True)
    expected = reference(reference_input)
    module_names = module.patch_vlm_conv3d(candidate)
    actual = candidate(candidate_input)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(candidate_input.grad, reference_input.grad)
    torch.testing.assert_close(candidate[0].weight.grad, reference[0].weight.grad)
    return {"required": True, "self_test": "passed", "module_names": module_names}
