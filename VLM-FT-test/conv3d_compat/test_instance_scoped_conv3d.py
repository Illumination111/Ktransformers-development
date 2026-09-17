"""Detailed local validation for the KT-owned torch 2.9 VLM Conv3D fallback."""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


DEFAULT_SOURCE = Path(
    "/mnt/data2/wbw/ktransformers/kt-kernel/python/sft/conv3d_compat.py"
)
SOURCE = Path(os.getenv("VLM_KT_CONV3D_COMPAT", str(DEFAULT_SOURCE))).resolve()
SPEC = importlib.util.spec_from_file_location("local_kt_conv3d_compat", SOURCE)
assert SPEC is not None and SPEC.loader is not None
compat = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compat
SPEC.loader.exec_module(compat)


def make_vlm(
    module: torch.nn.Module, *, model_type: str = "qwen3_vl_moe"
) -> torch.nn.Module:
    model = torch.nn.Sequential(module)
    model.config = SimpleNamespace(model_type=model_type, vision_config=object())
    return model


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("kernel", "input_shape"),
    [
        ((2, 4, 4), (1, 3, 4, 8, 8)),
        ((2, 2, 2), (2, 3, 6, 6, 8)),
        ((1, 4, 4), (2, 3, 3, 8, 12)),
    ],
)
def test_forward_backward_matches_native(monkeypatch, bias, dtype, kernel, input_shape):
    monkeypatch.setattr(compat, "_requires_conv3d_patch", lambda: True)
    reference = make_vlm(
        torch.nn.Conv3d(3, 5, kernel_size=kernel, stride=kernel, bias=bias).to(dtype)
    )
    candidate = copy.deepcopy(reference)
    reference_input = torch.randn(*input_shape, dtype=dtype, requires_grad=True)
    candidate_input = reference_input.detach().clone().requires_grad_(True)
    global_forward = torch.nn.Conv3d.forward

    assert compat.patch_vlm_conv3d(candidate) == ["0"]
    expected = reference(reference_input)
    actual = candidate(candidate_input)
    expected.square().mean().backward()
    actual.square().mean().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(candidate_input.grad, reference_input.grad)
    torch.testing.assert_close(candidate[0].weight.grad, reference[0].weight.grad)
    if bias:
        torch.testing.assert_close(candidate[0].bias.grad, reference[0].bias.grad)
    assert torch.nn.Conv3d.forward is global_forward
    assert compat.is_vlm_conv3d_compatible(candidate)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stride": (1, 4, 4)}, "stride="),
        ({"padding": (0, 1, 0)}, "padding="),
        ({"dilation": (1, 2, 1)}, "dilation="),
        ({"groups": 3, "out_channels": 6}, "groups="),
    ],
)
def test_rejects_unsupported_contract_without_partial_patch(
    monkeypatch, overrides, message
):
    monkeypatch.setattr(compat, "_requires_conv3d_patch", lambda: True)
    options = {
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": (2, 4, 4),
        "stride": (2, 4, 4),
    }
    options.update(overrides)
    model = make_vlm(
        torch.nn.Sequential(
            torch.nn.Conv3d(3, 4, kernel_size=(2, 4, 4), stride=(2, 4, 4)),
            torch.nn.Conv3d(**options),
        )
    )

    with pytest.raises(RuntimeError, match=message):
        compat.patch_vlm_conv3d(model)

    assert all(
        not hasattr(module, "_kt_conv3d_compatible") for module in model.modules()
    )


def test_patch_is_instance_scoped_and_idempotent(monkeypatch):
    monkeypatch.setattr(compat, "_requires_conv3d_patch", lambda: True)
    first = make_vlm(torch.nn.Conv3d(3, 4, kernel_size=2, stride=2))
    second = make_vlm(torch.nn.Conv3d(3, 4, kernel_size=2, stride=2))
    first_original = first[0].forward
    second_original = second[0].forward

    assert compat.patch_vlm_conv3d(first) == ["0"]
    patched_forward = first[0].forward
    assert compat.patch_vlm_conv3d(first) == ["0"]

    assert first[0].forward == patched_forward
    assert first[0]._kt_original_conv3d_forward == first_original
    assert second[0].forward == second_original
    assert not hasattr(second[0], "_kt_conv3d_compatible")


@pytest.mark.parametrize("model_type", ["qwen3_moe", "unsupported_vlm"])
def test_non_vlm_and_unsupported_vlm_are_unchanged(monkeypatch, model_type):
    monkeypatch.setattr(compat, "_requires_conv3d_patch", lambda: True)
    model = make_vlm(
        torch.nn.Conv3d(3, 4, kernel_size=2, stride=2), model_type=model_type
    )
    original = model[0].forward

    assert compat.patch_vlm_conv3d(model) == []
    assert model[0].forward == original
    assert not hasattr(model[0], "_kt_conv3d_compatible")


def test_non_torch_29_path_is_a_noop(monkeypatch):
    monkeypatch.setattr(compat, "_requires_conv3d_patch", lambda: False)
    model = make_vlm(torch.nn.Conv3d(3, 4, kernel_size=2, stride=2))

    assert compat.patch_vlm_conv3d(model) == []
    assert compat.is_vlm_conv3d_compatible(model)
