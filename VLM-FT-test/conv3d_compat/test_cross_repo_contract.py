"""Local contract checks spanning the KTransformers and LlamaFactory PRs."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_KTRANSFORMERS_ROOT = Path("/mnt/data2/wbw/ktransformers")
DEFAULT_LLAMAFACTORY_ROOT = Path("/mnt/data2/wbw/LLaMA-Factory")
KTRANSFORMERS_ROOT = Path(
    os.getenv("VLM_KTRANSFORMERS_ROOT", str(DEFAULT_KTRANSFORMERS_ROOT))
).resolve()
LLAMAFACTORY_ROOT = Path(
    os.getenv("VLM_LLAMAFACTORY_ROOT", str(DEFAULT_LLAMAFACTORY_ROOT))
).resolve()


def read(relative_path: str, *, root: Path) -> str:
    return (root / relative_path).read_text(encoding="utf-8")


def test_llamafactory_keeps_one_ordinary_kt_requirement():
    requirement = read("requirements/ktransformers.txt", root=LLAMAFACTORY_ROOT)

    assert requirement.splitlines() == ["ktransformers[sft]"]
    assert "vlm-sft" not in requirement
    assert "transformers==" not in requirement
    assert "accelerate==" not in requirement


def test_ktransformers_owns_conv3d_patch_installation():
    wrapper = read("kt-kernel/python/sft/wrapper.py", root=KTRANSFORMERS_ROOT)

    assert "from .conv3d_compat import patch_vlm_conv3d" in wrapper
    assert "patched_conv3d = patch_vlm_conv3d(model)" in wrapper
    assert "ms-swift" not in wrapper


def test_llamafactory_only_checks_the_kt_instance_marker():
    loader = read("src/llamafactory/model/loader.py", root=LLAMAFACTORY_ROOT)

    assert 'getattr(module, "_kt_conv3d_compatible", False)' in loader
    assert "model_args.use_kt" in loader
    assert "is_trainable" in loader
    assert "patch_vlm_conv3d" not in loader
    assert "from kt_kernel.sft.conv3d_compat" not in loader
    assert "import kt_kernel.sft.conv3d_compat" not in loader
