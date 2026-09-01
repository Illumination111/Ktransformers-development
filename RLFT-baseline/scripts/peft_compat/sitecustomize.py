"""Compatibility shim for PEFT 0.20 with SGLang's torchao 0.9 dependency.

The pure-GPU LoRA path does not use torchao quantization. PEFT probes the
optional torchao LoRA dispatcher unconditionally, but PEFT 0.20 rejects the
older torchao release bundled by SGLang 0.5.8. Make that optional dispatcher
report unavailable while leaving all regular PEFT LoRA modules unchanged.
"""

import importlib.util
import torch


_find_spec = importlib.util.find_spec


def _find_spec_without_old_torchao(name, *args, **kwargs):
    if name == "torchao":
        return None
    return _find_spec(name, *args, **kwargs)


importlib.util.find_spec = _find_spec_without_old_torchao


def _patch_sgl_kernel_optional_symbols():
    """Let SGLang register native Qwen3-MoE with its fused QK path disabled."""
    try:
        import sgl_kernel
    except Exception:
        return
    if hasattr(sgl_kernel, "fused_qk_norm_rope"):
        return

    def fused_qk_norm_rope(*args, **kwargs):
        raise RuntimeError(
            "fused_qk_norm_rope is unavailable in sgl-kernel 0.3.18; "
            "keep SGLang enable_fused_qk_norm_rope disabled"
        )

    sgl_kernel.fused_qk_norm_rope = fused_qk_norm_rope


_patch_sgl_kernel_optional_symbols()


def _patch_sglang_lora_shapes():
    """Adapt SGLang 0.5.8 LoRA torch-native kernels to 3-D hybrid inputs."""
    try:
        from sglang.srt.lora.torch_ops import lora_ops
    except Exception:
        return
    if getattr(lora_ops, "_verl_shape_patch", False):
        return
    original_a = lora_ops.sgemm_lora_a_fwd
    original_b = lora_ops.sgemm_lora_b_fwd

    def sgemm_lora_a_fwd(inputs, *args, **kwargs):
        shape = inputs.shape
        if inputs.ndim == 2:
            return original_a(inputs, *args, **kwargs)
        result = original_a(inputs.reshape(-1, shape[-1]), *args, **kwargs)
        return result.reshape(*shape[:-1], result.shape[-1])

    def sgemm_lora_b_fwd(inputs, *args, **kwargs):
        shape = inputs.shape
        base_output = kwargs.get("base_output")
        if inputs.ndim == 2:
            return original_b(inputs, *args, **kwargs)
        if base_output is not None:
            kwargs["base_output"] = base_output.reshape(-1, base_output.shape[-1])
        result = original_b(inputs.reshape(-1, shape[-1]), *args, **kwargs)
        return result.reshape(*shape[:-1], result.shape[-1])

    lora_ops.sgemm_lora_a_fwd = sgemm_lora_a_fwd
    lora_ops.sgemm_lora_b_fwd = sgemm_lora_b_fwd
    # torch_ops/__init__.py and torch_backend.py import these symbols by
    # value, so update their already-exported references as well.
    try:
        import sglang.srt.lora.torch_ops as torch_ops
        torch_ops.sgemm_lora_a_fwd = sgemm_lora_a_fwd
        torch_ops.sgemm_lora_b_fwd = sgemm_lora_b_fwd
    except Exception:
        pass
    try:
        from sglang.srt.lora.backend import torch_backend
        torch_backend.sgemm_lora_a_fwd = sgemm_lora_a_fwd
        torch_backend.sgemm_lora_b_fwd = sgemm_lora_b_fwd
    except Exception:
        pass
    lora_ops._verl_shape_patch = True


_patch_sglang_lora_shapes()
