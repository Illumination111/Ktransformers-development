"""LLaMA-Factory entrypoint for the GLM-4.5-Air BF16 server sweep."""

from __future__ import annotations

import os
import sys


EXPECTED_MODEL_TYPE = "glm4_moe"
EXPECTED_ARCHITECTURE = "Glm4MoeForCausalLM"
EXPECTED_LOGICAL_PARAMETERS = 106_852_245_504
SUPPORTED_BACKENDS = {"kt", "deepspeed"}


def _validate_full_finetuning_contract(model) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Validate full tuning while accounting for KT-managed expert weights."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    wrappers = list(getattr(model, "_kt_wrappers", None) or [])
    train_mode = getattr(model, "_kt_train_mode", None)
    model_full_weight_grad = getattr(model, "_kt_full_weight_grad", None)
    if (
        train_mode not in (None, "full")
        or model_full_weight_grad is False
        or not wrappers
    ):
        raise RuntimeError(
            "KTransformers full fine-tuning mode is not active: "
            f"train_mode={train_mode!r}, "
            f"full_weight_grad={model_full_weight_grad!r}, "
            f"wrappers={len(wrappers)}"
        )

    non_full_wrappers = [
        int(getattr(wrapper, "layer_idx", -1))
        for wrapper in wrappers
        if not bool(getattr(wrapper, "_full_weight_grad", False))
    ]
    if non_full_wrappers:
        raise RuntimeError(
            "KTransformers wrappers are missing full-weight gradients: "
            f"layers={non_full_wrappers[:8]}"
        )

    named_parameters = list(model.named_parameters())
    placeholders = [
        (name, parameter)
        for name, parameter in named_parameters
        if bool(getattr(parameter, "_kt_zero_storage", False))
    ]
    if not placeholders:
        raise RuntimeError("KTransformers expert zero-storage placeholders were not found")

    trainable_placeholders = [name for name, parameter in placeholders if parameter.requires_grad]
    if trainable_placeholders:
        raise RuntimeError(
            "KTransformers expert placeholders must stay frozen: "
            f"parameters={trainable_placeholders[:5]}"
        )

    frozen_real_parameters = [
        name
        for name, parameter in named_parameters
        if not bool(getattr(parameter, "_kt_zero_storage", False))
        and not parameter.requires_grad
    ]
    if frozen_real_parameters:
        raise RuntimeError(
            "Full fine-tuning contract failed: real model parameters are frozen: "
            f"parameters={frozen_real_parameters[:5]}"
        )

    try:
        from kt_kernel.sft import get_kt_trainable_params
    except ImportError as exc:
        raise RuntimeError(
            "KTransformers full fine-tuning parameter collector is unavailable"
        ) from exc

    kt_parameters = list(get_kt_trainable_params(model) or [])
    owner_wrappers = [
        wrapper
        for wrapper in wrappers
        if getattr(wrapper, "wrapper", None) is not None
    ]
    is_owner = rank == 0
    if is_owner:
        if len(owner_wrappers) != len(wrappers):
            raise RuntimeError(
                "KTransformers owner rank does not own every expert wrapper: "
                f"owned={len(owner_wrappers)}, total={len(wrappers)}"
            )
        if not kt_parameters:
            raise RuntimeError(
                "KTransformers owner rank exposes no optimizer-visible expert parameters"
            )
        frozen_kt_parameters = [
            index
            for index, parameter in enumerate(kt_parameters)
            if not parameter.requires_grad
        ]
        if frozen_kt_parameters:
            raise RuntimeError(
                "KTransformers optimizer-visible expert parameters are frozen: "
                f"indices={frozen_kt_parameters[:8]}"
            )
    else:
        if owner_wrappers or kt_parameters:
            raise RuntimeError(
                "KTransformers non-owner rank unexpectedly owns expert parameters: "
                f"wrappers={len(owner_wrappers)}, parameters={len(kt_parameters)}"
            )

    placeholder_numel = sum(parameter.numel() for _, parameter in placeholders)
    kt_numel = sum(parameter.numel() for parameter in kt_parameters)
    if is_owner and kt_numel != placeholder_numel:
        raise RuntimeError(
            "KTransformers expert parameter coverage mismatch: "
            f"managed={kt_numel}, placeholders={placeholder_numel}"
        )

    registered_trainable = sum(
        parameter.numel()
        for _, parameter in named_parameters
        if parameter.requires_grad
    )
    registered_total = sum(parameter.numel() for _, parameter in named_parameters)
    logical_trainable = registered_trainable + placeholder_numel
    if logical_trainable != registered_total:
        raise RuntimeError(
            "Full fine-tuning contract failed after KT parameter accounting: "
            f"trainable={logical_trainable}, total={registered_total}"
        )

    return {
        "registered_trainable": registered_trainable,
        "placeholder_numel": placeholder_numel,
        "kt_managed_numel": kt_numel,
        "logical_trainable": logical_trainable,
        "logical_total": registered_total,
        "kt_wrappers": len(wrappers),
    }


def _validate_deepspeed_full_finetuning_contract(
    model,
) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Validate full tuning before DeepSpeed constructs the optimizer."""
    named_parameters = list(model.named_parameters())
    frozen_parameters = [
        name for name, parameter in named_parameters if not parameter.requires_grad
    ]
    if frozen_parameters:
        raise RuntimeError(
            "DeepSpeed full fine-tuning contract failed: parameters are frozen: "
            f"{frozen_parameters[:5]}"
        )
    if getattr(model, "_kt_wrappers", None):
        raise RuntimeError(
            "DeepSpeed full fine-tuning unexpectedly contains KTransformers wrappers"
        )

    def logical_numel(parameter) -> int:  # type: ignore[no-untyped-def]
        return int(getattr(parameter, "ds_numel", parameter.numel()))

    logical_total = sum(logical_numel(parameter) for _, parameter in named_parameters)
    logical_trainable = sum(
        logical_numel(parameter)
        for _, parameter in named_parameters
        if parameter.requires_grad
    )
    if logical_total <= 0 or logical_trainable != logical_total:
        raise RuntimeError(
            "DeepSpeed full fine-tuning parameter coverage mismatch: "
            f"trainable={logical_trainable}, total={logical_total}"
        )
    return {
        "registered_trainable": logical_trainable,
        "placeholder_numel": 0,
        "kt_managed_numel": 0,
        "logical_trainable": logical_trainable,
        "logical_total": logical_total,
        "kt_wrappers": 0,
    }


def _training_backend() -> str:
    backend = os.environ.get("FFT_TRAINING_BACKEND", "").strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise RuntimeError(
            "GLM-4.5-Air LLaMA-Factory entrypoint requires backend "
            f"{sorted(SUPPORTED_BACKENDS)}, got {backend!r}"
        )
    return backend


def _configure_rank_threads() -> None:
    backend = _training_backend()
    if backend == "deepspeed":
        threads = int(os.environ["FFT_CPU_THREADS"])
        if threads <= 0:
            raise RuntimeError("FFT_CPU_THREADS must be positive")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            os.environ[name] = str(threads)
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        print(
            f"[glm45_bf16_threads] rank={rank} role=deepspeed "
            f"cpu_threads={threads}",
            flush=True,
        )
        return

    owner_threads = int(os.environ["FFT_KT_OWNER_THREADS"])
    non_owner_threads = int(os.environ["FFT_KT_NON_OWNER_THREADS"])
    if owner_threads <= 0 or non_owner_threads <= 0:
        raise RuntimeError("KT owner and non-owner thread counts must be positive")

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    threads = owner_threads if rank == 0 else non_owner_threads
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "ACCELERATE_KT_OMP_NUM_THREADS",
        "FFT_CPU_THREADS",
    ):
        os.environ[name] = str(threads)

    print(
        f"[glm45_bf16_threads] rank={rank} "
        f"role={'kt_owner' if rank == 0 else 'non_owner'} "
        f"cpu_threads={threads}",
        flush=True,
    )


def _install_model_contract() -> None:
    """Fail before optimizer construction if a non-GLM checkpoint is loaded."""
    import llamafactory.model as model_api
    from llamafactory.model import loader

    original_load_model = loader.load_model

    def load_glm_model(*args, **kwargs):  # type: ignore[no-untyped-def]
        model = original_load_model(*args, **kwargs)
        get_base_model = getattr(model, "get_base_model", None)
        base_model = get_base_model() if callable(get_base_model) else model
        model_type = getattr(base_model.config, "model_type", None)
        architectures = list(
            getattr(base_model.config, "architectures", None) or []
        )
        if model_type != EXPECTED_MODEL_TYPE:
            raise RuntimeError(
                f"Expected model_type={EXPECTED_MODEL_TYPE!r}, got {model_type!r}"
            )
        if EXPECTED_ARCHITECTURE not in architectures:
            raise RuntimeError(
                f"Expected architecture {EXPECTED_ARCHITECTURE!r}, "
                f"got {architectures!r}"
            )
        if type(base_model).__name__ != EXPECTED_ARCHITECTURE:
            raise RuntimeError(
                f"Expected {EXPECTED_ARCHITECTURE}, constructed "
                f"{type(base_model).__name__}"
            )

        backend = _training_backend()
        if backend == "kt":
            stats = _validate_full_finetuning_contract(model)
        else:
            stats = _validate_deepspeed_full_finetuning_contract(model)
        if stats["logical_total"] != EXPECTED_LOGICAL_PARAMETERS:
            raise RuntimeError(
                "Unexpected GLM-4.5-Air parameter count: "
                f"got={stats['logical_total']}, "
                f"expected={EXPECTED_LOGICAL_PARAMETERS}"
            )
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                "[glm45_model_contract] contract=OK "
                f"backend={backend} "
                f"class={type(base_model).__name__} "
                f"model_type={model_type} "
                f"logical_trainable={stats['logical_trainable']} "
                f"logical_total={stats['logical_total']} "
                f"registered_trainable={stats['registered_trainable']} "
                f"kt_managed={stats['kt_managed_numel']} "
                f"expert_placeholders={stats['placeholder_numel']} "
                f"kt_wrappers={stats['kt_wrappers']}",
                flush=True,
            )
        return model

    loader.load_model = load_glm_model
    model_api.load_model = load_glm_model


def _install_timing() -> None:
    precision = os.environ.get("FFT_PRECISION", "bf16").strip().lower()
    if precision not in {"bf16", "bfloat16"}:
        raise RuntimeError(f"This benchmark is BF16-only, got {precision!r}")
    if os.environ.get("FFT_DISABLE_PERF_PROBES", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError("FFT_DISABLE_PERF_PROBES=1 is required")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1 and rank != 0:
        output = os.environ["FFT_STEP_TIMING_OUT_DIR"]
        os.environ["FFT_STEP_TIMING_OUT_DIR"] = f"{output}.rank{rank}"

    from step_phase_timer import install_step_phase_timing

    install_step_phase_timing()
    print(
        f"[glm45_bf16_timing] rank={rank}/{world_size} "
        f"out={os.environ.get('FFT_STEP_TIMING_OUT_DIR')}",
        flush=True,
    )


def _disable_benchmark_saves() -> None:
    if os.environ.get("FFT_SKIP_FINAL_SAVE", "1").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    from transformers import Trainer

    def skip_save_model(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.is_world_process_zero():
            print("[glm45_bf16] final model save skipped", flush=True)

    def skip_save_state(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.is_world_process_zero():
            print("[glm45_bf16] trainer state save skipped", flush=True)

    Trainer.save_model = skip_save_model
    Trainer.save_state = skip_save_state


def main() -> None:
    _configure_rank_threads()
    _disable_benchmark_saves()
    _install_model_contract()
    _install_timing()
    if sys.argv[1:2] == ["train"]:
        del sys.argv[1]

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
