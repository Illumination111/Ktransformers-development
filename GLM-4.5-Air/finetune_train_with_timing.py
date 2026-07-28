"""LLaMA-Factory entrypoint for the GLM-4.5-Air BF16 server sweep."""

from __future__ import annotations

import os
import sys


EXPECTED_MODEL_TYPE = "glm4_moe"
EXPECTED_ARCHITECTURE = "Glm4MoeForCausalLM"


def _configure_kt_rank_threads() -> None:
    if os.environ.get("FFT_TRAINING_BACKEND", "").strip().lower() != "kt":
        raise RuntimeError("GLM-4.5-Air sweep supports only the KTransformers backend")

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

        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        if trainable != total:
            raise RuntimeError(
                "Full fine-tuning contract failed: "
                f"trainable={trainable}, total={total}"
            )
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                "[glm45_model_contract] contract=OK "
                f"class={type(base_model).__name__} "
                f"model_type={model_type} trainable={trainable} total={total}",
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
    _configure_kt_rank_threads()
    _disable_benchmark_saves()
    _install_model_contract()
    _install_timing()
    if sys.argv[1:2] == ["train"]:
        del sys.argv[1]

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
