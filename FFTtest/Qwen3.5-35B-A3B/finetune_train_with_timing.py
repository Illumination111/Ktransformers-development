"""LLaMA-Factory entrypoint with minimal, rank-safe step phase timing."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def _configure_kt_rank_threads() -> None:
    """Give the global KT owner a large CPU pool without oversubscribing peers."""
    if os.environ.get("FFT_TRAINING_BACKEND", "").strip().lower() != "kt":
        return

    owner_text = os.environ.get("FFT_KT_OWNER_THREADS")
    non_owner_text = os.environ.get("FFT_KT_NON_OWNER_THREADS")
    if owner_text is None or non_owner_text is None:
        return

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    owner_threads = int(owner_text)
    non_owner_threads = int(non_owner_text)
    if owner_threads <= 0 or non_owner_threads <= 0:
        raise RuntimeError(
            "FFT_KT_OWNER_THREADS and FFT_KT_NON_OWNER_THREADS must be positive"
        )

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
        f"[qwen35_bf16_threads] rank={rank} role={'kt_owner' if rank == 0 else 'non_owner'} "
        f"cpu_threads={threads}",
        flush=True,
    )


def _install_timing() -> None:
    precision = os.environ.get("FFT_PRECISION", "bf16").strip().lower()
    if precision not in {"bf16", "bfloat16"}:
        raise RuntimeError(f"This benchmark is BF16-only, got FFT_PRECISION={precision!r}")
    if os.environ.get("FFT_DISABLE_PERF_PROBES", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError("FFT_DISABLE_PERF_PROBES=1 is required for this benchmark")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1 and rank != 0:
        base = os.environ.get("FFT_STEP_TIMING_OUT_DIR", "step_timing_out")
        os.environ["FFT_STEP_TIMING_OUT_DIR"] = f"{base}.rank{rank}"

    from step_phase_timer import install_step_phase_timing

    install_step_phase_timing()
    print(
        f"[qwen35_bf16_timing] rank={rank}/{world_size} "
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
            print("[qwen35_bf16] final model save skipped for TPS benchmark", flush=True)

    def skip_save_state(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.is_world_process_zero():
            print("[qwen35_bf16] trainer state save skipped for TPS benchmark", flush=True)

    Trainer.save_model = skip_save_model
    Trainer.save_state = skip_save_state


def _run_training_in_current_rank() -> None:
    """Enter LLaMA-Factory without launching a second distributed job.

    The sweep runner already starts this module once per rank with torchrun or
    Accelerate.  Calling ``llamafactory.cli.main`` here would inspect all
    visible GPUs and launch another torchrun from every existing rank.
    """
    if sys.argv[1:2] == ["train"]:
        del sys.argv[1]

    from llamafactory.train.tuner import run_exp

    run_exp()


def _manifest_backend(runtime_backend: str) -> str:
    normalized = runtime_backend.strip().lower()
    return "ktransformers" if normalized == "kt" else normalized


def _load_training_config(path: str) -> dict[str, object]:
    """Load one generated YAML before passing it to LLaMA-Factory.

    ``run_exp(args=...)`` treats a non-None list as already-tokenized CLI
    arguments.  Passing ``[path]`` therefore does not trigger LLaMA-Factory's
    YAML loading branch and leaves required fields such as
    ``model_name_or_path`` unset.
    """
    from omegaconf import OmegaConf

    config_path = Path(path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(config_path))
    if not isinstance(config, dict):
        raise TypeError(
            f"Training config must contain a YAML mapping: {config_path}"
        )
    return config


def _run_persistent_sweep(manifest_path: str) -> None:
    import torch.distributed as dist

    from llamafactory.train.tuner import run_exp
    from persistent_sweep import (
        activate_cuda_cache_hold,
        collect_without_releasing_cuda,
        emit_monitor_phase,
        load_manifest,
        reset_accelerate_state,
        write_case_cuda_snapshot,
        write_case_exit,
    )
    from step_phase_timer import install_step_phase_timing

    runtime_backend = os.environ["FFT_TRAINING_BACKEND"].strip().lower()
    manifest_backend = _manifest_backend(runtime_backend)
    manifest = load_manifest(manifest_path, manifest_backend)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    original_destroy_process_group = dist.destroy_process_group

    def retain_process_group(*args: object, **kwargs: object) -> None:
        print(
            "[persistent-sweep] retaining distributed process group "
            "until profile end",
            flush=True,
        )

    dist.destroy_process_group = retain_process_group
    try:
        for index, case in enumerate(manifest["cases"]):
            sequence = int(case["sequence_length"])
            for name, value in (case.get("environment") or {}).items():
                if value is None:
                    os.environ.pop(str(name), None)
                else:
                    os.environ[str(name)] = str(value)
            os.environ["FFT_STEP_TIMING_OUT_DIR"] = str(
                case["timing_output_dir"]
            )
            os.environ["FFT_STEP_TIMING_WARMUP_STEPS"] = str(
                case["warmup_steps"]
            )
            os.environ["FFT_STEP_TIMING_TOKENS_PER_STEP"] = str(
                case["tokens_per_step"]
            )
            emit_monitor_phase(manifest, f"seq_{sequence}")
            print(
                f"[persistent-sweep] BEGIN seq={sequence} "
                f"case={index + 1}/{len(manifest['cases'])}",
                flush=True,
            )
            recorder = install_step_phase_timing()
            run_exp(args=_load_training_config(str(case["training_config"])))
            recorder.write()
            activate_cuda_cache_hold(manifest)
            write_case_cuda_snapshot(manifest, case, manifest_backend)
            if rank == 0:
                write_case_exit(case, 0)
            print(
                f"[persistent-sweep] END seq={sequence}; CUDA caching "
                "allocator and distributed process group retained",
                flush=True,
            )
            collect_without_releasing_cuda()
            if index + 1 < len(manifest["cases"]):
                reset_accelerate_state()
        emit_monitor_phase(manifest, "profile_release")
    except BaseException:
        if rank == 0:
            write_case_exit(case, 1)
        emit_monitor_phase(manifest, "profile_abort")
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group = original_destroy_process_group
        if dist.is_initialized():
            original_destroy_process_group()


def main() -> None:
    _configure_kt_rank_threads()
    _disable_benchmark_saves()
    from qwen35_text_only import install_text_only_loading

    install_text_only_loading()
    if "--sweep-manifest" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--sweep-manifest", required=True)
        args = parser.parse_args()
        _run_persistent_sweep(args.sweep_manifest)
    else:
        _install_timing()
        _run_training_in_current_rank()


if __name__ == "__main__":
    main()
