#!/usr/bin/env python3
"""Compare next-token log-probabilities for SGLang+KT and HF+KT.

The comparison is deliberately teacher-forced: both backends receive the
same token ids and are compared on the probability of every next prompt token.
This avoids sampling noise and makes a backend mismatch easy to localize.

The 397B model is supported by the CLI, but the default invocation is the
available Qwen3-30B-A3B checkpoint.  A KT weight directory is required for a
real run.  It may be a BF16/FP8 directory for the BF16/FP8 KT methods or a
converted AMX/GGUF directory for the corresponding method.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_MODELS = {
    "30b": "/mnt/qjh007/models/Qwen3-30B-A3B",
    # Fill this in when the local 397B checkpoint is mounted.
    "397b": "/mnt/qjh007/models/Qwen3.5-397B-A17B",
}
DEFAULT_PROMPTS = [
    "Explain why deterministic teacher-forced inference is useful when validating two language model backends.",
    "Write a short Python function that computes the factorial of a non-negative integer.",
]
DEFAULT_KT_SRC = Path(
    os.environ.get("KT_SRC", "/home/wubowen/ktransformers-RL/ktransformers")
)
DEFAULT_SGLANG_SRC = Path(
    os.environ.get("SGLANG_SRC", str(DEFAULT_KT_SRC / "third_party" / "sglang"))
)
DEFAULT_VERL_SRC = Path(
    os.environ.get(
        "VERL_SRC",
        "/home/wubowen/ktransformers-RL/verl",
    )
)
DEFAULT_OVERLAY = Path(
    os.environ.get(
        "KT_OVERLAY",
        "/home/wubowen/ktransformers-RL",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=sorted(DEFAULT_MODELS), default="30b")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--kt-weight-path", type=Path)
    parser.add_argument(
        "--kt-method",
        default="BF16",
        help="KT method passed to SGLang and documented for HF injection (default: BF16).",
    )
    parser.add_argument("--engine", choices=("both", "sglang", "hf"), default="both")
    parser.add_argument("--prompt", action="append", help="Prompt; may be supplied more than once.")
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--port", type=int, default=30087)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--kt-cpuinfer", type=int)
    parser.add_argument("--kt-threadpool-count", type=int, default=1)
    parser.add_argument("--kt-num-threads", type=int, default=24)
    parser.add_argument("--kt-model-max-length", type=int, default=512)
    parser.add_argument("--kt-num-gpu-experts", type=int)
    parser.add_argument("--kt-gpu-experts-ratio", type=float)
    parser.add_argument("--kt-src", type=Path, default=DEFAULT_KT_SRC)
    parser.add_argument("--sglang-src", type=Path, default=DEFAULT_SGLANG_SRC)
    parser.add_argument("--verl-src", type=Path, default=DEFAULT_VERL_SRC)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--kt-python",
        type=Path,
        default=Path(os.environ.get("KT_PYTHON", sys.executable)),
        help="Python executable for the validated KT runtime; defaults to the active interpreter.",
    )
    parser.add_argument(
        "--hf-device-map",
        choices=("auto", "balanced"),
        default="auto",
        help="Accelerate device map for the sequential HF+KT phase.",
    )
    parser.add_argument("--hf-max-memory-gib", type=int, default=76)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Root directory; each invocation gets its own timestamped experiment directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional exact report path. By default, result.json is written under --output-dir/<experiment>/.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print commands only.")
    parser.add_argument(
        "--probe-plain-hf",
        action="store_true",
        help="Also score with plain Transformers HF (no KT) to isolate KT numerical drift.",
    )
    parser.add_argument(
        "--probe-plain-sglang",
        action="store_true",
        help="Also score with plain SGLang (no KT) to isolate SGLang numerical drift.",
    )
    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def runtime_pythonpath(args: argparse.Namespace) -> str:
    """Build the local KT runtime path without importing the legacy archive."""

    entries = [
        args.verl_src,
        args.sglang_src / "python",
        args.kt_src / "kt-kernel" / "python",
        args.kt_src / "kt-kernel" / "build" / "lib.linux-x86_64-cpython-311",
        args.overlay,
    ]
    existing = [str(path) for path in entries if path.exists()]
    if os.environ.get("PYTHONPATH"):
        existing.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(existing)


def runtime_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = runtime_pythonpath(args)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    environment.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    environment.setdefault("VERL_KT_DISABLE_TORCHAO", "1")
    environment.setdefault("VERL_KT_RELEASE_WEIGHTS_ONLY", "1")
    return environment


def install_runtime_import_path(args: argparse.Namespace) -> None:
    """Make the local KT and SGLang sources take precedence over stale installs."""

    entries = runtime_pythonpath(args).split(os.pathsep)
    for entry in reversed([item for item in entries if item]):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    # Bypass the editable-install finder, which otherwise resolves the stale
    # build/lib tree before sys.path. This registers the selected RL extension
    # as the process-wide ``kt_kernel_ext`` module.
    load_local_kt_kernel(args)


def load_local_kt_kernel(args: argparse.Namespace) -> None:
    package_dir = args.kt_src / "kt-kernel" / "python"
    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        raise FileNotFoundError(f"local kt_kernel package does not exist: {init_file}")
    loaded = sys.modules.get("kt_kernel")
    if loaded is not None and str(getattr(loaded, "__file__", "")).startswith(str(package_dir)):
        return
    for name in list(sys.modules):
        if name == "kt_kernel" or name.startswith("kt_kernel.") or name == "kt_kernel_ext":
            sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        "kt_kernel", init_file, submodule_search_locations=[str(package_dir)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load local kt_kernel package: {init_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kt_kernel"] = module
    spec.loader.exec_module(module)


def validate_runtime_paths(args: argparse.Namespace) -> None:
    required = {
        "KT source": args.kt_src,
        "SGLang source": args.sglang_src,
        "veRL source": args.verl_src,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Validated local runtime paths are missing:\n" + "\n".join(missing)
        )


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_server(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    # Health endpoints intentionally return an empty body, so do not parse them as JSON.
    health_url = f"{base_url}/health_generate"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited during startup with status {process.returncode}")
        try:
            with urlopen(Request(health_url), timeout=10) as response:
                if response.status == 200:
                    return
                time.sleep(1)
        except (OSError, URLError):
            time.sleep(1)
    raise TimeoutError(f"SGLang did not become ready within {timeout:.0f}s")


def launch_sglang(
    args: argparse.Namespace, model_path: Path, kt_path: Path | None, log_path: Path, *, use_kt: bool = True
) -> tuple[subprocess.Popen[bytes], str, Any]:
    command = [
        str(args.kt_python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp-size",
        str(args.tp_size),
        "--trust-remote-code",
        "--disable-cuda-graph",
        "--max-running-requests",
        "1",
        "--attention-backend",
        "torch_native",
        "--sampling-backend",
        "pytorch",
        "--disable-shared-experts-fusion",
        "--skip-server-warmup",
    ]
    if use_kt:
        command += [
            "--kt-method", args.kt_method,
            "--kt-weight-path", str(kt_path),
            "--kt-threadpool-count", str(args.kt_threadpool_count),
        ]
        if args.kt_cpuinfer is None:
            command += ["--kt-cpuinfer", str(args.kt_num_threads)]
        if args.kt_num_gpu_experts is None:
            command += ["--kt-num-gpu-experts", "0"]
        if args.kt_cpuinfer is not None:
            command += ["--kt-cpuinfer", str(args.kt_cpuinfer)]
        if args.kt_num_gpu_experts is not None:
            command += ["--kt-num-gpu-experts", str(args.kt_num_gpu_experts)]
        if args.kt_gpu_experts_ratio is not None:
            command += ["--kt-gpu-experts-ratio", str(args.kt_gpu_experts_ratio)]
    print("[sglang]", " ".join(command))
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=runtime_environment(args),
        start_new_session=True,
    )
    base_url = f"http://{args.host}:{args.port}"
    try:
        wait_for_server(base_url, process, args.timeout)
    except Exception:
        log_file.flush()
        workers = descendant_pids(process.pid)
        signal_pids(workers, signal.SIGTERM)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            signal_pids(workers | {process.pid}, signal.SIGKILL)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        log_file.close()
        raise
    return process, base_url, log_file


def descendant_pids(root_pid: int) -> set[int]:
    """Snapshot descendants before SGLang can re-parent its TP workers."""

    parents: dict[int, int] = {}
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            _, rest = stat.rsplit(")", 1)
            fields = rest.split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    descendants: set[int] = set()
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        children = [pid for pid, ppid in parents.items() if ppid == parent]
        for child in children:
            if child not in descendants:
                descendants.add(child)
                frontier.append(child)
    return descendants


def signal_pids(pids: set[int], signum: signal.Signals) -> None:
    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            continue


def stop_sglang(process: subprocess.Popen[bytes] | None, log_file: Any) -> None:
    if process is not None:
        workers = descendant_pids(process.pid)
        targets = workers | {process.pid}
        signal_pids(workers, signal.SIGTERM)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            signal_pids(targets, signal.SIGKILL)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)
    if log_file is not None:
        log_file.close()


def create_experiment_dir(args: argparse.Namespace, model_path: Path) -> Path:
    if args.output is not None:
        return args.output.expanduser().parent
    stamp = time.strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    name = f"{args.model_size}_{args.kt_method.lower()}_{args.engine}_{stamp}_{unique}"
    path = args.output_dir.expanduser() / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def tokenize_prompts(model_path: Path, prompts: list[str], max_tokens: int) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    encoded = []
    for prompt in prompts:
        ids = tokenizer(prompt, add_special_tokens=True, return_tensors=None)["input_ids"]
        if len(ids) < 2:
            raise ValueError("Each prompt must produce at least two tokens")
        encoded.append(ids[:max_tokens])
    return encoded


def sglang_logprobs(base_url: str, token_ids: list[int]) -> list[float]:
    payload = {
        "input_ids": token_ids,
        "sampling_params": {"temperature": 1.0, "max_new_tokens": 0, "ignore_eos": True},
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": 0,
    }
    response = request_json(f"{base_url}/generate", payload, timeout=1800)
    values = response["meta_info"]["input_token_logprobs"]
    # SGLang reports no meaningful probability for the first input token.
    return [float(item[0]) for item in values[1:]]


def _model_input_device(model: Any, torch_module: Any) -> Any:
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    if device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("HF+KT model has no materialized input device")


def hf_kt_logprobs(
    args: argparse.Namespace, model_path: Path, kt_path: Path, token_ids: list[int]
) -> list[float]:
    """Score through the current transformers-kt/kt-kernel integration.

    The local runtime installs KT through ``transformers.integrations.kt`` and
    ``verl.workers.engine.fsdp.kt_actor_bridge``.  The old archive path used
    ``KTransformersOps`` and is intentionally not imported here.
    """

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    bridge_file = args.verl_src / "verl" / "workers" / "engine" / "fsdp" / "kt_actor_bridge.py"
    if not bridge_file.is_file():
        raise FileNotFoundError(f"KTActorBridge source does not exist: {bridge_file}")
    bridge_spec = __import__("importlib.util").util.spec_from_file_location(
        "rlft_local_kt_actor_bridge", bridge_file
    )
    if bridge_spec is None or bridge_spec.loader is None:
        raise ImportError(f"cannot load KTActorBridge source: {bridge_file}")
    bridge_module = __import__("importlib.util").util.module_from_spec(bridge_spec)
    sys.modules[bridge_spec.name] = bridge_module
    bridge_spec.loader.exec_module(bridge_module)
    KTActorBridge = bridge_module.KTActorBridge

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "eager"
    backend = "AMXBF16" if args.kt_method.upper() == "BF16" else args.kt_method.upper()
    # Probability comparison loads no PEFT adapter, so select the kernel's
    # explicit base-model backend that skips LoRA buffer initialization.
    if backend in {"AMXBF16", "AMXINT8", "AMXINT4"}:
        backend = f"{backend}_SkipLoRA"
    bridge = KTActorBridge(
        enabled=True,
        config={
            "kt_backend": backend,
            "kt_num_gpu_experts": 0,
            "kt_num_threads": args.kt_num_threads,
            "kt_tp_enabled": False,
            "kt_threadpool_count": args.kt_threadpool_count,
            "kt_max_cache_depth": 1,
            "kt_share_backward_bb": False,
            "kt_weight_path": str(kt_path),
            "kt_model_max_length": args.kt_model_max_length,
            "kt_skip_expert_loading": True,
            # This is a base-model probability check; no LoRA adapter is loaded.
            "kt_skip_expert_lora_adaptation": True,
            "kt_force_fused_expert_lora": False,
            "force_fused_expert_lora": False,
        },
    )
    gpu_count = torch.cuda.device_count()
    max_memory = {index: f"{args.hf_max_memory_gib}GiB" for index in range(gpu_count)}
    with bridge.model_load_scope():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=args.hf_device_map,
            max_memory=max_memory or None,
            low_cpu_mem_usage=True,
        )
    wrappers = bridge.verify_wrapped_model(model, minimum_wrappers=1)
    model.eval()
    device = _model_input_device(model, torch)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(len(token_ids), dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[0].float()
    if not wrappers:
        raise RuntimeError("HF+KT model did not expose any KT wrappers")
    # logits[t-1] scores token_ids[t].
    targets = torch.tensor(token_ids[1:], dtype=torch.long, device=logits.device)
    return torch.log_softmax(logits[:-1], dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).cpu().tolist()


def hf_plain_logprobs(
    args: argparse.Namespace, model_path: Path, token_ids: list[int]
) -> list[float]:
    """Score the same teacher-forced input with plain HF, without KT injection."""

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "eager"
    gpu_count = torch.cuda.device_count()
    max_memory = {index: f"{args.hf_max_memory_gib}GiB" for index in range(gpu_count)}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.hf_device_map,
        max_memory=max_memory or None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = _model_input_device(model, torch)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(len(token_ids), dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[0].float()
    targets = torch.tensor(token_ids[1:], dtype=torch.long, device=logits.device)
    values = torch.log_softmax(logits[:-1], dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    del outputs, model
    return values.cpu().tolist()


def summarize(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    count = min(len(reference), len(candidate))
    if count == 0:
        raise ValueError("No comparable log-probabilities returned")
    diffs = [candidate[i] - reference[i] for i in range(count)]
    abs_diffs = [abs(value) for value in diffs]
    return {
        "tokens_compared": count,
        "max_abs_error": max(abs_diffs),
        "mean_abs_error": sum(abs_diffs) / count,
        "rmse": math.sqrt(sum(value * value for value in diffs) / count),
        "mean_signed_error": sum(diffs) / count,
        "length_mismatch": abs(len(reference) - len(candidate)),
        "token_diffs": diffs,
    }


def main() -> int:
    args = parse_args()
    model_path = (args.model_path or Path(DEFAULT_MODELS[args.model_size])).expanduser()
    prompts = args.prompt or DEFAULT_PROMPTS
    require_path(model_path, "model path")
    if args.engine in ("both", "sglang", "hf") and args.kt_weight_path is None:
        raise SystemExit("--kt-weight-path is required for KT comparison; this must be a converted KT weight directory")
    kt_path = args.kt_weight_path.expanduser() if args.kt_weight_path else None
    if kt_path:
        require_path(kt_path, "KT weight path")
    validate_runtime_paths(args)
    report: dict[str, Any] = {
        "model_size": args.model_size,
        "model_path": str(model_path),
        "kt_weight_path": str(kt_path) if kt_path else None,
        "kt_method": args.kt_method,
        "prompts": prompts,
        "cases": [],
        "runtime": {
            "kt_src": str(args.kt_src),
            "sglang_src": str(args.sglang_src),
            "verl_src": str(args.verl_src),
            "overlay": str(args.overlay),
            "python": str(args.kt_python),
            "execution_order": ["sglang", "hf"] if args.engine == "both" else [args.engine],
        },
    }
    if args.dry_run:
        report["status"] = "dry-run"
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "A CUDA device is required for the configured SGLang+KT/HF+KT "
            "comparison. This node reports torch.cuda.is_available() == False."
        )
    install_runtime_import_path(args)
    experiment_dir = create_experiment_dir(args, model_path)
    report_path = args.output.expanduser() if args.output else experiment_dir / "result.json"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    report["experiment_dir"] = str(experiment_dir)
    report["report_path"] = str(report_path)
    report["sglang_log"] = str(experiment_dir / "sglang.log")
    (experiment_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")

    process = None
    log_file = None
    try:
        token_batches = tokenize_prompts(model_path, prompts, args.max_prompt_tokens)
        sglang_values_by_index: list[list[float] | None] = [None] * len(token_batches)
        hf_values_by_index: list[list[float] | None] = [None] * len(token_batches)
        plain_hf_values_by_index: list[list[float] | None] = [None] * len(token_batches)
        plain_sglang_values_by_index: list[list[float] | None] = [None] * len(token_batches)

        if args.engine in ("both", "sglang"):
            process, sglang_url, log_file = launch_sglang(
                args, model_path, kt_path, experiment_dir / "sglang.log", use_kt=True
            )  # type: ignore[arg-type]
            try:
                for index, token_ids in enumerate(token_batches):
                    sglang_values_by_index[index] = sglang_logprobs(sglang_url, token_ids)
            finally:
                stop_sglang(process, log_file)
                process = None
                log_file = None

        if args.probe_plain_sglang:
            process, plain_sglang_url, log_file = launch_sglang(
                args, model_path, None, experiment_dir / "plain_sglang.log", use_kt=False
            )
            try:
                for index, token_ids in enumerate(token_batches):
                    plain_sglang_values_by_index[index] = sglang_logprobs(plain_sglang_url, token_ids)
            finally:
                stop_sglang(process, log_file)
                process = None
                log_file = None

        # SGLang is fully stopped before HF+KT loads.  This mirrors the
        # local probes, which reserve the requested GPUs for one runtime at a
        # time and avoid a false OOM caused by co-locating both backends.
        if args.engine in ("both", "hf"):
            for index, token_ids in enumerate(token_batches):
                hf_values_by_index[index] = hf_kt_logprobs(args, model_path, kt_path, token_ids)  # type: ignore[arg-type]

        if args.probe_plain_hf:
            for index, token_ids in enumerate(token_batches):
                plain_hf_values_by_index[index] = hf_plain_logprobs(args, model_path, token_ids)

        for index, token_ids in enumerate(token_batches):
            values: dict[str, Any] = {"prompt_index": index, "prompt_tokens": len(token_ids)}
            sglang_values = sglang_values_by_index[index]
            hf_values = hf_values_by_index[index]
            if sglang_values is not None:
                values["sglang_logprobs"] = sglang_values
            if hf_values is not None:
                values["hf_logprobs"] = hf_values
            plain_hf_values = plain_hf_values_by_index[index]
            if plain_hf_values is not None:
                values["plain_hf_logprobs"] = plain_hf_values
            plain_sglang_values = plain_sglang_values_by_index[index]
            if plain_sglang_values is not None:
                values["plain_sglang_logprobs"] = plain_sglang_values
            if sglang_values is not None and hf_values is not None:
                values["comparison"] = summarize(sglang_values, hf_values)
            if plain_hf_values is not None and hf_values is not None:
                values["kt_vs_plain_hf"] = summarize(plain_hf_values, hf_values)
            if plain_hf_values is not None and sglang_values is not None:
                values["sglang_vs_plain_hf"] = summarize(plain_hf_values, sglang_values)
            if plain_sglang_values is not None and sglang_values is not None:
                values["kt_sglang_vs_plain_sglang"] = summarize(plain_sglang_values, sglang_values)
            if plain_sglang_values is not None and plain_hf_values is not None:
                values["plain_sglang_vs_plain_hf"] = summarize(plain_sglang_values, plain_hf_values)
            report["cases"].append(values)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        raise
    finally:
        stop_sglang(process, log_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
