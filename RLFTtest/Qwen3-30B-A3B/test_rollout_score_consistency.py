#!/usr/bin/env python3
"""Compare SGLang+KT rollout logprobs with HuggingFace+KT score logprobs.

The comparison is deliberately token-aligned: the response sampled by the
rollout is replayed by the score forward.  This is the probability-consistency
check required by the RLFT old_logprob/new_logprob contract, not a text
generation comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "/mnt/data3/models/Qwen3-30B-A3B"
DEFAULT_PROMPT = "Explain why teacher-forced token probabilities are useful in on-policy RL training."
DEFAULT_CACHE_ROOT = "/mnt/data2/wbw/.cache/kt-RLFT"


def _configure_cache_dirs() -> None:
    """Keep runtime caches on the data volume even when called directly."""
    root = os.environ.setdefault("KT_RLFT_CACHE_ROOT", DEFAULT_CACHE_ROOT)
    defaults = {
        "XDG_CACHE_HOME": f"{root}/xdg",
        "HF_HOME": f"{root}/huggingface",
        "TRANSFORMERS_CACHE": f"{root}/huggingface/transformers",
        "HUGGINGFACE_HUB_CACHE": f"{root}/huggingface/hub",
        "TORCH_HOME": f"{root}/torch",
        "TRITON_CACHE_DIR": f"{root}/triton",
        "PIP_CACHE_DIR": f"{root}/pip",
        "CUDA_CACHE_PATH": f"{root}/cuda",
        "TMPDIR": f"{root}/tmp",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    for value in defaults.values():
        Path(value).mkdir(parents=True, exist_ok=True)


_configure_cache_dirs()


@dataclass(frozen=True)
class Comparison:
    active_mask: list[bool]
    token_ids: list[int]
    rollout_logprobs: list[float]
    score_logprobs: list[float]
    diffs: list[float]
    ratios: list[float]
    max_abs_logprob_diff: float
    mean_abs_logprob_diff: float
    max_abs_ratio_delta: float
    passed: bool


def compare_logprobs(
    token_ids: list[int],
    rollout_logprobs: list[float],
    score_logprobs: list[float],
    active_mask: list[bool] | None = None,
    abs_tol: float = 1e-3,
    ratio_tol: float = 1e-3,
) -> Comparison:
    """Compare only active response positions and fail on alignment errors."""
    n = len(token_ids)
    if len(rollout_logprobs) != n or len(score_logprobs) != n:
        raise ValueError(
            "token/logprob length mismatch: "
            f"tokens={n}, rollout={len(rollout_logprobs)}, score={len(score_logprobs)}"
        )
    mask = list(active_mask) if active_mask is not None else [True] * n
    if len(mask) != n:
        raise ValueError(f"active mask length {len(mask)} != token length {n}")
    active = [i for i, value in enumerate(mask) if value]
    if not active:
        raise ValueError("active response mask contains no tokens")

    diffs = [float(score_logprobs[i] - rollout_logprobs[i]) for i in range(n)]
    ratios = [float(math.exp(diffs[i])) for i in range(n)]
    active_diffs = [abs(diffs[i]) for i in active]
    active_ratio_deltas = [abs(ratios[i] - 1.0) for i in active]
    max_diff = max(active_diffs)
    mean_diff = sum(active_diffs) / len(active_diffs)
    max_ratio_delta = max(active_ratio_deltas)
    return Comparison(
        active_mask=mask,
        token_ids=[int(x) for x in token_ids],
        rollout_logprobs=[float(x) for x in rollout_logprobs],
        score_logprobs=[float(x) for x in score_logprobs],
        diffs=diffs,
        ratios=ratios,
        max_abs_logprob_diff=max_diff,
        mean_abs_logprob_diff=mean_diff,
        max_abs_ratio_delta=max_ratio_delta,
        passed=max_diff <= abs_tol and max_ratio_delta <= ratio_tol,
    )


def _parse_args() -> argparse.Namespace:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    inferred_tp = len([item for item in visible_devices.split(",") if item.strip()])
    if not inferred_tp:
        inferred_tp = 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "16")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SGLANG_PORT", "30000")))
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.environ.get("SGLANG_TP_SIZE", str(inferred_tp))),
        help="SGLang tensor parallel size; defaults to the number of visible CUDA devices.",
    )
    parser.add_argument("--base-url", default=os.environ.get("SGLANG_BASE_URL"))
    parser.add_argument("--server-start-timeout", type=float, default=900.0)
    parser.add_argument("--kt-num-threads", type=int, default=int(os.environ.get("KT_NUM_THREADS", "16")))
    parser.add_argument("--kt-threadpool-count", type=int, default=int(os.environ.get("KT_THREADPOOL_COUNT", "1")))
    parser.add_argument("--kt-num-gpu-experts", type=int, default=int(os.environ.get("KT_NUM_GPU_EXPERTS", "32")))
    parser.add_argument("--abs-tol", type=float, default=float(os.environ.get("ABS_TOL", "1e-3")))
    parser.add_argument("--ratio-tol", type=float, default=float(os.environ.get("RATIO_TOL", "1e-3")))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _timestamp_dir(explicit: Path | None) -> Path:
    if explicit:
        explicit.mkdir(parents=True, exist_ok=True)
        return explicit
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(__file__).resolve().parent / "test_log" / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--kt-weight-path",
        args.model,
        "--kt-cpuinfer",
        str(args.kt_num_threads),
        "--kt-threadpool-count",
        str(args.kt_threadpool_count),
        "--kt-num-gpu-experts",
        str(args.kt_num_gpu_experts),
        "--kt-method",
        "BF16",
        "--attention-backend",
        "flashinfer",
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.80",
        "--chunked-prefill-size",
        "8192",
        "--max-running-requests",
        "2",
        "--served-model-name",
        Path(args.model).name,
        "--enable-mixed-chunk",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--enable-p2p-check",
        "--disable-shared-experts-fusion",
    ]


def _wait_for_server(base_url: str, process: subprocess.Popen[str], timeout: float) -> None:
    import requests

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited early with code {process.returncode}")
        try:
            response = requests.get(base_url + "/health", timeout=3)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang server did not become healthy within {timeout:.0f}s")


def _assert_port_available(port: int) -> None:
    """Avoid mistaking an unrelated server on the same port for our child."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(
                f"SGLang port {port} is already in use. Choose another port with "
                "SGLANG_PORT=<port>, or use an existing server with "
                "SGLANG_BASE_URL=http://127.0.0.1:<port>."
            )


def _run_rollout(base_url: str, prompt_ids: list[int], args: argparse.Namespace) -> dict[str, Any]:
    import requests

    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": prompt_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": True,
            "top_logprobs_num": 0,
            "logprob_start_len": len(prompt_ids),
        },
        timeout=max(600, args.server_start_timeout),
    )
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list):
        if len(result) != 1:
            raise ValueError(f"expected one SGLang result, got {len(result)}")
        result = result[0]
    meta = result.get("meta_info", {})
    records = meta.get("output_token_logprobs")
    if not records:
        raise ValueError("SGLang response has no output_token_logprobs")
    token_ids = [int(record[1]) for record in records]
    logprobs = [float(record[0]) for record in records]
    if len(token_ids) != args.max_new_tokens:
        raise ValueError(
            f"rollout returned {len(token_ids)} tokens, expected {args.max_new_tokens}; "
            "use ignore_eos or inspect stop conditions"
        )
    return {"token_ids": token_ids, "logprobs": logprobs, "response": result}


def _load_hg_kt_score(model_path: str, full_ids: list[int], prompt_len: int, repo_root: Path) -> list[float]:
    import torch
    from transformers import AutoConfig

    # load_kt_model is the repository's supported standalone HG+KT entrypoint.
    sys.path.insert(0, str(repo_root / "ktransformers" / "kt-kernel" / "python"))
    from kt_kernel.sft import KTConfig, load_kt_model

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    kt_config = KTConfig(
        kt_backend="AMXBF16",
        kt_num_threads=int(os.environ.get("KT_NUM_THREADS", "16")),
        kt_threadpool_count=int(os.environ.get("KT_THREADPOOL_COUNT", "1")),
        kt_num_gpu_experts=int(os.environ.get("KT_NUM_GPU_EXPERTS", "32")),
        # The HF checkpoint is the source for BF16 expert tensors.  Passing it
        # as kt_weight_path would select the .kt-file loader instead.
        kt_expert_checkpoint_path=model_path,
        kt_skip_expert_loading=True,
        kt_model_max_length=max(4096, len(full_ids)),
    )
    model = load_kt_model(
        config=config,
        kt_plugin=kt_config,
        model_name_or_path=model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits
    response_len = len(full_ids) - prompt_len
    positions = torch.arange(prompt_len - 1, len(full_ids) - 1, device=logits.device)
    target_ids = input_ids[0, prompt_len:]
    selected = logits[0, positions].float().log_softmax(dim=-1).gather(-1, target_ids[:, None]).squeeze(-1)
    if selected.numel() != response_len:
        raise RuntimeError(f"score produced {selected.numel()} values, expected {response_len}")
    return selected.cpu().tolist()


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    args = _parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    command = _server_command(args)
    print(f"model: {model_path}")
    print("server command:", " ".join(command))
    if args.dry_run:
        return 0
    output_dir = _timestamp_dir(args.output_dir)
    print(f"output: {output_dir}")

    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=True)
    prompt_ids = tokenizer(args.prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0].tolist()
    base_url = args.base_url or f"http://127.0.0.1:{args.port}"
    process: subprocess.Popen[str] | None = None
    server_log = (output_dir / "sglang_server.log").open("w", encoding="utf-8")
    try:
        if process is None and args.base_url is None:
            _assert_port_available(args.port)
            print("starting SGLang+KT rollout server", flush=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(args.repo_root / "sglang" / "python") + os.pathsep + env.get("PYTHONPATH", "")
            process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, text=True, env=env)
            _wait_for_server(base_url, process, args.server_start_timeout)
            print("SGLang+KT server is healthy; running rollout forward", flush=True)
        elif args.base_url:
            print(f"using existing SGLang server at {base_url}", flush=True)
            requests.get(base_url + "/health", timeout=10).raise_for_status()

        rollout = _run_rollout(base_url, prompt_ids, args)
        print(f"rollout forward complete: {len(rollout['token_ids'])} response tokens", flush=True)
        (output_dir / "rollout.json").write_text(
            json.dumps(rollout["response"], indent=2) + "\n", encoding="utf-8"
        )
        # Release the rollout runtime before loading a second 30B runtime on
        # the same GPU.  An explicitly supplied external server is not owned
        # by this process and is therefore left running.
        if process is not None:
            _stop_server(process)
            print("rollout server stopped; starting HG+KT score forward", flush=True)
        full_ids = prompt_ids + rollout["token_ids"]
        score_logprobs = _load_hg_kt_score(str(model_path), full_ids, len(prompt_ids), args.repo_root)
        print(f"score forward complete: {len(score_logprobs)} response tokens", flush=True)
        comparison = compare_logprobs(
            rollout["token_ids"], rollout["logprobs"], score_logprobs,
            abs_tol=args.abs_tol, ratio_tol=args.ratio_tol,
        )
        report = {
            "model": str(model_path),
            "prompt": args.prompt,
            "prompt_token_ids": prompt_ids,
            "response_token_ids": comparison.token_ids,
            "active_response_mask": comparison.active_mask,
            "rollout_forward_old_logprobs": comparison.rollout_logprobs,
            "score_forward_new_logprobs": comparison.score_logprobs,
            "score_minus_rollout": comparison.diffs,
            "exp_score_minus_rollout": comparison.ratios,
            "max_abs_logprob_diff": comparison.max_abs_logprob_diff,
            "mean_abs_logprob_diff": comparison.mean_abs_logprob_diff,
            "max_abs_ratio_delta": comparison.max_abs_ratio_delta,
            "abs_tol": args.abs_tol,
            "ratio_tol": args.ratio_tol,
            "passed": comparison.passed,
        }
        (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("max_abs_logprob_diff", "mean_abs_logprob_diff", "max_abs_ratio_delta", "passed")}, indent=2))
        return 0 if comparison.passed else 1
    finally:
        if process is not None:
            _stop_server(process)
        server_log.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
