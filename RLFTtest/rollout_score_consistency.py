#!/usr/bin/env python3
"""Compare rollout token logprobs with fixed-response score logprobs.

The rollout path generates a response and records its token ids and
``output_token_logprobs``. The score paths then receive the exact same
prompt+response ids and do teacher-forced scoring; they never resample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from urllib.request import Request, urlopen

from compare_probability import (
    DEFAULT_MODELS,
    create_experiment_dir,
    hf_kt_logprobs,
    hf_plain_logprobs,
    launch_sglang,
    require_path,
    stop_sglang,
    tokenize_prompts,
    validate_runtime_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=sorted(DEFAULT_MODELS), default="30b")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--kt-weight-path", type=Path, required=True)
    parser.add_argument("--kt-method", default="BF16")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30107)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--kt-threadpool-count", type=int, default=1)
    parser.add_argument("--kt-num-threads", type=int, default=48)
    parser.add_argument("--kt-model-max-length", type=int, default=512)
    parser.add_argument("--kt-cpuinfer", type=int)
    parser.add_argument("--kt-num-gpu-experts", type=int, default=0)
    parser.add_argument("--kt-gpu-experts-ratio", type=float)
    parser.add_argument("--kt-src", type=Path, default=Path("/home/wubowen/ktransformers-RL/ktransformers"))
    parser.add_argument("--sglang-src", type=Path)
    parser.add_argument("--verl-src", type=Path, default=Path("/home/wubowen/ktransformers-RL/verl"))
    parser.add_argument("--overlay", type=Path, default=Path("/home/wubowen/ktransformers-RL"))
    parser.add_argument("--kt-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--hf-device-map", choices=("auto", "balanced"), default="auto")
    parser.add_argument("--hf-max-memory-gib", type=int, default=76)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--replay-report", type=Path, help="Replay HF+KT score from a saved rollout report")
    parser.add_argument("--score-backend", choices=("kt", "plain", "both"), default="kt")
    return parser.parse_args()


def request_json(url: str, payload: dict[str, Any], timeout: float = 1800.0) -> dict[str, Any]:
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def as_logprob(item: Any) -> float:
    if isinstance(item, (list, tuple)):
        return float(item[0])
    return float(item)


def as_token_id(item: Any) -> int | None:
    if isinstance(item, (list, tuple)) and len(item) > 1 and item[1] is not None:
        return int(item[1])
    return None


def rollout(base_url: str, prompt_ids: list[int], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": True,
            "sampling_seed": args.seed,
        },
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": 0,
    }
    response = request_json(f"{base_url}/generate", payload, args.timeout)
    meta = response.get("meta_info", {})
    response_ids = [int(x) for x in response.get("output_ids", [])]
    raw_values = meta.get("output_token_logprobs", [])
    output_values = [as_logprob(x) for x in raw_values]
    output_ids_from_values = [as_token_id(x) for x in raw_values]
    if not response_ids:
        raise RuntimeError(f"SGLang response did not contain output_ids: {response.keys()}")
    if len(output_values) != len(response_ids):
        raise RuntimeError(
            f"rollout logprob/token length mismatch: ids={len(response_ids)} values={len(output_values)}"
        )
    if any(x is not None for x in output_ids_from_values):
        checked = [int(x) for x in output_ids_from_values if x is not None]
        if checked != response_ids:
            raise RuntimeError("SGLang output logprob token ids do not match output_ids")
    return {"response_ids": response_ids, "rollout_logprobs": output_values, "raw_meta": meta}


def sglang_score(base_url: str, full_ids: list[int], prompt_len: int, timeout: float) -> list[float]:
    response = request_json(
        f"{base_url}/generate",
        {
            "input_ids": full_ids,
            "sampling_params": {"temperature": 1.0, "max_new_tokens": 0, "ignore_eos": True},
            "return_logprob": True,
            "return_text_in_logprobs": False,
            # Request the complete sequence. This avoids version-dependent
            # nonzero logprob_start_len sentinel semantics; response token t
            # is then at the full-sequence index prompt_len - 1 + t.
            "logprob_start_len": 0,
        },
        timeout,
    )
    raw_values = response.get("meta_info", {}).get("input_token_logprobs", [])
    # SGLang versions differ on whether the unscored boundary token is
    # represented by a bare None or by a [None, ...] tuple.
    raw_values = [
        item
        for item in raw_values
        if item is not None and not (isinstance(item, (list, tuple)) and item and item[0] is None)
    ]
    values = [as_logprob(x) for x in raw_values]
    expected_full = len(full_ids) - 1
    if len(values) == len(full_ids):
        values = values[1:]
    if len(values) != expected_full:
        raise RuntimeError(f"score logprob/token length mismatch: expected_full={expected_full} got={len(values)}")
    return values[prompt_len - 1 :]


def summarize(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise RuntimeError(f"cannot compare unequal lengths: {len(reference)} vs {len(candidate)}")
    diffs = [candidate[i] - reference[i] for i in range(len(reference))]
    abs_diffs = [abs(x) for x in diffs]
    ratios = [__import__("math").exp(x) for x in diffs]
    return {
        "tokens": len(diffs),
        "mean_abs_diff": sum(abs_diffs) / len(diffs),
        "max_abs_diff": max(abs_diffs),
        "p95_abs_diff": sorted(abs_diffs)[max(0, int(len(abs_diffs) * 0.95) - 1)],
        "sum_diff": sum(diffs),
        "sequence_ratio": __import__("math").exp(sum(diffs)),
        "mean_ratio": sum(ratios) / len(ratios),
        "max_ratio_deviation": max(abs(x - 1.0) for x in ratios),
        "clip_fraction_eps_0.2": sum(abs(x - 1.0) > 0.2 for x in ratios) / len(ratios),
        "diffs": diffs,
    }


def main() -> int:
    args = parse_args()
    model_path = (args.model_path or Path(DEFAULT_MODELS[args.model_size])).expanduser()
    kt_path = args.kt_weight_path.expanduser()
    if args.sglang_src is None:
        args.sglang_src = args.kt_src / "third_party" / "sglang"
    # Reuse the existing launcher/report helpers with a dedicated experiment
    # name; this is not one of compare_probability.py's engine choices.
    args.engine = "rollout_score"
    args.output = None
    require_path(model_path, "model path")
    require_path(kt_path, "KT weight path")
    validate_runtime_paths(args)
    prompts = args.prompt or ["Explain why deterministic inference is useful for RL training."]
    if args.replay_report is not None:
        source = json.loads(args.replay_report.read_text(encoding="utf-8"))
        replay = dict(source)
        replay["status"] = "replay_running"
        replay["replay_source"] = str(args.replay_report)
        for case in replay.get("cases", []):
            full_ids = case["prompt_ids"] + case["response_ids"]
            if args.score_backend in ("kt", "both"):
                hf_all = hf_kt_logprobs(args, model_path, kt_path, full_ids)
                hf_scores = hf_all[len(case["prompt_ids"]) - 1 :]
                if len(hf_scores) != len(case["response_ids"]):
                    raise RuntimeError(f"HF score length mismatch: {len(hf_scores)}")
                case["hf_kt_score_logprobs"] = hf_scores
                case["rollout_vs_hf_kt_score"] = summarize(case["rollout_logprobs"], hf_scores)
                case["sglang_kt_score_vs_hf_kt_score"] = summarize(case["sglang_kt_score_logprobs"], hf_scores)
            if args.score_backend in ("plain", "both"):
                plain_all = hf_plain_logprobs(args, model_path, full_ids)
                plain_scores = plain_all[len(case["prompt_ids"]) - 1 :]
                if len(plain_scores) != len(case["response_ids"]):
                    raise RuntimeError(f"plain HF score length mismatch: {len(plain_scores)}")
                case["hf_plain_score_logprobs"] = plain_scores
                case["rollout_vs_hf_plain_score"] = summarize(case["rollout_logprobs"], plain_scores)
                if "hf_kt_score_logprobs" in case:
                    case["hf_kt_score_vs_hf_plain_score"] = summarize(case["hf_plain_score_logprobs"], case["hf_kt_score_logprobs"])
        replay["status"] = "completed"
        output = args.replay_report.with_name(f"hf_{args.score_backend}_replay_result.json")
        output.write_text(json.dumps(replay, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(replay, indent=2, ensure_ascii=False))
        print(f"report={output}")
        return 0
    prompt_batches = tokenize_prompts(model_path, prompts, args.max_prompt_tokens)
    experiment_dir = create_experiment_dir(args, model_path)
    result_path = experiment_dir / "rollout_score_result.json"
    report: dict[str, Any] = {
        "status": "running",
        "model_path": str(model_path),
        "kt_weight_path": str(kt_path),
        "prompts": prompts,
        "config": {key: str(value) for key, value in vars(args).items()},
        "cases": [],
    }
    process = None
    log_file = None
    try:
        process, url, log_file = launch_sglang(args, model_path, kt_path, experiment_dir / "sglang.log", use_kt=True)
        for index, prompt_ids in enumerate(prompt_batches):
            generated = rollout(url, prompt_ids, args)
            full_ids = prompt_ids + generated["response_ids"]
            sglang_scores = sglang_score(url, full_ids, len(prompt_ids), args.timeout)
            case: dict[str, Any] = {
                "prompt_index": index,
                "prompt_ids": prompt_ids,
                "response_ids": generated["response_ids"],
                "rollout_logprobs": generated["rollout_logprobs"],
                "sglang_kt_score_logprobs": sglang_scores,
                "rollout_vs_sglang_kt_score": summarize(generated["rollout_logprobs"], sglang_scores),
            }
            # Persist the fixed response before loading another backend. This
            # makes replay possible even when HF initialization later fails.
            report["cases"].append(case)
            report["status"] = "rollout_completed"
            result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if not args.skip_hf:
                hf_all = hf_kt_logprobs(args, model_path, kt_path, full_ids)
                hf_scores = hf_all[len(prompt_ids) - 1 :]
                if len(hf_scores) != len(generated["response_ids"]):
                    raise RuntimeError(f"HF score length mismatch: {len(hf_scores)}")
                case["hf_kt_score_logprobs"] = hf_scores
                case["rollout_vs_hf_kt_score"] = summarize(generated["rollout_logprobs"], hf_scores)
                case["sglang_kt_score_vs_hf_kt_score"] = summarize(sglang_scores, hf_scores)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_sglang(process, log_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
