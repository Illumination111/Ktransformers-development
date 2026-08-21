from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compare_logprobs(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError(f"invalid lengths: reference={len(reference)}, candidate={len(candidate)}")
    diffs = [float(b - a) for a, b in zip(reference, candidate, strict=True)]
    absolute = [abs(value) for value in diffs]
    ordered = sorted(absolute)

    def percentile(q: float) -> float:
        return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]

    ratios = [math.exp(max(-80.0, min(80.0, value))) for value in diffs]
    outside = [value < 0.8 or value > 1.2 for value in ratios]
    return {
        "tokens": len(diffs),
        "mean_abs_diff": sum(absolute) / len(absolute),
        "max_abs_diff": max(absolute),
        "p95_abs_diff": percentile(0.95),
        "p99_abs_diff": percentile(0.99),
        "sum_diff": sum(diffs),
        "sequence_ratio": math.exp(max(-80.0, min(80.0, sum(diffs)))),
        "ratio_outside_0_8_1_2_fraction": sum(outside) / len(outside),
        "strict_max_abs_le_1e_3": max(absolute) <= 1e-3,
        "grpo_pause_threshold_exceeded": sum(outside) / len(outside) > 0.01,
        "diffs": diffs,
        "ratios": ratios,
    }
