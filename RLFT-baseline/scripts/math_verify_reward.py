#!/usr/bin/env python3
"""Shared correctness and output-format reward for math GRPO and evaluation."""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


ANSWER_INSTRUCTION = (
    'Solve the problem carefully. Put the final answer on the last non-empty line as "Answer: <answer>".'
)

_ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*:\s*(.+?)\s*$", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_LOGGER = logging.getLogger(__name__)
_PARSER_ERROR_LOG_LIMIT = 20
_parser_error_logs = 0
CORRECTNESS_REWARD_WEIGHT = 1.0
FORMAT_REWARD_WEIGHT = 0.1


def ensure_answer_instruction(prompt: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return a copied chat prompt with an explicit final-answer contract."""
    messages = [{"role": str(item["role"]), "content": str(item["content"])} for item in prompt]
    if not messages:
        raise ValueError("prompt must contain at least one message")
    user_indexes = [index for index, item in enumerate(messages) if item["role"] == "user"]
    if not user_indexes:
        raise ValueError("prompt must contain a user message")
    index = user_indexes[-1]
    content = messages[index]["content"].strip()
    if "Answer: <answer>" not in content:
        messages[index]["content"] = f"{ANSWER_INSTRUCTION}\n\n{content}"
    return messages


def _last_nonempty_line(text: str) -> str:
    return next((line.strip() for line in reversed(str(text).splitlines()) if line.strip()), "")


def _prediction_candidate(solution: str) -> tuple[str, float]:
    last_line = _last_nonempty_line(solution)
    if _ANSWER_LINE_RE.match(last_line):
        return last_line, 1.0
    if _BOXED_RE.search(solution):
        return solution, 0.0
    return last_line, 0.0


def _parse_gold(ground_truth: str):
    # RewardLoopWorker scores on a worker thread. math_verify's default
    # signal.alarm timeout only works in the main thread, so disable it.
    wrapped = f"${ground_truth}$"
    return parse(
        wrapped,
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        fallback_mode="first_match",
        parsing_timeout=None,
    )


def _parse_prediction(candidate: str, format_ok: bool):
    extraction_config = [
        LatexExtractionConfig(try_extract_without_anchor=not format_ok, boxed_match_priority=0),
        ExprExtractionConfig(try_extract_without_anchor=not format_ok),
    ]
    return parse(
        candidate,
        extraction_config=extraction_config,
        fallback_mode="first_match",
        parsing_timeout=None,
    )


def _log_parser_failure(exc: BaseException, ground_truth: str, candidate: str) -> None:
    """Emit a bounded traceback so Ray workers cannot swallow verifier failures."""
    global _parser_error_logs
    if _parser_error_logs >= _PARSER_ERROR_LOG_LIMIT:
        return
    _parser_error_logs += 1
    gold_preview = ground_truth[:200]
    pred_preview = candidate[:200]
    message = (
        f"math_verify failed ({type(exc).__name__}: {exc}); "
        f"ground_truth={gold_preview!r} candidate={pred_preview!r}"
    )
    print(f"[math_verify_reward] {message}", file=sys.stderr, flush=True)
    _LOGGER.exception(message)
    if _parser_error_logs == _PARSER_ERROR_LOG_LIMIT:
        print(
            "[math_verify_reward] further parser failures will be silent",
            file=sys.stderr,
            flush=True,
        )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Return correctness plus a small additive final-answer format reward.

    Correctness remains the dominant signal. ``format_ok`` rewards the
    requested ``Answer: ...`` last non-empty line. Boxed answers remain
    eligible for correctness parsing but receive no format bonus.
    """
    del data_source, extra_info, kwargs
    candidate, format_ok = _prediction_candidate(str(solution_str))
    parser_error = 0.0
    extracted = 0.0
    correct = 0.0
    try:
        gold = _parse_gold(str(ground_truth))
        prediction = _parse_prediction(candidate, bool(format_ok)) if candidate else []
        extracted = float(bool(prediction))
        if gold and prediction:
            correct = float(bool(verify(gold, prediction, timeout_seconds=None)))
    except Exception as exc:
        parser_error = 1.0
        _log_parser_failure(exc, str(ground_truth), candidate)

    format_reward = FORMAT_REWARD_WEIGHT * format_ok
    correctness_reward = CORRECTNESS_REWARD_WEIGHT * correct
    return {
        "score": correctness_reward + format_reward,
        "acc": correct,
        "correctness_reward": correctness_reward,
        "format_reward": format_reward,
        "format_ok": format_ok,
        "answer_extracted": extracted,
        "parser_error": parser_error,
    }


__all__ = [
    "ANSWER_INSTRUCTION",
    "CORRECTNESS_REWARD_WEIGHT",
    "FORMAT_REWARD_WEIGHT",
    "compute_score",
    "ensure_answer_instruction",
]
