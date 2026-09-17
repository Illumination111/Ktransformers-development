#!/usr/bin/env python3
"""Shared correctness and output-format reward for math GRPO and evaluation."""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from typing import Any

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


ANSWER_INSTRUCTION = (
    'Solve the problem carefully. Put the final answer on the last non-empty line as "Answer: <answer>".'
)

_ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*:\s*(.+?)\s*$", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_ANSWER_MARKER_RE = re.compile(
    r"(?:final\s+answers?|answer\s+choice|correct\s+(?:answer|choice)|answers?)"
    r"(?:\s*\([^\n)]*\))?(?:\s+is)?\s*(?:[:：]|(?=[*#✅\s]*$))",
    re.IGNORECASE,
)
_LATEX_CHOICE_RE = re.compile(
    r"\\(?:text|textbf|mathrm)\s*\{\s*\(?\s*([A-E])\s*\)?(?:\s*[:.)-])?[^}]*\}",
)
_PLAIN_CHOICE_RE = re.compile(
    r"^\s*(?:choice|option)?\s*\(?\s*([A-E])\s*\)?"
    r"(?:\s*(?:[.):=-]|\bis\b)|\s+(?=[$\\\d({[])|\s*$)",
)
_ENUMERATED_ANSWER_RE = re.compile(r"^\s*([A-Za-z]|\d+)\s*[.)]\s*(.+?)\s*$")
_OPTION_LABEL_RE = re.compile(
    r"(?:\\(?:text|textbf|mathrm)\s*\{\s*)?\(\s*([A-E])\s*\)(?:\s*\})?",
    re.IGNORECASE,
)
_LOGGER = logging.getLogger(__name__)
_PARSER_ERROR_LOG_LIMIT = 20
_parser_error_logs = 0
CORRECTNESS_REWARD_WEIGHT = 1.0


def _reward_weight_from_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return value


FORMAT_REWARD_WEIGHT = _reward_weight_from_env("MATH_FORMAT_REWARD_WEIGHT", 0.1)


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


def _boxed_contents(text: str) -> list[str]:
    """Extract balanced boxed/fbox bodies instead of parsing the whole response."""
    contents: list[str] = []
    for match in _BOXED_RE.finditer(text):
        depth = 1
        index = match.end()
        start = index
        while index < len(text) and depth:
            if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
            index += 1
        if depth == 0:
            contents.append(text[start : index - 1].strip())
    return contents


def _append_unique(items: list[str], value: str) -> None:
    value = value.strip()
    if value and value not in items:
        items.append(value)


def _answer_blocks(solution: str) -> list[str]:
    """Return explicit answer-labelled blocks, with the final block first."""
    lines = solution.splitlines()
    markers = [index for index, line in enumerate(lines) if _ANSWER_MARKER_RE.search(line)]
    blocks = []
    for position, start in enumerate(markers):
        end = markers[position + 1] if position + 1 < len(markers) else len(lines)
        blocks.append("\n".join(lines[start:end]))
    return list(reversed(blocks))


def _strip_math_wrappers(text: str) -> str:
    value = text.strip().strip("`*")
    while len(value) >= 2 and value[0] == "$" and value[-1] == "$":
        value = value[1:-1].strip()
    return value.strip().strip("`*")


def _strip_choice_prefix(text: str) -> str:
    """Remove a leading option label while preserving its mathematical value."""
    value = _strip_math_wrappers(text)
    value = re.sub(
        r"^\\(?:text|textbf|mathrm)\s*\{\s*\(?\s*[A-E]\s*\)?(?:\s*[:.)-])?[^}]*\}\s*",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^\s*\(?\s*[A-E]\s*\)\s*[:.)-]?\s*", "", value, count=1, flags=re.IGNORECASE)
    value = re.sub(r"^\s*[A-E]\s*[.):-]\s*", "", value, count=1)
    value = re.sub(r"^\s*[A-E]\s+(?=[$\\\d({[])", "", value, count=1)
    value = re.sub(r"^(?:\\[ ,;!]\s*)+", "", value)
    return value.strip()


def _candidate_variants(text: str) -> list[str]:
    """Produce final-answer candidates from one answer block or response tail."""
    candidates: list[str] = []
    boxes = _boxed_contents(text)
    if len(boxes) > 1:
        _append_unique(candidates, f"\\boxed{{{','.join(boxes)}}}")
    for body in reversed(boxes):
        _append_unique(candidates, f"\\boxed{{{body}}}")
        _append_unique(candidates, body)
        stripped_choice = _strip_choice_prefix(body)
        if stripped_choice != body:
            _append_unique(candidates, f"\\boxed{{{stripped_choice}}}")
        _append_unique(candidates, stripped_choice)
        if "=" in body:
            rhs = body.rsplit("=", 1)[1]
            rhs = re.split(r"\\quad\s*\\text\s*\{\s*(?:for|where|with)\b", rhs, maxsplit=1, flags=re.IGNORECASE)[0]
            _append_unique(candidates, f"\\boxed{{{rhs.strip()}}}")
    for value in _simple_equality_values(text):
        _append_unique(candidates, value)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        marker = _ANSWER_MARKER_RE.search(line)
        if marker:
            answer_value = line[marker.end() :]
            _append_unique(candidates, answer_value)
            if answer_value.count("=") == 1:
                _append_unique(candidates, answer_value.rsplit("=", 1)[1])
            _append_unique(candidates, line)
        elif not boxes:
            _append_unique(candidates, line)
    return candidates


def _prediction_candidates(solution: str) -> tuple[list[str], float]:
    last_line = _last_nonempty_line(solution)
    strict_match = _ANSWER_LINE_RE.match(last_line)
    format_ok = float(strict_match is not None)
    candidates: list[str] = []

    blocks = _answer_blocks(solution)
    if blocks:
        # A later explicit answer supersedes earlier tentative answers. This
        # prevents reward hacking such as ending with a wrong answer after a
        # correct intermediate "Answer:" line.
        for candidate in _candidate_variants(blocks[0]):
            _append_unique(candidates, candidate)
        # Some multiple-choice datasets store the option value (for example
        # 20/19) rather than the letter as ground truth.  Model responses often
        # present that value in a "Final Answer" block and then finish with
        # "Answer: C".  Retain the preceding block only for this letter-only
        # ending.  Choice ground truths still honor the first/final letter, and
        # an earlier value cannot rescue a later wrong numeric answer.
        final_choices = [
            choice for candidate in candidates if (choice := _prediction_choice(candidate))
        ]
        if final_choices and len(blocks) > 1:
            for candidate in _candidate_variants(blocks[1]):
                _append_unique(candidates, candidate)
    elif _BOXED_RE.search(solution):
        for candidate in _candidate_variants(solution):
            _append_unique(candidates, candidate)
    else:
        _append_unique(candidates, last_line)

    if strict_match:
        # Preserve the anchored form expected by math_verify, but also retain
        # the right-hand side for choice/list normalization.
        candidates.insert(0, last_line)
        _append_unique(candidates, strict_match.group(1))
    return candidates, format_ok


def _ground_truth_choice(ground_truth: str) -> str | None:
    value = _strip_math_wrappers(ground_truth)
    latex = re.fullmatch(
        r"\\(?:text|textbf|mathrm)\s*\{\s*\(?\s*([A-E])\s*\)?\s*\}", value
    )
    if latex:
        return latex.group(1).upper()
    plain = re.fullmatch(r"\(?\s*([A-E])\s*\)?", value)
    return plain.group(1) if plain else None


def _prediction_choice(candidate: str) -> str | None:
    value = _strip_math_wrappers(candidate)
    latex = _LATEX_CHOICE_RE.search(value)
    if latex:
        return latex.group(1)
    value = re.sub(r"^[#>*✅❌\s]+", "", value).strip("`*")
    plain = _PLAIN_CHOICE_RE.match(value)
    return plain.group(1) if plain else None


def _simple_equality_values(text: str) -> list[str]:
    """Extract simple numeric RHS values from final aligned answer blocks."""
    return re.findall(r"&?=\s*(-?\d+(?:\.\d+)?)", text)


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth = max(0, depth - 1)
        elif char in ",;" and depth == 0 and (index == 0 or text[index - 1] != "\\"):
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _boxed_answer_parts(text: str) -> list[str]:
    """Extract distinct top-level values from final boxed answer bodies."""
    parts: list[str] = []
    normalized: set[str] = set()
    for body in _boxed_contents(text):
        comma_parts = _split_top_level_commas(body)
        for comma_part in comma_parts:
            and_parts = re.split(
                r"\s+(?:\\text\s*\{\s*)?and(?:\s*\})?\s+",
                comma_part,
                flags=re.IGNORECASE,
            )
            for part in and_parts:
                value = re.sub(r"^(?:\\(?:,|;|quad|qquad|\s)\s*)+", "", part).strip()
                key = re.sub(r"(?:\\[,;]|\s)+", "", value)
                if value and key not in normalized:
                    normalized.add(key)
                    parts.append(value)
    return parts


def _answer_part_value(text: str) -> str:
    """Return the value side of a labelled equality when one is present."""
    value = re.sub(r"^(?:\\(?:,|;|quad|qquad|\s)\s*)+", "", text).strip()
    value = re.sub(r"\\(?:[,;!]|quad|qquad|\s)\s*", " ", value).strip()
    return value.rsplit("=", 1)[1].strip() if "=" in value else value


def _canonical_answer_part(text: str) -> str:
    """Canonical text fallback for tuples and notation math_verify cannot parse."""
    value = _strip_math_wrappers(_answer_part_value(text))
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\operatorname{arctg}", r"\arctan")
    value = value.replace(r"\tan^{-1}", r"\arctan")
    value = re.sub(r"\\arctan\s*\(([^()]*)\)", r"\\arctan\1", value)
    return re.sub(r"\s+", "", value)


def _canonical_text(text: str) -> str:
    """Conservative exact-text fallback for answers unsupported by math_verify."""
    value = _strip_math_wrappers(text)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace("∅", r"\varnothing").replace(r"\emptyset", r"\varnothing")
    value = re.sub(r"\\(?:text|textbf|mathrm)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\(?:quad|qquad|,|;|!| )", "", value)
    value = value.replace("$", "")
    value = re.sub(r"[{}\s]", "", value)
    return value.strip(".,;:")


def _canonical_text_equal(left: str, right: str) -> bool:
    left_value = _canonical_text(left)
    right_value = _canonical_text(right)
    if left_value == right_value:
        return True
    # Natural-language answers are case-insensitive, but one-letter algebraic
    # symbols are not interchangeable with upper-case choice labels.
    return min(len(left_value), len(right_value)) > 1 and left_value.casefold() == right_value.casefold()


def _prompt_text(extra_info: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    prompt = (extra_info or {}).get("verifier_prompt", kwargs.get("prompt"))
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(
            str(message.get("content", ""))
            for message in prompt
            if isinstance(message, dict) and message.get("role") == "user"
        )
    return ""


def _choice_options(prompt_text: str) -> dict[str, str]:
    """Extract labelled option values from common plain/LaTeX question forms."""
    matches = list(_OPTION_LABEL_RE.finditer(prompt_text))
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        if label in options:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt_text)
        value = prompt_text[match.end() : end].strip()
        value = re.sub(r"(?:\\qquad|\\quad|\$)+\s*$", "", value).strip()
        if value:
            options[label] = value
    return options


def _answers_equivalent(gold_text: str, prediction_text: str) -> bool:
    if _canonical_text_equal(gold_text, prediction_text):
        return True
    gold = _parse_gold(gold_text)
    prediction = _parse_prediction(prediction_text, False)
    return bool(gold and prediction and verify(gold, prediction, timeout_seconds=None))


def _verify_answer_parts(truth_parts: list[str], prediction_parts: list[str]) -> bool:
    """Compare compound answer values without depending on presentation order."""
    unmatched = [_answer_part_value(part) for part in prediction_parts]
    for truth_part in truth_parts:
        gold = _parse_gold(_answer_part_value(truth_part))
        match_index = None
        for index, prediction_part in enumerate(unmatched):
            if _canonical_answer_part(truth_part) == _canonical_answer_part(prediction_part):
                match_index = index
                break
            prediction = _parse_prediction(prediction_part, False)
            if gold and prediction and verify(gold, prediction, timeout_seconds=None):
                match_index = index
                break
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def _enumerated_answers(solution: str) -> list[str]:
    answers: list[tuple[str, str]] = []
    for line in solution.splitlines():
        match = _ENUMERATED_ANSWER_RE.match(line)
        if match:
            answers.append((match.group(1), _strip_math_wrappers(match.group(2))))
    if len(answers) < 2:
        return []

    labels = [label.lower() for label, _ in answers]
    if all(label.isalpha() for label in labels):
        expected = [chr(ord(labels[0]) + offset) for offset in range(len(labels))]
    elif all(label.isdigit() for label in labels):
        expected = [str(int(labels[0]) + offset) for offset in range(len(labels))]
    else:
        return []
    return [answer for _, answer in answers] if labels == expected else []


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
    del data_source
    solution = str(solution_str)
    truth = str(ground_truth)
    candidates, format_ok = _prediction_candidates(solution)
    explicit_answer = bool(_answer_blocks(solution) or _BOXED_RE.search(solution))
    prompt_text = _prompt_text(extra_info, kwargs)
    options = _choice_options(prompt_text) if prompt_text else {}
    parser_error = 0.0
    extracted = float(explicit_answer and bool(candidates))
    correct = 0.0
    try:
        gold_choice = _ground_truth_choice(truth)
        if gold_choice is not None:
            prediction_choices = [choice for candidate in candidates if (choice := _prediction_choice(candidate))]
            if prediction_choices:
                extracted = 1.0
                # Candidates are ordered from the final explicit answer
                # backwards. Honor the first unambiguous option found.
                correct = float(prediction_choices[0] == gold_choice)
            elif gold_choice in options:
                correct = float(any(_answers_equivalent(options[gold_choice], candidate) for candidate in candidates))
        else:
            prediction_choices = [choice for candidate in candidates if (choice := _prediction_choice(candidate))]
            mapped_final_choice = False
            if prediction_choices and prediction_choices[0] in options:
                extracted = 1.0
                mapped_final_choice = True
                correct = float(_answers_equivalent(truth, options[prediction_choices[0]]))
            truth_parts = _split_top_level_commas(truth)
            answer_blocks = _answer_blocks(solution)
            enumeration_source = answer_blocks[0] if answer_blocks else solution
            enumerated = _enumerated_answers(enumeration_source) if len(truth_parts) > 1 else []
            equality_values = _simple_equality_values(enumeration_source)
            boxed_parts = _boxed_answer_parts(enumeration_source) if len(truth_parts) > 1 else []
            prediction_parts: list[str] = []
            if len(truth_parts) > 1:
                if len(enumerated) == len(truth_parts):
                    prediction_parts = enumerated
                elif len(boxed_parts) == len(truth_parts):
                    prediction_parts = boxed_parts
                elif len(equality_values) == len(truth_parts):
                    prediction_parts = equality_values
            if not mapped_final_choice and not correct and prediction_parts:
                extracted = 1.0
                correct = float(_verify_answer_parts(truth_parts, prediction_parts))
            elif not mapped_final_choice and not correct:
                gold = _parse_gold(truth)
                for candidate in candidates:
                    if _canonical_text_equal(truth, candidate):
                        extracted = 1.0
                        correct = 1.0
                        break
                    candidate_is_answer_line = bool(_ANSWER_LINE_RE.match(candidate))
                    prediction = _parse_prediction(candidate, candidate_is_answer_line) if candidate else []
                    extracted = max(extracted, float(bool(prediction)))
                    if gold and prediction and verify(gold, prediction, timeout_seconds=None):
                        correct = 1.0
                        break
    except Exception as exc:
        parser_error = 1.0
        _log_parser_failure(exc, truth, candidates[0] if candidates else "")

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
