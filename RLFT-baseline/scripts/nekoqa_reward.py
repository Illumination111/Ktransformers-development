"""Reward function for NekoQA style-following GRPO.

NekoQA is open-ended dialogue, so mathematical exact-match scoring is not
meaningful. The score combines character-level overlap with the reference and
the dataset's explicit cat-persona markers, and stays in [0, 1].
"""

from __future__ import annotations

import re


_STYLE_MARKERS = ("主人", "喵", "no desu", "的说")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def _char_f1(prediction: str, reference: str) -> float:
    pred = list(_normalize(prediction))
    ref = list(_normalize(reference))
    if not pred or not ref:
        return 0.0
    ref_counts: dict[str, int] = {}
    for char in ref:
        ref_counts[char] = ref_counts.get(char, 0) + 1
    overlap = 0
    for char in pred:
        count = ref_counts.get(char, 0)
        if count:
            overlap += 1
            ref_counts[char] = count - 1
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, extra_info, kwargs
    prediction = _normalize(solution_str)
    reference = _normalize(ground_truth)
    overlap = _char_f1(prediction, reference)
    style_hits = sum(marker in prediction for marker in _STYLE_MARKERS)
    style = style_hits / len(_STYLE_MARKERS)
    score = 0.7 * overlap + 0.3 * style
    return {"score": float(score), "reference_overlap": float(overlap), "style_reward": float(style)}
