from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).with_name("test_rollout_score_consistency.py")
_SPEC = importlib.util.spec_from_file_location("rollout_score_consistency", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_active_response_tokens_pass_when_equal():
    result = _MODULE.compare_logprobs([11, 12], [-1.0, -2.0], [-1.0, -2.0])
    assert result.passed
    assert result.max_abs_logprob_diff == 0.0
    assert result.max_abs_ratio_delta == 0.0


def test_prompt_positions_can_be_masked():
    result = _MODULE.compare_logprobs(
        [1, 2, 3], [-1.0, -2.0, -3.0], [-100.0, -2.0, -3.0], [False, True, True]
    )
    assert result.passed
    assert result.max_abs_logprob_diff == 0.0


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="length mismatch"):
        _MODULE.compare_logprobs([1], [-1.0], [-1.0, -2.0])
