from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from math_verify_reward import compute_score, ensure_answer_instruction  # noqa: E402
from prepare_hard_math_data import normalized_problem, source_problem  # noqa: E402


class MathVerifyRewardTest(unittest.TestCase):
    def score(self, prediction: str, ground_truth: str) -> dict[str, float]:
        return compute_score("math_dapo", prediction, ground_truth)

    def test_numeric_answer_line(self) -> None:
        result = self.score("Reasoning.\nAnswer: 1440", "1440")
        self.assertEqual(result["score"], 1.1)
        self.assertEqual(result["correctness_reward"], 1.0)
        self.assertEqual(result["format_reward"], 0.1)
        self.assertEqual(result["format_ok"], 1.0)

    def test_equivalent_fraction(self) -> None:
        self.assertEqual(self.score(r"Answer: $0.5$", r"\frac{1}{2}")["score"], 1.1)

    def test_equivalent_interval(self) -> None:
        result = self.score(r"Therefore $\boxed{[-2,7]}$", r"x \in [-2,7]")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["format_ok"], 0.0)

    def test_equivalent_algebra(self) -> None:
        prediction = r"Answer: $\frac{9a+11}{20}$"
        self.assertEqual(self.score(prediction, r"\frac{11+9a}{20}")["score"], 1.1)

    def test_choice_answer(self) -> None:
        self.assertEqual(self.score(r"Answer: $\boxed{C}$", r"\text{(C)}")["score"], 1.1)

    def test_wrong_answer(self) -> None:
        result = self.score("Answer: 7", "8")
        self.assertEqual(result["score"], 0.1)
        self.assertEqual(result["acc"], 0.0)
        self.assertEqual(result["format_reward"], 0.1)

    def test_unformatted_last_line_is_scored_but_logged(self) -> None:
        result = self.score("The calculation is complete.\n$8$", "8")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["format_ok"], 0.0)
        self.assertEqual(result["answer_extracted"], 1.0)

    def test_threaded_parse_does_not_hit_signal_alarm(self) -> None:
        result: dict[str, float] = {}
        error: list[BaseException] = []

        def run() -> None:
            try:
                result.update(self.score("Answer: 1", "1"))
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive())
        self.assertEqual(error, [])
        self.assertEqual(result["score"], 1.1)
        self.assertEqual(result["parser_error"], 0.0)

    def test_parser_exception_sets_parser_error_and_logs(self) -> None:
        from unittest.mock import patch

        with patch("math_verify_reward._parse_gold", side_effect=RuntimeError("boom")):
            with self.assertLogs("math_verify_reward", level="ERROR") as captured:
                result = self.score("Answer: 1", "1")
        self.assertEqual(result["score"], 0.1)
        self.assertEqual(result["parser_error"], 1.0)
        self.assertTrue(any("boom" in line for line in captured.output))

    def test_instruction_is_idempotent(self) -> None:
        prompt = [{"role": "user", "content": "What is 1+1?"}]
        once = ensure_answer_instruction(prompt)
        twice = ensure_answer_instruction(once)
        self.assertEqual(once, twice)
        self.assertEqual(prompt[0]["content"], "What is 1+1?")

    def test_dapo_wrapper_is_removed_for_overlap_checks(self) -> None:
        wrapped = (
            "Solve the following math problem step by step. The last line of your response should be of the form "
            "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nWhat is 1 + 1?\n\n"
            'Remember to put your answer on its own line after "Answer:".'
        )
        self.assertEqual(source_problem(wrapped), "What is 1 + 1?")
        self.assertEqual(normalized_problem(wrapped), "what is 1 + 1?")


if __name__ == "__main__":
    unittest.main()
