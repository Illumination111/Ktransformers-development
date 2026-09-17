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
    def score(
        self, prediction: str, ground_truth: str, prompt: str | None = None
    ) -> dict[str, float]:
        extra_info = {"verifier_prompt": prompt} if prompt is not None else None
        return compute_score("math_dapo", prediction, ground_truth, extra_info=extra_info)

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

    def test_choice_with_text_and_option_content(self) -> None:
        prediction = r"Therefore $\boxed{\text{(D)}\ (1.1,-2.1,1.0)}$"
        self.assertEqual(self.score(prediction, "D")["correctness_reward"], 1.0)

    def test_markdown_final_choice_is_extracted(self) -> None:
        result = self.score("Reasoning.\n### ✅ Final Answer: **B**", "B")
        self.assertEqual(result["correctness_reward"], 1.0)
        self.assertEqual(result["answer_extracted"], 1.0)

    def test_wrong_final_choice_is_not_rescued_by_reasoning(self) -> None:
        prediction = "Option B looks plausible.\nAnswer: D"
        self.assertEqual(self.score(prediction, "B")["correctness_reward"], 0.0)

    def test_wrong_final_numeric_answer_supersedes_earlier_answer(self) -> None:
        prediction = "Answer: 8\nAfter reconsidering:\nAnswer: 7"
        self.assertEqual(self.score(prediction, "8")["correctness_reward"], 0.0)

    def test_final_choice_can_refer_to_preceding_numeric_option_value(self) -> None:
        prediction = (
            "Final Answer:\n"
            r"$\boxed{\frac{20}{19}}$"
            "\nSo the correct choice is C.\n"
            "Answer: **C**"
        )
        self.assertEqual(
            self.score(prediction, r"\frac{20}{19}")["correctness_reward"], 1.0
        )

    def test_lowercase_algebra_variable_is_not_a_choice(self) -> None:
        self.assertEqual(self.score(r"Answer: $a=b$", "b")["correctness_reward"], 1.0)

    def test_bare_choice_prefix_is_removed_from_algebra_answer(self) -> None:
        self.assertEqual(self.score(r"Final Answer: $\boxed{D.\ a_{11}}$", r"a_{11}")["correctness_reward"], 1.0)

    def test_numeric_answer_with_trailing_choice_label(self) -> None:
        prediction = "Reasoning.\n" r"Answer: $\boxed{\textbf{(D)}\ 49}$"
        self.assertEqual(self.score(prediction, "49")["score"], 1.1)

    def test_choice_letter_followed_by_numeric_option_value(self) -> None:
        prediction = r"### Answer: **C $48\mathrm{mph}$**"
        self.assertEqual(
            self.score(prediction, r"48\mathrm{}")["correctness_reward"], 1.0
        )

    def test_choice_ground_truth_accepts_matching_option_value(self) -> None:
        prompt = "How many?\n(A) 45\n(B) 50\n(C) 60\n(D) 70"
        result = self.score("Reasoning.\nAnswer: 60", "C", prompt)
        self.assertEqual(result["correctness_reward"], 1.0)
        self.assertEqual(result["answer_extracted"], 1.0)

    def test_value_ground_truth_accepts_matching_choice_letter(self) -> None:
        prompt = r"Least value? $\textbf{(A)}\ 1 \qquad\textbf{(B)}\ 2 \qquad\textbf{(C)}\ \sqrt{2}$"
        self.assertEqual(
            self.score(r"Answer: C", r"\sqrt{2}", prompt)["correctness_reward"],
            1.0,
        )

    def test_wrong_option_value_is_not_accepted(self) -> None:
        prompt = "How many?\n(A) 45\n(B) 50\n(C) 60\n(D) 70"
        self.assertEqual(self.score("Answer: 70", "C", prompt)["correctness_reward"], 0.0)

    def test_wrong_final_mapped_choice_is_not_rescued_by_preceding_value(self) -> None:
        prompt = "How many?\n(A) 45\n(B) 50\n(C) 60\n(D) 70"
        prediction = "Final Answer: $\\boxed{60}$\nAnswer: D"
        self.assertEqual(self.score(prediction, "60", prompt)["correctness_reward"], 0.0)

    def test_exact_symbol_and_word_answers(self) -> None:
        self.assertEqual(self.score("Answer: N", "N")["correctness_reward"], 1.0)
        self.assertEqual(self.score("Answer: blue", "blue")["correctness_reward"], 1.0)
        self.assertEqual(self.score("Answer: C", "c")["correctness_reward"], 0.0)

    def test_plain_assignment_rhs_matches_value_truth(self) -> None:
        self.assertEqual(self.score("Answer: y = x", "x")["correctness_reward"], 1.0)

    def test_function_list_text_fallback(self) -> None:
        prediction = r"Answer: f(x) = x \text{ or } f(x) = -x"
        truth = r"f(x)=x\quad\text{or}\quadf(x)=-x"
        self.assertEqual(self.score(prediction, truth)["correctness_reward"], 1.0)

    def test_multiple_boxed_forms_use_final_answer_block(self) -> None:
        prediction = "The result is $\\boxed{49}$.\n" r"**Answer:** $\boxed{\textbf{(D)}\ 49}$"
        self.assertEqual(self.score(prediction, "49")["correctness_reward"], 1.0)

    def test_enumerated_multi_answer(self) -> None:
        prediction = "a) $27$\nb) $8$\nc) $12$\nd) $6$\ne) $1$"
        result = self.score(prediction, "27,8,12,6,1")
        self.assertEqual(result["correctness_reward"], 1.0)
        self.assertEqual(result["answer_extracted"], 1.0)

    def test_multiple_boxed_parts_form_one_set_answer(self) -> None:
        prediction = r"Final Answer: $\boxed{3,7,8}$ or $\boxed{4,5,6}$"
        self.assertEqual(self.score(prediction, "3,7,8or4,5,6")["correctness_reward"], 1.0)

    def test_final_equation_can_match_ground_truth_rhs(self) -> None:
        prediction = (
            "Final Answer:\n"
            r"$\boxed{\lim_{n \to \infty} f_n(x)=\frac{1}{1-x} "
            r"\quad \text{for all } x\in[0,1)}$"
        )
        self.assertEqual(self.score(prediction, r"\frac{1}{1-x}")["correctness_reward"], 1.0)

    def test_aligned_numeric_answers(self) -> None:
        prediction = (
            "Final Answer:\n"
            "\\boxed{\\begin{aligned}a &= 13 \\\\ b &= 14 \\\\ c &= 15\\end{aligned}}"
        )
        self.assertEqual(self.score(prediction, "13;14;15")["correctness_reward"], 1.0)

    def test_compound_assignments_are_order_independent(self) -> None:
        prediction = "### Final Answer:\n" + r"$\boxed{b=11,\quad n=1989}$"
        self.assertEqual(self.score(prediction, "n=1989,b=11")["correctness_reward"], 1.0)

    def test_compound_tuples_are_compared_as_complete_values(self) -> None:
        prediction = "### Final Answer:\n" + r"$\boxed{(7,3,2),\quad(5,3,5),\quad(3,2,7)}$"
        self.assertEqual(
            self.score(prediction, "(7,3,2);(5,3,5);(3,2,7)")["correctness_reward"],
            1.0,
        )

    def test_all_compound_values_must_match(self) -> None:
        prediction = "### Final Answer:\n" + r"$\boxed{900\text{ m/min and }750\text{ m/min}}$"
        self.assertEqual(
            self.score(prediction, r"900\mathrm{},700\mathrm{}")["correctness_reward"],
            0.0,
        )

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
