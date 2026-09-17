from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from clean_openr1_grpo_data import classify_row  # noqa: E402


def row(prompt: str, truth: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"style": "rule", "ground_truth": truth},
    }


class CleanOpenR1DataTest(unittest.TestCase):
    def classify(self, prompt: str, truth: str) -> tuple[str, list[str]]:
        return classify_row(
            row(prompt, truth),
            seen_prompts=set(),
            benchmark_prompts=set(),
        )

    def test_normal_math_is_kept(self) -> None:
        self.assertEqual(self.classify("What is $1+1$?", "2"), ("keep", []))

    def test_translation_instruction_is_excluded(self) -> None:
        disposition, reasons = self.classify(
            "Translate the above problem into English and preserve formatting.", "6"
        )
        self.assertEqual(disposition, "exclude")
        self.assertIn("translation_or_meta_task", reasons)

    def test_embedded_image_is_excluded(self) -> None:
        disposition, reasons = self.classify("Find x. ![](https://example/x.png)", "3")
        self.assertEqual(disposition, "exclude")
        self.assertIn("embedded_image_unavailable", reasons)

    def test_visual_reference_without_payload_is_reviewed(self) -> None:
        disposition, reasons = self.classify("As shown in the figure, find x.", "3")
        self.assertEqual(disposition, "review")
        self.assertIn("visual_reference_needs_context_check", reasons)

    def test_malformed_label_is_reviewed(self) -> None:
        disposition, reasons = self.classify("Differentiate $x^2$.", r"\frac{^2}{}")
        self.assertEqual(disposition, "review")
        self.assertIn("malformed_ground_truth", reasons)

    def test_interval_comma_is_not_a_compound_label(self) -> None:
        self.assertEqual(self.classify("Find the interval.", "[-2,7]"), ("keep", []))
        self.assertEqual(self.classify("Find the interval.", r"(-\infty,-1]"), ("keep", []))

    def test_top_level_answer_list_is_reviewed(self) -> None:
        disposition, reasons = self.classify("Find all values.", "1,4,9")
        self.assertEqual(disposition, "review")
        self.assertIn("compound_ground_truth", reasons)

    def test_multiple_questions_with_one_label_are_reviewed(self) -> None:
        disposition, reasons = self.classify("What is x? What is y?", "3")
        self.assertEqual(disposition, "review")
        self.assertIn("multiple_questions_single_label", reasons)

    def test_duplicate_prompt_is_excluded(self) -> None:
        seen = {"what is 1+1?"}
        disposition, reasons = classify_row(
            row("What   is 1+1?", "2"), seen_prompts=seen, benchmark_prompts=set()
        )
        self.assertEqual(disposition, "exclude")
        self.assertIn("duplicate_prompt", reasons)


if __name__ == "__main__":
    unittest.main()
