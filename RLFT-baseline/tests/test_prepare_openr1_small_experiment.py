from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_openr1_small_experiment import (  # noqa: E402
    COMPACT_REASONING_INSTRUCTION,
    STRICT_COMPACT_REASONING_INSTRUCTION,
    compact_teacher_solution,
    ensure_compact_instruction,
    ensure_strict_compact_instruction,
    jaccard_similarity,
    problem_text,
    prompt_shingles,
    round_robin,
    round_robin_unique,
    teacher_quality_issue,
)


class PrepareOpenR1SmallExperimentTest(unittest.TestCase):
    def test_compaction_removes_duplicate_paragraphs_and_markup(self) -> None:
        source = (
            "<think>Use the [identity](https://example.test/id).</think>\n\n"
            "---\n\nUse the identity.\n\nUse the identity.\n\nAnswer: 2"
        )
        compact, removed = compact_teacher_solution(source)
        self.assertNotIn("https://", compact)
        self.assertNotIn("<think>", compact)
        self.assertNotIn("---", compact)
        self.assertEqual(compact.count("Use the identity."), 1)
        self.assertEqual(removed, 3)
        self.assertTrue(compact.endswith("Answer: 2"))

    def test_compact_instruction_is_idempotent(self) -> None:
        prompt = [{"role": "user", "content": "What is 1+1?"}]
        once = ensure_compact_instruction(prompt)
        twice = ensure_compact_instruction(once)
        self.assertEqual(once, twice)
        self.assertIn(COMPACT_REASONING_INSTRUCTION, once[0]["content"])
        self.assertEqual(prompt[0]["content"], "What is 1+1?")

    def test_strict_compact_instruction_is_idempotent(self) -> None:
        prompt = [{"role": "user", "content": "What is 1+1?"}]
        once = ensure_strict_compact_instruction(prompt)
        twice = ensure_strict_compact_instruction(once)
        self.assertEqual(once, twice)
        self.assertIn(STRICT_COMPACT_REASONING_INSTRUCTION, once[0]["content"])
        self.assertEqual(problem_text(once), "what is 1+1?")

    def test_teacher_quality_rejects_deliberation_and_multiple_answers(self) -> None:
        self.assertEqual(
            teacher_quality_issue("Wait, try another way.\n\nAnswer: 2"),
            "teacher_deliberation_marker",
        )
        self.assertEqual(
            teacher_quality_issue("Answer: 1\nThe corrected result is 2.\nAnswer: 2"),
            "teacher_multiple_answer_lines",
        )
        self.assertIsNone(teacher_quality_issue("Since $1+1=2$.\n\nAnswer: 2"))

    def test_prompt_shingle_similarity_ignores_contract(self) -> None:
        left = ensure_strict_compact_instruction(
            [{"role": "user", "content": "Find the value of x if x + 1 = 3."}]
        )
        right = [{"role": "user", "content": "Find the value of x if x + 1 = 3."}]
        self.assertEqual(jaccard_similarity(prompt_shingles(left), prompt_shingles(right)), 1.0)

    def test_round_robin_unique_discards_near_duplicate_prompt(self) -> None:
        rows = [
            {
                "prompt": [{"role": "user", "content": "Find x when x plus one equals three."}],
                "extra_info": {"sample_hash": "a"},
            },
            {
                "messages": [
                    {"role": "user", "content": "Find x when x plus one equals three."},
                    {"role": "assistant", "content": "Answer: 2"},
                ],
                "extra_info": {"sample_hash": "b"},
            },
            {
                "prompt": [{"role": "user", "content": "Compute the area of a unit square."}],
                "extra_info": {"sample_hash": "c"},
            },
        ]
        selected, skipped = round_robin_unique(rows, 3, 42, 0.9)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(skipped), 1)
        self.assertTrue(skipped <= {"a", "b"})

    def test_process_balanced_round_robin_preserves_hard_bucket(self) -> None:
        rows = []
        for index in range(8):
            rows.append(
                {
                    "prompt": [{"role": "user", "content": f"easy problem {index}"}],
                    "extra_info": {
                        "sample_hash": f"easy-{index}",
                        "raw_source": "source",
                        "problem_type": "Algebra",
                        "question_type": "word",
                        "process_difficulty": "consensus",
                    },
                }
            )
        rows.append(
            {
                "prompt": [{"role": "user", "content": "hard problem"}],
                "extra_info": {
                    "sample_hash": "hard",
                    "raw_source": "source",
                    "problem_type": "Algebra",
                    "question_type": "word",
                    "process_difficulty": "hard_verified",
                },
            }
        )
        selected = round_robin(
            rows,
            2,
            42,
            ("raw_source", "problem_type", "question_type", "process_difficulty"),
        )
        self.assertEqual(
            {row["extra_info"]["process_difficulty"] for row in selected},
            {"consensus", "hard_verified"},
        )

    def test_primary_problem_balance_covers_each_available_type(self) -> None:
        rows = []
        for problem_type in ("Algebra", "Calculus", "Geometry"):
            for source_index in range(5):
                rows.append(
                    {
                        "prompt": [
                            {
                                "role": "user",
                                "content": f"{problem_type} problem {source_index}",
                            }
                        ],
                        "extra_info": {
                            "sample_hash": f"{problem_type}-{source_index}",
                            "raw_source": f"source-{source_index}",
                            "problem_type": problem_type,
                            "question_type": "word",
                            "process_difficulty": "consensus",
                        },
                    }
                )
        selected = round_robin(
            rows,
            3,
            42,
            ("raw_source", "problem_type", "question_type", "process_difficulty"),
            "problem_type",
        )
        self.assertEqual(
            {row["extra_info"]["problem_type"] for row in selected},
            {"Algebra", "Calculus", "Geometry"},
        )


if __name__ == "__main__":
    unittest.main()
