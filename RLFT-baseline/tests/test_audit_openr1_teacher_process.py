from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_openr1_teacher_process import build_judge_messages, parse_judgment  # noqa: E402


class AuditOpenR1TeacherProcessTest(unittest.TestCase):
    def test_parse_pass(self) -> None:
        self.assertEqual(
            parse_judgment("VERDICT: PASS\nREASON: Every step is valid.", "stop"),
            ("PASS", "Every step is valid."),
        )

    def test_parse_rejects_ambiguous_or_truncated_output(self) -> None:
        verdict, _ = parse_judgment(
            "VERDICT: PASS\nREASON: valid\nVERDICT: FAIL\nREASON: invalid", "stop"
        )
        self.assertEqual(verdict, "PARSE_ERROR")
        self.assertEqual(parse_judgment("VERDICT: PASS\nREASON: valid", "length")[0], "PARSE_ERROR")

    def test_prompt_contains_all_audit_inputs(self) -> None:
        messages = build_judge_messages("Find x.", "2", "x=2.\n\nAnswer: 2")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Find x.", messages[1]["content"])
        self.assertIn("EXPECTED ANSWER:\n2", messages[1]["content"])
        self.assertIn("Answer: 2", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
