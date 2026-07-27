from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from finetune_train_with_timing import _manifest_backend
from persistent_sweep import load_manifest
from verify_gpu_peak_hold import verify


SCRIPT_DIR = Path(__file__).resolve().parent


class PersistentSweepTest(unittest.TestCase):
    def test_summary_finalizer_is_registered_before_training(self) -> None:
        script = (
            SCRIPT_DIR / "run_finetune_perf_sweep_bf16_common.sh"
        ).read_text(encoding="utf-8")
        trap = "trap 'finalize_sweep_on_exit $?' EXIT"
        self.assertIn(trap, script)
        self.assertLess(script.index(trap), script.index("run_one_sequence()"))
        self.assertIn(
            '"${VALIDATOR_PYTHON}" "${AGGREGATOR}" --root "${RUN_ROOT}"',
            script,
        )

    def test_ktransformers_runtime_alias_matches_manifest_backend(self) -> None:
        self.assertEqual(_manifest_backend("kt"), "ktransformers")
        self.assertEqual(_manifest_backend("deepspeed"), "deepspeed")

    def _manifest(self, root: Path, sequences: list[int]) -> Path:
        cases = [
            {
                "sequence_length": sequence,
                "run_dir": str(root / f"seq_{sequence}"),
                "timing_output_dir": str(
                    root / f"seq_{sequence}" / "step_timing"
                ),
            }
            for sequence in sequences
        ]
        path = root / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend": "deepspeed",
                    "profile": "consumer",
                    "devices": "0,1",
                    "cases": cases,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_manifest_requires_descending_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self._manifest(Path(temp), [16, 2048])
            with self.assertRaisesRegex(ValueError, "longest first"):
                load_manifest(path, "deepspeed")

    def test_peak_hold_confirmation_uses_every_later_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self._manifest(root, [2048, 16])
            manifest = load_manifest(path, "deepspeed")
            monitor = root / "monitor.csv"
            with monitor.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "phase",
                        "proc_gpu0_mem_mb",
                        "proc_gpu1_mem_mb",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "phase": "seq_2048",
                            "proc_gpu0_mem_mb": "10000",
                            "proc_gpu1_mem_mb": "9000",
                        },
                        {
                            "phase": "seq_16",
                            "proc_gpu0_mem_mb": "9600",
                            "proc_gpu1_mem_mb": "8500",
                        },
                    ]
                )
            report = verify(manifest, monitor, tolerance_mib=512)
            self.assertTrue(report["confirmed"])
            self.assertEqual(report["status"], "CONFIRMED")

    def test_peak_hold_drop_is_not_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self._manifest(root, [2048, 16])
            manifest = load_manifest(path, "deepspeed")
            monitor = root / "monitor.csv"
            with monitor.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "phase",
                        "proc_gpu0_mem_mb",
                        "proc_gpu1_mem_mb",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "phase": "seq_2048",
                            "proc_gpu0_mem_mb": "10000",
                            "proc_gpu1_mem_mb": "9000",
                        },
                        {
                            "phase": "seq_16",
                            "proc_gpu0_mem_mb": "8000",
                            "proc_gpu1_mem_mb": "8500",
                        },
                    ]
                )
            report = verify(manifest, monitor, tolerance_mib=512)
            self.assertFalse(report["confirmed"])
            self.assertEqual(report["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
