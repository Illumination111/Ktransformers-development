from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from aggregate_sweep_results import aggregate_run
from finetune_train_with_timing import _load_training_config, _manifest_backend
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

    def test_pure_bf16_is_scoped_to_ktransformers_config(self) -> None:
        script = (
            SCRIPT_DIR / "run_finetune_perf_sweep_bf16_common.sh"
        ).read_text(encoding="utf-8")
        make_config = script.split("make_train_config() {", 1)[1].split(
            "make_megatrain_config() {", 1
        )[0]
        backend_branch = make_config.split(
            'if [[ "${BACKEND}" == "ktransformers" ]]; then', 1
        )[1]
        kt_branch, other_backends = backend_branch.split("    else\n", 1)
        other_backends = other_backends.split("    fi\n", 1)[0]

        self.assertIn(
            'set_yaml_value "${config}" pure_bf16 "true"',
            kt_branch,
        )
        self.assertNotIn("pure_bf16", other_backends)

    def test_all_backends_use_original_unprotected_sequence_sweep(self) -> None:
        script = (
            SCRIPT_DIR / "run_finetune_perf_sweep_bf16_common.sh"
        ).read_text(encoding="utf-8")
        run_one = script.split("run_one_sequence() {", 1)[1].split(
            "write_profile_sweep_manifest() {", 1
        )[0]
        run_profile = script.split("run_profile() {", 1)[1].split(
            "check_files_and_environment", 1
        )[0]

        self.assertIn(
            'local -a execution_command=("${full_command[@]}")',
            run_one,
        )
        self.assertNotIn('"${GPU_LIFECYCLE_GUARD}"', run_one)
        self.assertIn("run_one_sequence", run_profile)
        self.assertNotIn("run_persistent_profile", run_profile)
        self.assertIn(
            "GPU lifecycle/peak-hold protection disabled",
            run_profile,
        )
        self.assertLess(
            script.index(
                "readonly -a SERVER_SEQUENCE_LENGTHS="
                "(32 64 128 256 512 1024 2048 4096)"
            ),
            script.index("usage()"),
        )
        self.assertLess(
            script.index(
                "readonly -a CONSUMER_SEQUENCE_LENGTHS="
                "(16 32 64 128 256 512 1024 2048)"
            ),
            script.index("usage()"),
        )

    def test_peak_hold_failure_does_not_overwrite_training_exit(self) -> None:
        script = (
            SCRIPT_DIR / "run_finetune_perf_sweep_bf16_common.sh"
        ).read_text(encoding="utf-8")
        persistent_profile = script.split(
            "run_persistent_profile() {", 1
        )[1].split("run_profile() {", 1)[0]

        self.assertIn("peak_hold_failed=1", persistent_profile)
        self.assertNotIn("exit_code=97", persistent_profile)

    def test_legacy_peak_hold_exit_still_aggregates_tps(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            profile_dir = Path(temp) / "consumer_2gpu_batch2"
            run_dir = profile_dir / "seq_2048"
            timing_dir = run_dir / "step_timing"
            timing_dir.mkdir(parents=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "backend": "ktransformers",
                        "profile": "consumer",
                        "benchmark_class": "exact_model_full_finetune",
                        "precision": "bf16",
                        "modality": "text_only",
                        "model_load_architecture": "Qwen3_5MoeForCausalLM",
                        "weight_source": "pretrained_checkpoint",
                        "checkpoint_compatible": True,
                        "llamafactory_backend": True,
                        "finetuning_type": "full",
                        "sequence_length": 2048,
                        "steps": 15,
                        "warmup_steps": 5,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "exit_code.txt").write_text("97\n", encoding="utf-8")
            (profile_dir / "gpu_peak_hold.json").write_text(
                json.dumps({"status": "FAILED", "confirmed": False}),
                encoding="utf-8",
            )
            (timing_dir / "step_timing.json").write_text(
                json.dumps(
                    {
                        "timing_mode": "coarse_host_wall_no_cuda_sync",
                        "num_stable_steps": 10,
                        "instrumentation": {
                            "forced_cuda_synchronize": False,
                            "backend_internal_probes": False,
                            "system_resource_monitor": False,
                            "per_step_file_io": False,
                        },
                        "aggregate_stable": {
                            "step_total_sec": {"mean_sec": 4.0},
                            "forward_sec": {"mean_sec": 1.0},
                            "backward_sec": {"mean_sec": 2.0},
                            "optimizer_sec": {"mean_sec": 1.0},
                        },
                        "tps_attribution": {"stable_tps": 1024.0},
                    }
                ),
                encoding="utf-8",
            )

            row = aggregate_run(run_dir / "run_config.json")

            self.assertEqual(row["stable_steps"], 10)
            self.assertEqual(row["stable_tps"], 1024.0)
            self.assertEqual(
                row["status"],
                "GPU_PEAK_HOLD_BROKEN_NOT_OOM",
            )

    def test_generated_training_yaml_is_loaded_as_argument_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "train_config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name_or_path: /models/qwen",
                        "use_kt: true",
                        "cutoff_len: 2048",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = _load_training_config(str(config_path))

            self.assertEqual(config["model_name_or_path"], "/models/qwen")
            self.assertIs(config["use_kt"], True)
            self.assertEqual(config["cutoff_len"], 2048)

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
