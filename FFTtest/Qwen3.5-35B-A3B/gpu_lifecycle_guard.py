#!/usr/bin/env python3
"""Run one persistent benchmark profile and verify final GPU release.

The profile process owns its allocations while it moves from the longest to the
shortest sequence. This guard does not reserve artificial buffers: process exit
is the final lifetime boundary. Any worker left behind after the launcher exits
is terminated before another profile starts.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GPU_BUSY_EXIT = 87
GPU_RELEASE_FAILED_EXIT = 88
LAUNCH_FAILED_EXIT = 86


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_devices(value: str) -> list[int]:
    devices: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item or not item.isdigit():
            raise argparse.ArgumentTypeError(
                f"devices must be comma-separated physical GPU indices: {value}"
            )
        device = int(item)
        if device in devices:
            raise argparse.ArgumentTypeError(f"duplicate GPU index: {device}")
        devices.append(device)
    if not devices:
        raise argparse.ArgumentTypeError("at least one GPU index is required")
    return devices


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_nvidia_smi(arguments: list[str]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"nvidia-smi failed: {detail}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_mib(value: str) -> int | None:
    cleaned = value.replace("MiB", "").strip()
    if cleaned in {"", "N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    return int(cleaned)


def query_gpu_state(devices: list[int]) -> dict[str, dict[str, Any]]:
    selected = set(devices)
    state: dict[str, dict[str, Any]] = {}
    uuid_to_index: dict[str, str] = {}
    for line in run_nvidia_smi(
        [
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ]
    ):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        index = int(fields[0])
        if index not in selected:
            continue
        key = str(index)
        state[key] = {
            "uuid": fields[1],
            "memory_used_mib": parse_mib(fields[2]),
            "compute_processes": [],
        }
        uuid_to_index[fields[1]] = key
    missing = selected.difference(int(index) for index in state)
    if missing:
        raise RuntimeError(
            f"selected physical GPU indices were not reported: {sorted(missing)}"
        )

    for line in run_nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    ):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or fields[0] not in uuid_to_index:
            continue
        state[uuid_to_index[fields[0]]]["compute_processes"].append(
            {
                "pid": int(fields[1]),
                "used_memory_mib": parse_mib(fields[2]),
            }
        )
    return state


def compute_pids(state: dict[str, dict[str, Any]]) -> set[int]:
    return {
        int(process["pid"])
        for gpu in state.values()
        for process in gpu["compute_processes"]
    }


def read_process_table() -> dict[int, dict[str, int]]:
    table: dict[int, dict[str, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            after_name = raw[raw.rfind(")") + 2 :].split()
            table[int(entry.name)] = {
                "ppid": int(after_name[1]),
                "pgrp": int(after_name[2]),
                "session": int(after_name[3]),
                "starttime": int(after_name[19]),
                "uid": (entry / "status").stat().st_uid,
            }
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    return table


def record_owned_processes(
    seen: dict[int, int],
    root_pid: int,
    session_id: int,
) -> dict[int, dict[str, int]]:
    table = read_process_table()
    owned = {
        pid
        for pid, info in table.items()
        if pid == root_pid or info["session"] == session_id
    }
    changed = True
    while changed:
        changed = False
        for pid, info in table.items():
            if pid not in owned and info["ppid"] in owned:
                owned.add(pid)
                changed = True
    current_uid = os.getuid()
    for pid in owned:
        info = table.get(pid)
        if info is not None and info["uid"] == current_uid:
            seen[pid] = info["starttime"]
    return table


def matching_seen_pids(
    seen: dict[int, int],
    table: dict[int, dict[str, int]] | None = None,
) -> set[int]:
    table = table or read_process_table()
    current_uid = os.getuid()
    return {
        pid
        for pid, starttime in seen.items()
        if pid in table
        and table[pid]["starttime"] == starttime
        and table[pid]["uid"] == current_uid
    }


def signal_owned_processes(
    seen: dict[int, int],
    session_id: int,
    signum: int,
) -> list[int]:
    table = read_process_table()
    survivors = matching_seen_pids(seen, table)
    session_members = {
        pid for pid in survivors if table[pid]["session"] == session_id
    }
    signaled: set[int] = set()
    if session_members:
        try:
            os.killpg(session_id, signum)
            signaled.update(session_members)
        except ProcessLookupError:
            pass
    for pid in survivors.difference(signaled):
        try:
            os.kill(pid, signum)
            signaled.add(pid)
        except ProcessLookupError:
            pass
    return sorted(signaled)


def wait_for_process_exit(seen: dict[int, int], timeout: float) -> set[int]:
    deadline = time.monotonic() + timeout
    survivors = matching_seen_pids(seen)
    while survivors and time.monotonic() < deadline:
        time.sleep(0.1)
        survivors = matching_seen_pids(seen)
    return survivors


def acquire_gpu_locks(devices: list[int]) -> list[Any]:
    handles: list[Any] = []
    for device in devices:
        handle: Any | None = None
        try:
            path = Path(tempfile.gettempdir()) / (
                f"fft-qwen35-uid{os.getuid()}-gpu-{device}.lock"
            )
            handle = path.open("a+", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if handle is not None and not handle.closed:
                handle.close()
            for acquired in handles:
                acquired.close()
            raise RuntimeError(
                f"could not acquire FFT GPU lock {path}: {error}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        handles.append(handle)
    return handles


def memory_returned_to_baseline(
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    tolerance_mib: int,
) -> bool:
    for index, before in baseline.items():
        before_used = before.get("memory_used_mib")
        after_used = current.get(index, {}).get("memory_used_mib")
        if before_used is None or after_used is None:
            continue
        if int(after_used) > int(before_used) + tolerance_mib:
            return False
    return True


def normalize_return_code(return_code: int) -> int:
    return 128 + abs(return_code) if return_code < 0 else return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=parse_devices, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--release-timeout", type=float, default=60.0)
    parser.add_argument("--cleanup-grace", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--memory-tolerance-mib", type=int, default=512)
    parser.add_argument(
        "--skip-device-query",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.release_timeout <= 0 or args.cleanup_grace <= 0:
        parser.error("timeouts must be positive")
    if args.poll_interval <= 0 or args.memory_tolerance_mib < 0:
        parser.error("poll interval must be positive and tolerance non-negative")

    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": utc_now(),
        "devices": args.devices,
        "allocation_lifetime": "persistent_profile_process_lifetime",
        "artificial_gpu_reservation": False,
        "empty_cache_after_longest_sequence": False,
        "exclusive_fft_gpu_locks": True,
        "release_required_before_next_profile": True,
        "gpu_busy_is_oom": False,
        "release_failure_is_oom": False,
        "command": command,
        "cwd": str(args.cwd) if args.cwd else None,
    }
    lock_handles: list[Any] = []
    process: subprocess.Popen[Any] | None = None
    seen: dict[int, int] = {}
    received_signal: int | None = None

    def forward_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    try:
        try:
            lock_handles = acquire_gpu_locks(args.devices)
        except RuntimeError as error:
            report.update(
                {
                    "status": "GPU_LOCK_BUSY",
                    "error": str(error),
                    "finished_at": utc_now(),
                }
            )
            write_report(args.report, report)
            print(f"[gpu-lifecycle] {error}", file=sys.stderr)
            return GPU_BUSY_EXIT

        baseline: dict[str, dict[str, Any]] = {}
        if not args.skip_device_query:
            try:
                baseline = query_gpu_state(args.devices)
            except (OSError, RuntimeError, ValueError) as error:
                report.update(
                    {
                        "status": "GPU_QUERY_FAILED_BEFORE_TEST",
                        "error": str(error),
                        "finished_at": utc_now(),
                    }
                )
                write_report(args.report, report)
                print(f"[gpu-lifecycle] {error}", file=sys.stderr)
                return GPU_BUSY_EXIT
            report["baseline"] = baseline
            busy_pids = sorted(compute_pids(baseline))
            if busy_pids:
                report.update(
                    {
                        "status": "GPU_BUSY_BEFORE_TEST",
                        "preexisting_compute_pids": busy_pids,
                        "finished_at": utc_now(),
                    }
                )
                write_report(args.report, report)
                print(
                    "[gpu-lifecycle] selected GPUs already have compute "
                    f"processes: {busy_pids}; benchmark not started",
                    file=sys.stderr,
                )
                return GPU_BUSY_EXIT

        try:
            process = subprocess.Popen(
                command,
                cwd=args.cwd,
                start_new_session=True,
            )
        except OSError as error:
            report.update(
                {
                    "status": "LAUNCH_FAILED",
                    "error": str(error),
                    "finished_at": utc_now(),
                }
            )
            write_report(args.report, report)
            print(f"[gpu-lifecycle] launch failed: {error}", file=sys.stderr)
            return LAUNCH_FAILED_EXIT

        report["launcher_pid"] = process.pid
        report["session_id"] = process.pid
        while process.poll() is None:
            record_owned_processes(seen, process.pid, process.pid)
            time.sleep(args.poll_interval)
        child_exit = normalize_return_code(process.returncode)
        record_owned_processes(seen, process.pid, process.pid)
        report["training_exit_code"] = child_exit
        report["observed_processes"] = sorted(seen)

        cleanup: dict[str, Any] = {}
        survivors = matching_seen_pids(seen)
        if survivors:
            cleanup["sigterm_pids"] = signal_owned_processes(
                seen,
                process.pid,
                signal.SIGTERM,
            )
            survivors = wait_for_process_exit(seen, args.cleanup_grace)
        if survivors:
            cleanup["sigkill_pids"] = signal_owned_processes(
                seen,
                process.pid,
                signal.SIGKILL,
            )
            survivors = wait_for_process_exit(seen, args.cleanup_grace)
        cleanup["surviving_pids"] = sorted(survivors)
        report["worker_cleanup"] = cleanup

        release_confirmed = not survivors
        final_state: dict[str, dict[str, Any]] = {}
        foreign_pids: set[int] = set()
        release_error: str | None = None
        if release_confirmed and not args.skip_device_query:
            deadline = time.monotonic() + args.release_timeout
            while time.monotonic() < deadline:
                try:
                    final_state = query_gpu_state(args.devices)
                except (OSError, RuntimeError, ValueError) as error:
                    release_error = str(error)
                    release_confirmed = False
                    break
                current_pids = compute_pids(final_state)
                owned_gpu_pids = current_pids.intersection(seen)
                foreign_pids = current_pids.difference(seen)
                if foreign_pids:
                    release_confirmed = False
                    break
                if (
                    not owned_gpu_pids
                    and memory_returned_to_baseline(
                        baseline,
                        final_state,
                        args.memory_tolerance_mib,
                    )
                ):
                    release_confirmed = True
                    break
                release_confirmed = False
                time.sleep(args.poll_interval)
        report["final"] = final_state
        report["foreign_compute_pids_after_test"] = sorted(foreign_pids)
        report["release_error"] = release_error
        report["release_confirmed"] = release_confirmed
        report["finished_at"] = utc_now()

        if not release_confirmed:
            report["status"] = "GPU_RELEASE_UNCONFIRMED"
            write_report(args.report, report)
            print(
                "[gpu-lifecycle] GPU release was not confirmed; stopping "
                "before another benchmark item can start",
                file=sys.stderr,
            )
            return GPU_RELEASE_FAILED_EXIT

        report["status"] = "RELEASED"
        write_report(args.report, report)
        print(
            "[gpu-lifecycle] training workers exited and selected GPU "
            "memory returned to baseline",
            file=sys.stderr,
        )
        if received_signal is not None:
            return 128 + received_signal
        return child_exit
    finally:
        for handle in lock_handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
