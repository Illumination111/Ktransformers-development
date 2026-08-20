#!/usr/bin/env python3
"""Read-only preflight for the local KT probability-test runtime."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from compare_probability import (
    DEFAULT_KT_SRC,
    DEFAULT_OVERLAY,
    DEFAULT_SGLANG_SRC,
    DEFAULT_VERL_SRC,
    load_local_kt_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kt-src", type=Path, default=DEFAULT_KT_SRC)
    parser.add_argument("--sglang-src", type=Path, default=DEFAULT_SGLANG_SRC)
    parser.add_argument("--verl-src", type=Path, default=DEFAULT_VERL_SRC)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def load_bridge(path: Path) -> str:
    bridge = path / "verl" / "workers" / "engine" / "fsdp" / "kt_actor_bridge.py"
    spec = importlib.util.spec_from_file_location("rlft_environment_kt_bridge", bridge)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {bridge}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return str(bridge)


def install_runtime_path(args: argparse.Namespace) -> None:
    entries = [
        args.verl_src,
        args.sglang_src / "python",
        args.kt_src / "kt-kernel" / "python",
        args.kt_src / "kt-kernel" / "build" / "lib.linux-x86_64-cpython-311",
        args.overlay,
    ]
    entries.extend(Path(item) for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item)
    for entry in reversed(entries):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    load_local_kt_kernel(args)


def main() -> int:
    args = parse_args()
    install_runtime_path(args)
    paths = {
        "kt_src": args.kt_src,
        "sglang_src": args.sglang_src,
        "verl_src": args.verl_src,
        "overlay": args.overlay,
    }
    result: dict[str, object] = {
        "python": sys.executable,
        "paths": {name: str(path) for name, path in paths.items()},
        "path_checks": {name: path.exists() for name, path in paths.items()},
        "imports": {},
        "status": "PASS",
    }
    failures: list[str] = [name for name, present in result["path_checks"].items() if not present]  # type: ignore[union-attr]
    for module_name in ("transformers", "transformers.integrations.kt", "kt_kernel", "sglang"):
        try:
            module = importlib.import_module(module_name)
            result["imports"][module_name] = {"status": "OK", "file": str(getattr(module, "__file__", ""))}  # type: ignore[index]
        except Exception as exc:
            result["imports"][module_name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}  # type: ignore[index]
            failures.append(f"import:{module_name}")
    if not failures:
        try:
            result["kt_actor_bridge"] = load_bridge(args.verl_src)
        except Exception as exc:
            failures.append("import:KTActorBridge")
            result["kt_actor_bridge_error"] = f"{type(exc).__name__}: {exc}"
    try:
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
        result["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda,
            "device_count": torch.cuda.device_count(),
        }
        if args.require_cuda and not cuda:
            failures.append("cuda")
    except Exception as exc:
        failures.append("import:torch")
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    if failures:
        result["status"] = "FAIL"
        result["failures"] = failures
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 20


if __name__ == "__main__":
    raise SystemExit(main())
