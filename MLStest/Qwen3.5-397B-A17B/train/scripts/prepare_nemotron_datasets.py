#!/usr/bin/env python3
"""Prepare Nemotron SFT datasets into LLaMA-Factory openai jsonl.

Sources (under MLS dataset root):
  - Nemotron-SFT-CUDA-v1/data/train.jsonl
  - Nemotron-SFT-SWE-v3/data/*.parquet
  - Nemotron-SFT-Competitive-Programming-v2/data/competitive_programming_cpp_*.jsonl

Outputs (under train/data/):
  - nemotron_cuda.jsonl
  - nemotron_swe.jsonl
  - nemotron_cpp.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


TASKS = ("cuda", "swe", "cpp")


def _normalize_tools(tools: Any) -> str | None:
    if tools is None:
        return None
    if isinstance(tools, str):
        text = tools.strip()
        return text if text else None
    return json.dumps(tools, ensure_ascii=False)


def _normalize_arguments(arguments: Any) -> str:
    """Store tool arguments as a JSON *string* of an object.

    - HuggingFace ``datasets`` cannot Arrow-cast varying ``arguments`` dict schemas
      across rows (keys differ per tool); a string column is stable.
    - LLaMA-Factory ``FunctionFormatter`` (patched) accepts string or dict and
      feeds Qwen3.5 tool_utils a JSON object string for ``.items()``.
    """
    parsed: Any
    if isinstance(arguments, dict):
        parsed = arguments
    elif isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            parsed = {}
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {"input": arguments}
    elif arguments is None:
        parsed = {}
    else:
        parsed = arguments

    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    return json.dumps(parsed, ensure_ascii=False)


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, list):
        return None
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        item = dict(tc)
        fn = item.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            fn["arguments"] = _normalize_arguments(fn.get("arguments"))
            if "name" in fn and fn["name"] is not None:
                fn["name"] = str(fn["name"])
            item["function"] = fn
        out.append(item)
    return out


def _normalize_messages(messages: Any) -> list[dict[str, Any]] | None:
    if not isinstance(messages, list) or not messages:
        return None
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        if role is None:
            return None
        # Keep tool_calls / name / etc. for openai converter.
        item = dict(msg)
        item["role"] = str(role)
        if "content" not in item or item["content"] is None:
            item["content"] = ""
        elif not isinstance(item["content"], str):
            item["content"] = json.dumps(item["content"], ensure_ascii=False)
        if "tool_calls" in item:
            normalized_calls = _normalize_tool_calls(item.get("tool_calls"))
            if normalized_calls is not None:
                item["tool_calls"] = normalized_calls
            else:
                item.pop("tool_calls", None)
        # Drop null-only noise fields that confuse converters.
        for key in ("reasoning_content", "name", "tool_call_id"):
            if key in item and item[key] is None:
                item.pop(key)
        out.append(item)
    return out


def _row_to_example(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = _normalize_messages(row.get("messages"))
    if messages is None:
        return None
    example: dict[str, Any] = {"messages": messages}
    tools = _normalize_tools(row.get("tools"))
    if tools is not None:
        example["tools"] = tools
    return example


def _write_jsonl(path: Path, examples: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def _iter_jsonl(paths: list[Path], max_samples: int | None) -> Iterator[dict[str, Any]]:
    written = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as err:
                    print(f"[warn] skip bad json {path}:{line_no}: {err}", file=sys.stderr)
                    continue
                example = _row_to_example(row)
                if example is None:
                    print(f"[warn] skip invalid messages {path}:{line_no}", file=sys.stderr)
                    continue
                yield example
                written += 1
                if max_samples is not None and written >= max_samples:
                    return


def prepare_cuda(src_root: Path, out_path: Path, max_samples: int | None) -> int:
    src = src_root / "Nemotron-SFT-CUDA-v1" / "data" / "train.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"CUDA jsonl not found: {src}")
    return _write_jsonl(out_path, _iter_jsonl([src], max_samples))


def prepare_cpp(src_root: Path, out_path: Path, max_samples: int | None) -> int:
    data_dir = src_root / "Nemotron-SFT-Competitive-Programming-v2" / "data"
    paths = sorted(data_dir.glob("competitive_programming_cpp_*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No competitive_programming_cpp_*.jsonl under {data_dir}. "
            "Finish HF download first."
        )
    return _write_jsonl(out_path, _iter_jsonl(paths, max_samples))


def prepare_swe(src_root: Path, out_path: Path, max_samples: int | None) -> int:
    data_dir = src_root / "Nemotron-SFT-SWE-v3" / "data"
    paths = sorted(data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {data_dir}")

    try:
        import pyarrow.parquet as pq
    except ImportError as err:
        raise ImportError("pyarrow is required to prepare SWE parquet") from err

    def _iter() -> Iterator[dict[str, Any]]:
        written = 0
        for path in paths:
            names = set(pq.ParquetFile(path).schema_arrow.names)
            if "messages" not in names:
                print(f"[warn] skip parquet without messages: {path}", file=sys.stderr)
                continue
            cols = ["messages"] + (["tools"] if "tools" in names else [])
            table = pq.read_table(path, columns=cols)
            messages_col = table.column("messages").to_pylist()
            tools_col = (
                table.column("tools").to_pylist() if "tools" in table.column_names else [None] * len(messages_col)
            )
            for messages, tools in zip(messages_col, tools_col):
                example = _row_to_example({"messages": messages, "tools": tools})
                if example is None:
                    continue
                yield example
                written += 1
                if max_samples is not None and written >= max_samples:
                    return

    n = _write_jsonl(out_path, _iter())
    expected = sorted(data_dir.glob("train-*-of-*.parquet"))
    if expected:
        # Heuristic: train-00000-of-00096.parquet
        try:
            total = int(expected[0].name.split("-of-")[1].split(".")[0])
            if len(paths) < total:
                print(
                    f"[warn] SWE download incomplete: found {len(paths)}/{total} parquet shards; "
                    f"prepared {n} samples from available files.",
                    file=sys.stderr,
                )
        except (IndexError, ValueError):
            pass
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/data2/wbw/MLStest/dataset"),
        help="Root containing Nemotron-SFT-* directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory for prepared jsonl (+ dataset_info.json)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[*TASKS, "all"],
        default=["all"],
        help="Which datasets to prepare",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per task (for smoke / partial runs)",
    )
    args = parser.parse_args()

    tasks = list(TASKS) if "all" in args.tasks else list(dict.fromkeys(args.tasks))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preparers = {
        "cuda": ("nemotron_cuda.jsonl", prepare_cuda),
        "swe": ("nemotron_swe.jsonl", prepare_swe),
        "cpp": ("nemotron_cpp.jsonl", prepare_cpp),
    }

    failed = 0
    for task in tasks:
        filename, fn = preparers[task]
        out_path = args.output_dir / filename
        print(f"[prepare] task={task} -> {out_path}")
        try:
            n = fn(args.dataset_root, out_path, args.max_samples)
        except Exception as err:
            failed += 1
            print(f"[error] {task}: {err}", file=sys.stderr)
            continue
        print(f"[ok] {task}: {n} examples")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
