#!/usr/bin/env python3
"""
后台系统指标采集脚本 —— Qwen3.5-35B-A3B FFT 测试监控器

采集指标（每 INTERVAL 秒一次）：
  - 每张 GPU 的已用显存、总显存、SM 利用率、显存利用率（整机）
  - 系统 RAM（总量、已用、可用）
  - 可选：按 --pid 进程树过滤的 proc_ram_gb / proc_gpu*_mem_mb
  - 可选：按 resource_contract.json 采集独立 cgroup 的真实内存计费
  - 磁盘 I/O 速率（读写 MB/s），优先采 /mnt/data2 所在设备
  - CPU 利用率（总体 + 每 NUMA 节点估算）

事件标记：
  监听命名管道 FIFO（--fifo 参数），训练脚本向其写入如下格式的事件：
    phase:<name>
    event:<checkpoint_start|checkpoint_end|step_start|step_end|backward_start>
    pid:<root_pid>

用法：
  python monitor.py --out /path/to/monitor.csv [--fifo /path/to/tmp/monitor_events.fifo] \
                    [--interval 2] [--disk-mount /mnt/data2] [--pid $$]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import stat
import subprocess
import tempfile
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# 依赖检查
# --------------------------------------------------------------------------- #
try:
    import psutil
except ImportError:
    print("[monitor] psutil 未安装，运行: pip install psutil", file=sys.stderr)
    sys.exit(1)

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    _GPU_COUNT = pynvml.nvmlDeviceGetCount()
except Exception:
    _NVML_AVAILABLE = False
    _GPU_COUNT = 0

# 若 pynvml 不可用，尝试通过 nvidia-smi 子进程获取 GPU 信息
def _nvidia_smi_query() -> list[dict]:
    """通过 nvidia-smi 获取 GPU 指标（pynvml 不可用时的备用方案）。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "mem_used_mb": int(float(parts[1])),
                    "mem_total_mb": int(float(parts[2])),
                    "sm_util_pct": int(float(parts[3])),
                    "mem_util_pct": int(float(parts[4])),
                })
        return gpus
    except Exception:
        return []

# 自动探测 GPU 数量（用于 CSV 列定义）
if not _NVML_AVAILABLE:
    _smi_result = _nvidia_smi_query()
    _GPU_COUNT = len(_smi_result)
    if _GPU_COUNT > 0:
        print(f"[monitor] pynvml 不可用，将使用 nvidia-smi 获取 GPU 信息（{_GPU_COUNT} 张）")

# --------------------------------------------------------------------------- #
# CSV 列定义
# --------------------------------------------------------------------------- #
def _gpu_columns(n: int) -> list[str]:
    cols = []
    for i in range(n):
        cols += [
            f"gpu{i}_mem_used_mb",
            f"gpu{i}_mem_total_mb",
            f"gpu{i}_mem_util_pct",
            f"gpu{i}_sm_util_pct",
            f"proc_gpu{i}_mem_mb",
        ]
    return cols

BASE_COLUMNS = [
    "timestamp",
    "elapsed_sec",
    "phase",
    "event",
    "cpu_util_pct",
    "ram_used_gb",
    "ram_total_gb",
    "ram_avail_gb",
    "root_pid",
    "proc_count",
    "proc_ram_gb",
    "proc_cpu_pct",
    "cgroup_memory_gb",
    "cgroup_swap_gb",
    "cgroup_anon_gb",
    "cgroup_file_gb",
    "cgroup_shmem_gb",
    "disk_read_mbps",
    "disk_write_mbps",
    "disk_read_iops",
    "disk_write_iops",
]

# --------------------------------------------------------------------------- #
# 磁盘设备解析
# --------------------------------------------------------------------------- #
def _resolve_disk_device(mount_point: str) -> str | None:
    """返回挂载点所在的磁盘设备名（如 'sda' / 'nvme0n1'）。"""
    try:
        for part in psutil.disk_partitions(all=True):
            if part.mountpoint == mount_point:
                dev = Path(part.device).name
                # 去掉分区号后缀，取磁盘整体（如 nvme0n1p1 → nvme0n1）
                for suffix in ["p1","p2","p3","p4","1","2","3","4"]:
                    if dev.endswith(suffix):
                        candidate = dev[: -len(suffix)]
                        if Path(f"/sys/block/{candidate}").exists():
                            return candidate
                if Path(f"/sys/block/{dev}").exists():
                    return dev
    except Exception:
        pass
    return None


def _get_disk_io(device: str | None) -> dict:
    """返回磁盘 I/O 计数器快照（字节数、IOPS）。"""
    try:
        counters = psutil.disk_io_counters(perdisk=True)
        if device and device in counters:
            c = counters[device]
        else:
            # 汇总所有设备
            c = psutil.disk_io_counters(perdisk=False)
        if c is None:
            return {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}
        return {
            "read_bytes": c.read_bytes,
            "write_bytes": c.write_bytes,
            "read_count": c.read_count,
            "write_count": c.write_count,
        }
    except Exception:
        return {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}


# --------------------------------------------------------------------------- #
# GPU 指标
# --------------------------------------------------------------------------- #
def _sample_gpu() -> list[dict]:
    """采集所有 GPU 指标，返回列表（每卡一个 dict）。"""
    # 优先使用 pynvml（低开销）；不可用时 fallback 到 nvidia-smi
    if not _NVML_AVAILABLE:
        return _nvidia_smi_query()
    results = []
    for i in range(_GPU_COUNT):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            results.append({
                "mem_used_mb": mem.used // (1024 * 1024),
                "mem_total_mb": mem.total // (1024 * 1024),
                "mem_util_pct": util.memory,
                "sm_util_pct": util.gpu,
            })
        except Exception:
            results.append({
                "mem_used_mb": 0,
                "mem_total_mb": 0,
                "mem_util_pct": 0,
                "sm_util_pct": 0,
            })
    return results


# --------------------------------------------------------------------------- #
# Per-process-tree GPU/CPU metrics
# --------------------------------------------------------------------------- #
def _safe_used_gpu_memory_bytes(proc_info) -> int:
    try:
        value = int(getattr(proc_info, "usedGpuMemory", 0) or 0)
    except Exception:
        return 0
    return value if value > 0 else 0


def _sample_gpu_proc_mem_nvml(pid_set: set[int]) -> list[int]:
    out = [0] * _GPU_COUNT
    if not pid_set or not _NVML_AVAILABLE:
        return out
    getters = [
        getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
        getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
        getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses_v3", None),
        getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
    ]
    for index in range(_GPU_COUNT):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:
            continue
        seen: set[int] = set()
        total = 0
        for getter in getters:
            if getter is None:
                continue
            try:
                processes = getter(handle) or []
            except Exception:
                continue
            for process in processes:
                try:
                    pid = int(process.pid)
                except Exception:
                    continue
                if pid not in pid_set or pid in seen:
                    continue
                seen.add(pid)
                total += _safe_used_gpu_memory_bytes(process)
        out[index] = total // (1024 * 1024)
    return out


def _sample_gpu_proc_mem_smi(pid_set: set[int]) -> list[int]:
    out = [0] * _GPU_COUNT
    if not pid_set or _GPU_COUNT <= 0:
        return out
    try:
        uuid_map: dict[str, int] = {}
        metadata = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if metadata.returncode == 0:
            for line in metadata.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 2:
                    uuid_map[parts[1]] = int(parts[0])

        applications = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if applications.returncode != 0:
            return out
        for line in applications.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[1])
                memory_mb = int(float(parts[2]))
            except ValueError:
                continue
            gpu_index = uuid_map.get(parts[0])
            if (
                pid in pid_set
                and memory_mb > 0
                and gpu_index is not None
                and 0 <= gpu_index < _GPU_COUNT
            ):
                out[gpu_index] += memory_mb
    except Exception:
        pass
    return out


def _sample_gpu_proc_mem(pid_set: set[int]) -> list[int]:
    if _NVML_AVAILABLE:
        return _sample_gpu_proc_mem_nvml(pid_set)
    return _sample_gpu_proc_mem_smi(pid_set)


def _collect_process_tree(root_pid: int | None) -> tuple[set[int], float, float]:
    """Return process-tree PIDs, summed RSS in decimal GB, and summed CPU%."""
    if root_pid is None or root_pid <= 0:
        return set(), 0.0, 0.0
    try:
        root = psutil.Process(root_pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return set(), 0.0, 0.0
    try:
        processes = [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        processes = [root]

    exclude: set[int] = {os.getpid()}
    try:
        exclude.update(
            child.pid
            for child in psutil.Process(os.getpid()).children(recursive=True)
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    pid_set: set[int] = set()
    rss = 0
    cpu = 0.0
    for process in processes:
        try:
            if process.pid in exclude:
                continue
            pid_set.add(process.pid)
            rss += int(process.memory_info().rss)
            cpu += float(process.cpu_percent(interval=None))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return pid_set, rss / 1e9, cpu


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_memory_stat(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            key, raw = line.split()
            values[key] = int(raw)
    except (OSError, ValueError):
        return {}
    return values


def _sample_cgroup_memory(cgroup: Path | None) -> dict[str, float | None]:
    if cgroup is None:
        return {}
    current = _read_int(cgroup / "memory.current")
    swap = _read_int(cgroup / "memory.swap.current")
    stat_values = _read_memory_stat(cgroup / "memory.stat")

    def decimal_gb(value: int | None) -> float | None:
        return None if value is None else value / 1e9

    return {
        "memory_gb": decimal_gb(current),
        "swap_gb": decimal_gb(swap),
        "anon_gb": decimal_gb(stat_values.get("anon")),
        "file_gb": decimal_gb(stat_values.get("file")),
        "shmem_gb": decimal_gb(stat_values.get("shmem")),
    }


# --------------------------------------------------------------------------- #
# 主监控循环
# --------------------------------------------------------------------------- #
class Monitor:
    def __init__(
        self,
        out_path: str,
        fifo_path: str | None,
        interval: float,
        disk_mount: str,
        root_pid: int | None = None,
        resource_contract: str | None = None,
    ):
        self.out_path = out_path
        self.fifo_path = fifo_path
        self.interval = interval
        self.disk_mount = disk_mount
        self.disk_device = _resolve_disk_device(disk_mount)
        self.root_pid = root_pid
        self.resource_contract = (
            Path(resource_contract) if resource_contract else None
        )
        self.cgroup_path: Path | None = None
        self._running = True
        self._phase = "init"
        self._event = ""
        self._start_time = time.time()

        # NUMA CPU 利用率（用 /proc/stat 近似，每插槽 CPU 列表）
        self._numa_cpu_map = self._build_numa_map()

        print(f"[monitor] 输出文件: {out_path}", flush=True)
        print(f"[monitor] 磁盘设备: {self.disk_device or '(all devices)'}", flush=True)
        print(f"[monitor] GPU 数量: {_GPU_COUNT}", flush=True)
        print(f"[monitor] FIFO 路径: {fifo_path}", flush=True)
        print(f"[monitor] 进程树根 PID: {root_pid}", flush=True)
        print(
            f"[monitor] 资源合同: {self.resource_contract or '(disabled)'}",
            flush=True,
        )

        # 打开 CSV 写入
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cols = BASE_COLUMNS + _gpu_columns(_GPU_COUNT)
        self._f = open(out_path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._f, fieldnames=cols, extrasaction="ignore")
        self._writer.writeheader()
        self._f.flush()

        # FIFO 监听线程
        if fifo_path:
            self._ensure_fifo(fifo_path)
            self._fifo_thread = threading.Thread(
                target=self._fifo_reader, daemon=True
            )
            self._fifo_thread.start()

    @staticmethod
    def _build_numa_map() -> dict[int, list[int]]:
        """解析 /sys/devices/system/node/ 得到 NUMA 节点→CPU 列表映射。"""
        numa_map: dict[int, list[int]] = {}
        base = Path("/sys/devices/system/node")
        if not base.exists():
            return numa_map
        for node_dir in sorted(base.glob("node[0-9]*")):
            try:
                node_id = int(node_dir.name[4:])
                cpulist_file = node_dir / "cpulist"
                cpulist = cpulist_file.read_text().strip()
                cpus: list[int] = []
                for part in cpulist.split(","):
                    if "-" in part:
                        lo, hi = part.split("-")
                        cpus.extend(range(int(lo), int(hi) + 1))
                    else:
                        cpus.append(int(part))
                numa_map[node_id] = cpus
            except Exception:
                pass
        return numa_map

    @staticmethod
    def _ensure_fifo(path: str) -> None:
        p = Path(path)
        if p.exists():
            if not stat.S_ISFIFO(p.stat().st_mode):
                p.unlink()
                os.mkfifo(path)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            os.mkfifo(path)

    def _fifo_reader(self) -> None:
        """持续从 FIFO 读取事件行（阻塞式）。"""
        while self._running:
            try:
                # Keeping a writer endpoint open avoids an EOF/reopen race when
                # short-lived event writers close the FIFO.
                fd = os.open(self.fifo_path, os.O_RDWR)
                with os.fdopen(fd, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("phase:"):
                            self._phase = line[6:]
                            print(f"[monitor] Phase 切换 → {self._phase}", flush=True)
                        elif line.startswith("event:"):
                            self._event = line[6:]
                            print(f"[monitor] 事件: {self._event}", flush=True)
                        elif line.startswith("pid:"):
                            try:
                                self.root_pid = int(line[4:].strip())
                                print(
                                    f"[monitor] 进程树根 PID 更新 → {self.root_pid}",
                                    flush=True,
                                )
                            except ValueError:
                                print(
                                    f"[monitor] 忽略非法 pid 事件: {line}",
                                    flush=True,
                                )
            except Exception:
                if self._running:
                    time.sleep(0.5)

    def _resolve_cgroup_path(self) -> Path | None:
        if self.cgroup_path is not None:
            return self.cgroup_path
        if self.resource_contract is None or not self.resource_contract.is_file():
            return None
        try:
            contract = json.loads(self.resource_contract.read_text())
            candidate = Path(str(contract["cgroup"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not str(candidate).startswith("/sys/fs/cgroup/"):
            print(f"[monitor] 忽略非法 cgroup 路径: {candidate}", flush=True)
            return None
        if not (candidate / "memory.current").is_file():
            return None
        self.cgroup_path = candidate
        print(f"[monitor] cgroup 内存路径: {candidate}", flush=True)
        return candidate

    def _sample_once(self, prev_disk: dict, dt: float) -> dict:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        elapsed = time.time() - self._start_time

        # CPU
        cpu_util = psutil.cpu_percent(interval=None)

        # RAM
        vm = psutil.virtual_memory()
        ram_used_gb = vm.used / 1e9
        ram_total_gb = vm.total / 1e9
        ram_avail_gb = vm.available / 1e9
        pid_set, proc_ram_gb, proc_cpu_pct = _collect_process_tree(
            self.root_pid
        )
        proc_gpu = _sample_gpu_proc_mem(pid_set)
        cgroup_memory = _sample_cgroup_memory(self._resolve_cgroup_path())

        # Disk
        curr_disk = _get_disk_io(self.disk_device)
        read_mbps = (curr_disk["read_bytes"] - prev_disk["read_bytes"]) / 1e6 / dt
        write_mbps = (curr_disk["write_bytes"] - prev_disk["write_bytes"]) / 1e6 / dt
        read_iops = (curr_disk["read_count"] - prev_disk["read_count"]) / dt
        write_iops = (curr_disk["write_count"] - prev_disk["write_count"]) / dt

        row: dict = {
            "timestamp": now,
            "elapsed_sec": f"{elapsed:.1f}",
            "phase": self._phase,
            "event": self._event,
            "cpu_util_pct": f"{cpu_util:.1f}",
            "ram_used_gb": f"{ram_used_gb:.2f}",
            "ram_total_gb": f"{ram_total_gb:.2f}",
            "ram_avail_gb": f"{ram_avail_gb:.2f}",
            "root_pid": self.root_pid if self.root_pid else "",
            "proc_count": len(pid_set),
            "proc_ram_gb": f"{proc_ram_gb:.2f}",
            "proc_cpu_pct": f"{proc_cpu_pct:.1f}",
            "cgroup_memory_gb": self._format_optional(
                cgroup_memory.get("memory_gb")
            ),
            "cgroup_swap_gb": self._format_optional(
                cgroup_memory.get("swap_gb")
            ),
            "cgroup_anon_gb": self._format_optional(
                cgroup_memory.get("anon_gb")
            ),
            "cgroup_file_gb": self._format_optional(
                cgroup_memory.get("file_gb")
            ),
            "cgroup_shmem_gb": self._format_optional(
                cgroup_memory.get("shmem_gb")
            ),
            "disk_read_mbps": f"{read_mbps:.1f}",
            "disk_write_mbps": f"{write_mbps:.1f}",
            "disk_read_iops": f"{read_iops:.0f}",
            "disk_write_iops": f"{write_iops:.0f}",
        }

        # GPU
        for i, g in enumerate(_sample_gpu()):
            row[f"gpu{i}_mem_used_mb"] = g["mem_used_mb"]
            row[f"gpu{i}_mem_total_mb"] = g["mem_total_mb"]
            row[f"gpu{i}_mem_util_pct"] = g["mem_util_pct"]
            row[f"gpu{i}_sm_util_pct"] = g["sm_util_pct"]
            row[f"proc_gpu{i}_mem_mb"] = (
                proc_gpu[i] if i < len(proc_gpu) else 0
            )

        # 重置事件（单次触发）
        self._event = ""
        return row, curr_disk

    @staticmethod
    def _format_optional(value: float | None) -> str:
        return "" if value is None else f"{value:.6f}"

    def run(self) -> None:
        # 初始化 CPU percent（第一次调用返回 0）
        psutil.cpu_percent(interval=None)
        _collect_process_tree(self.root_pid)
        prev_disk = _get_disk_io(self.disk_device)
        prev_t = time.time()

        print("[monitor] 开始采样...", flush=True)
        while self._running:
            time.sleep(self.interval)
            now_t = time.time()
            dt = max(now_t - prev_t, 0.01)
            row, prev_disk = self._sample_once(prev_disk, dt)
            self._writer.writerow(row)
            prev_t = now_t

        self._f.close()
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        print("[monitor] 已停止。", flush=True)

    def stop(self) -> None:
        self._running = False


# --------------------------------------------------------------------------- #
# 信号处理
# --------------------------------------------------------------------------- #
_monitor_instance: Monitor | None = None


def _sig_handler(signum, frame):
    print(f"\n[monitor] 收到信号 {signum}，正在停止...", flush=True)
    if _monitor_instance:
        _monitor_instance.stop()


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main():
    global _monitor_instance

    parser = argparse.ArgumentParser(description="FFT 系统指标监控器")
    parser.add_argument("--out", required=True, help="输出 CSV 文件路径")
    parser.add_argument(
        "--fifo",
        default=str(Path(tempfile.gettempdir()) / "fft_monitor_events.fifo"),
        help="命名管道路径（训练脚本向此写入事件）",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="采样间隔（秒）")
    parser.add_argument(
        "--disk-mount",
        default="/mnt/data2",
        help="重点监控的磁盘挂载点",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="训练启动脚本 PID；监控该进程树的 CPU/GPU 内存",
    )
    parser.add_argument(
        "--resource-contract",
        help="resource_contract.json used to resolve the dedicated cgroup",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    _monitor_instance = Monitor(
        out_path=args.out,
        fifo_path=args.fifo,
        interval=args.interval,
        disk_mount=args.disk_mount,
        root_pid=args.pid,
        resource_contract=args.resource_contract,
    )
    _monitor_instance.run()


if __name__ == "__main__":
    main()
