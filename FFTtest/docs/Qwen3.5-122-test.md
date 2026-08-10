# Qwen3.5-122B-A10B Server 测试结果分析

本文分析 `Qwen3.5-122B-A10B/test_log` 中三个后端的 **server** 全量扫描日志（本轮未跑 consumer，也未包含 APTMoE）：

| 后端 | Server |
|---|---|
| KTransformers | [summary](../Qwen3.5-122B-A10B/test_log/20260803_115140_KTRANSFORMERS_BF16_FULL_SWEEP/summary.md) |
| MegaTrain | [summary](../Qwen3.5-122B-A10B/test_log/20260802_094707_MEGATRAIN_BF16_FULL_SWEEP/summary.md) |
| DeepSpeed | [summary](../Qwen3.5-122B-A10B/test_log/20260802_133440_DEEPSPEED_BF16_FULL_SWEEP/summary.md) |

## 测量口径

- 三组测试均为 BF16、text-only、配置为 full finetuning、LoRA rank 0，per-device batch size 1，gradient accumulation steps 1。
- Server 使用 8 张 GPU、global batch size 8，测试 sequence length 32–4096；每个 sequence 均为独立训练进程（`persistent_profile_process=false`）。
- 每档配置 15 step，前 5 step 为 warmup；成功档的时间均为后 10 个稳定 step 的均值。
- `TPS = global batch size × sequence length / 稳定 step 平均时间`，单位为 token/s。它按配置长度计算 token 数，不会自动扣除 padding、mask 或被跳过的 batch。
- 表中 CPU 内存是训练进程树的 RSS 求和峰值，单位为十进制 GB。多进程共享页可能被重复计入，因此它适合比较同一采集方法下的进程树占用，不等同于整机实际新增物理内存。同目录 `memory_summary.json` 另给出 `host_used_peak_gb_decimal`（整机 used 采样峰值）。新一轮 KTransformers 还为每个 case 建立了独立 cgroup v2 scope，并以 `memory.current` 峰值作为该任务的主内存口径；该 scope 的 `memory.max` 和 `memory.swap.max` 均为 `max`，只改进记账隔离，不限制或节省内存。
- Server 表中的“单卡 GPU 峰值”取该档 8 张任务 GPU 的 `task_peak_gib` 最大值，不是 8 张卡的显存之和；单位为 GiB，不包含其他任务和驱动基线占用。
- DeepSpeed、KTransformers 的计时模式为 `coarse_host_wall_no_cuda_sync`，没有在每个阶段强制 CUDA 同步。MegaTrain 为 `megatrain_host_wall_with_backend_cuda_sync`：forward/backward 使用 CUDA event，optimizer 和 step total 使用 host wall time。不同计时模式的阶段时间不宜直接当作完全相同的测量量。
- forward、backward、optimizer 是各自计时边界的平均值；由于异步执行、同步点和阶段交叠，它们不保证严格相加等于 step time。
- 1 TiB 的判断阈值是 1099.51 GB。日志只采样并可视化内存，没有因超过阈值自动终止，也没有自动把超过阈值判成 OOM。
- 本轮 KTransformers 已在 shared-expert 参数契约修复之后运行（见 [Qwen3.5-KTransformers-shared-expert-fix.md](./Qwen3.5-KTransformers-shared-expert-fix.md)）；成功档日志中 `logical_total=122111526912`、`contract=OK`。

## 测试身份与可比性

| 后端 | benchmark class | 权重来源 | 模型范围 | Server 状态 |
|---|---|---|---|---|
| KTransformers | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5-122B text model 端到端全量微调 | 8/8 独立进程成功 |
| MegaTrain | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5-122B text model 端到端全量微调 | 8/8 独立进程成功 |
| DeepSpeed | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5-122B text model 端到端全量微调 | 8/8 均在训练启动后被杀死，无稳定 timing |

## Server（8 GPU / global batch size 8）

### KTransformers

原始汇总：[summary.md](../Qwen3.5-122B-A10B/test_log/20260803_115140_KTRANSFORMERS_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../Qwen3.5-122B-A10B/test_log/20260803_115140_KTRANSFORMERS_BF16_FULL_SWEEP/sweep_results.csv)

| Seq | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 27.68 | 2460.26 | 66.816 | 490.42 | 16.252 | 42.671 | 7.672 | SUCCESS |
| 2048 | 20.25 | 2307.67 | 41.188 | 397.78 | 9.893 | 23.294 | 7.793 | SUCCESS |
| 1024 | 19.20 | 2230.71 | 28.003 | 292.54 | 6.640 | 13.527 | 7.635 | SUCCESS |
| 512 | 16.02 | 2192.86 | 21.594 | 189.68 | 5.120 | 9.173 | 7.099 | SUCCESS |
| 256 | 16.34 | 2173.35 | 18.389 | 111.37 | 4.296 | 7.319 | 6.571 | SUCCESS |
| 128 | 16.21 | 2164.19 | 17.786 | 57.57 | 3.888 | 6.586 | 7.112 | SUCCESS |
| 64 | 16.20 | 2158.99 | 16.457 | 31.11 | 3.658 | 5.591 | 7.004 | SUCCESS |
| 32 | 16.18 | 2157.08 | 15.865 | 16.14 | 3.380 | 5.349 | 6.935 | SUCCESS |

8 档退出码均为 0，每档都有 10 个稳定 step，`contract=OK`（`logical_total=122111526912`，`kt_wrappers=48`）。进程树 RSS 求和峰值约 2157–2460 GB，但专用 cgroup `memory.current` 峰值为 1684.82–1966.85 GB，说明 RSS 求和明显重复计入了多进程共享页。

seq=4096 稳定 step 为 66.816 s，TPS 490.42；单卡任务显存峰值 27.68 GiB，cgroup 峰值 1966.85 GB，其中 anonymous memory 约 1961.71 GB、swap 为 0。整机总内存 2164.13 GB，采样峰值 `host_used=2003.94 GB`，最低 `MemAvailable=160.19 GB`，因此“还有约 200 GB 余量”更准确地说是约 160 GB。

该档能跑通的直接原因是峰值仍在物理容量内，而且 sequence length 不会让全部内存随之翻倍：122B 模型权重、梯度和优化器状态是与序列长度无关的固定大头，只有激活等长度相关项增长。配置中的 non-reentrant gradient checkpointing、FSDP2 `reshard_after_forward=true` 以及 KTransformers distributed checkpoint-forward reuse 抑制了激活和中间结果的常驻增长。因此从 seq=2048 到 4096，cgroup 峰值只从 1836.40 GB 增到 1966.85 GB（+130.46 GB），而非整体翻倍；单卡 GPU 也仍有约 20.31 GiB 未使用。

与 2026-08-01 失败轮比较，模型、数据集、BF16/FSDP2、batch size、gradient checkpointing 与 distributed checkpoint-forward reuse 配置都相同；旧轮在进入训练后约 8 分钟收到 SIGKILL，未产生任何完整 step。旧轮最高进程树 RSS 2458.52 GB，与新轮 2460.26 GB 几乎相同；而旧轮已采样的整机峰值仅 1963.72 GB，当时仍有 200.41 GB `MemAvailable`。因此，旧轮失败不能仅凭 SIGKILL 就定性为“稳态整机 RAM 不足”；更可能是监控间隔内未捕获的瞬时分配、NUMA/锁页压力，或外部/session 级 kill。新轮的独立 systemd cgroup scope 提供了更准确的任务记账和进程隔离，但因为 `memory.max=max`，它本身不是内存节省优化。当前账号无权读取内核/systemd-oomd 历史日志，所以旧 SIGKILL 的唯一根因仍无法从现有产物中证实。

### MegaTrain

原始汇总：[summary.md](../Qwen3.5-122B-A10B/test_log/20260802_094707_MEGATRAIN_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../Qwen3.5-122B-A10B/test_log/20260802_094707_MEGATRAIN_BF16_FULL_SWEEP/sweep_results.csv)

| Seq | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 46.69 | 5600.82 | 76.417 | 428.81 | 9.680 | 46.657 | 6.274 | SUCCESS |
| 2048 | 46.70 | 5600.83 | 74.773 | 219.12 | 9.667 | 45.897 | 6.344 | SUCCESS |
| 1024 | 44.82 | 5600.45 | 71.908 | 113.92 | 9.664 | 43.428 | 6.234 | SUCCESS |
| 512 | 42.60 | 5597.83 | 74.769 | 54.78 | 9.656 | 45.453 | 6.670 | SUCCESS |
| 256 | 42.44 | 5600.52 | 70.600 | 29.01 | 9.662 | 42.155 | 6.376 | SUCCESS |
| 128 | 42.38 | 5597.82 | 75.229 | 13.61 | 9.656 | 46.034 | 6.635 | SUCCESS |
| 64 | 42.34 | 5600.47 | 76.329 | 6.71 | 9.656 | 44.887 | 7.066 | SUCCESS |
| 32 | 42.48 | 5599.66 | 77.839 | 3.29 | 9.654 | 48.692 | 6.447 | SUCCESS |

八档均为 `SUCCESS`、exit code 0，每档 10 个完整稳定 step，`contract=OK`（`trainable=total=122111526912`）。计时模式为 `megatrain_host_wall_with_backend_cuda_sync`。

进程树 RSS 求和峰值约 5598–5601 GB，远高于 KTransformers/DeepSpeed；但同档 `host_used` 仅约 1694–1697 GB，说明多进程共享页被进程树 RSS 大幅重复计入。单卡 GPU 峰值约 42–47 GiB，明显高于另外两个后端。步时大致落在 70–78 s，TPS 随 sequence 近似线性上升；短序列下 TPS 明显低于 KTransformers 成功档。

### DeepSpeed

原始汇总：[summary.md](../Qwen3.5-122B-A10B/test_log/20260802_133440_DEEPSPEED_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../Qwen3.5-122B-A10B/test_log/20260802_133440_DEEPSPEED_BF16_FULL_SWEEP/sweep_results.csv)

| Seq | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 18.94 | 2406.45 | - | - | - | - | - | FAILED |
| 2048 | 17.36 | 2398.63 | - | - | - | - | - | FAILED |
| 1024 | 17.24 | 2401.35 | - | - | - | - | - | FAILED |
| 512 | 15.75 | 2397.86 | - | - | - | - | - | FAILED |
| 256 | 15.74 | 2396.97 | - | - | - | - | - | FAILED |
| 128 | 17.28 | 2388.46 | - | - | - | - | - | FAILED |
| 64 | 15.77 | 2402.25 | - | - | - | - | - | FAILED |
| 32 | 15.87 | 2390.79 | - | - | - | - | - | FAILED |

八档均失败，无可用稳定 step / TPS。日志显示 ZeRO-3 已激活，`qwen35_122b_model_contract` 为 `contract=OK`（`logical_total=122111526912`），并打印 `***** Running training *****`，随后某一 local rank 被杀死：多数为 `Signal 9 (SIGKILL)` / exitcode -9，seq=512 观测到 `Signal 15 (SIGTERM)` / exitcode -15。单卡任务显存峰值仅约 15.7–18.9 GiB，未见 CUDA OOM；进程树 RSS 峰值约 2388–2406 GB，`host_used` 峰值约 2141–2162 GB，是三组中整机 used 最高的一档，与 ZeRO-3 CPU offload 的主机内存压力一致。

计时器在失败前已初始化为 `coarse_host_wall_no_cuda_sync`，但没有完成可汇总的稳定 step，因此表中时间列一律记为 `-`。

## 横向对比注意事项

- 本轮只有 DeepSpeed、KTransformers、MegaTrain 三类 exact-model server 数据；没有 APTMoE，也没有 consumer 剖面。
- 在可比的成功档中，KTransformers 在 seq≤2048 的 TPS 明显高于 MegaTrain（例如 seq=2048：397.78 vs 219.12）；本次 KTransformers 也跑通了 seq=4096（TPS 490.42）。但两者计时模式不同，不能当作严格同口径名次。
- 三组进程树 RSS 峰值都超过 1 TiB。MegaTrain 的进程树峰值约 5.6 TB，但其 `host_used`（约 1.69–1.70 TB）反而低于 KTransformers 成功档和 DeepSpeed 失败档，不能把进程树 RSS 直接读成整机物理占用。
- DeepSpeed 八档全部在训练启动后被系统信号杀死，当前配置下没有可用 throughput 数字；若要继续对比，需要降低 CPU offload 压力或扩充主机可用内存后再测。
- KTransformers seq=4096 已补齐；该点的单卡 GPU 峰值仅 27.68/47.99 GiB，真正接近上限的资源仍是主机内存。对 2164.13 GB 主机而言，160.19 GB 最低可用量可以跑通本次 15 step，但不应视为充足的长时稳态安全余量。
- 若要得到可发表的严格后端对比，仍应统一 CUDA 同步计时口径、确认每个后端确实更新全部参数，并补齐 consumer 剖面或在相同主机内存预算下重跑 DeepSpeed。
