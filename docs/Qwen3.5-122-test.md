# Qwen3.5-122B-A10B Server 测试结果分析

本文分析 `Qwen3.5-122B-A10B/test_log` 中三个后端的 **server** 全量扫描日志（本轮未跑 consumer，也未包含 APTMoE）：

| 后端 | Server |
|---|---|
| KTransformers | [summary](../Qwen3.5-122B-A10B/test_log/20260801_175712_KTRANSFORMERS_BF16_FULL_SWEEP/summary.md) |
| MegaTrain | [summary](../Qwen3.5-122B-A10B/test_log/20260802_094707_MEGATRAIN_BF16_FULL_SWEEP/summary.md) |
| DeepSpeed | [summary](../Qwen3.5-122B-A10B/test_log/20260802_133440_DEEPSPEED_BF16_FULL_SWEEP/summary.md) |

## 测量口径

- 三组测试均为 BF16、text-only、配置为 full finetuning、LoRA rank 0，per-device batch size 1，gradient accumulation steps 1。
- Server 使用 8 张 GPU、global batch size 8，测试 sequence length 32–4096；每个 sequence 均为独立训练进程（`persistent_profile_process=false`）。
- 每档配置 15 step，前 5 step 为 warmup；成功档的时间均为后 10 个稳定 step 的均值。
- `TPS = global batch size × sequence length / 稳定 step 平均时间`，单位为 token/s。它按配置长度计算 token 数，不会自动扣除 padding、mask 或被跳过的 batch。
- 表中 CPU 内存是训练进程树的 RSS 求和峰值，单位为十进制 GB。多进程共享页可能被重复计入，因此它适合比较同一采集方法下的进程树占用，不等同于整机实际新增物理内存。同目录 `memory_summary.json` 另给出 `host_used_peak_gb_decimal`（整机 used 采样峰值），本文在失败分析中引用该值作对照。
- Server 表中的“单卡 GPU 峰值”取该档 8 张任务 GPU 的 `task_peak_gib` 最大值，不是 8 张卡的显存之和；单位为 GiB，不包含其他任务和驱动基线占用。
- DeepSpeed、KTransformers 的计时模式为 `coarse_host_wall_no_cuda_sync`，没有在每个阶段强制 CUDA 同步。MegaTrain 为 `megatrain_host_wall_with_backend_cuda_sync`：forward/backward 使用 CUDA event，optimizer 和 step total 使用 host wall time。不同计时模式的阶段时间不宜直接当作完全相同的测量量。
- forward、backward、optimizer 是各自计时边界的平均值；由于异步执行、同步点和阶段交叠，它们不保证严格相加等于 step time。
- 1 TiB 的判断阈值是 1099.51 GB。日志只采样并可视化内存，没有因超过阈值自动终止，也没有自动把超过阈值判成 OOM。
- 本轮 KTransformers 已在 shared-expert 参数契约修复之后运行（见 [Qwen3.5-KTransformers-shared-expert-fix.md](./Qwen3.5-KTransformers-shared-expert-fix.md)）；成功档日志中 `logical_total=122111526912`、`contract=OK`。

## 测试身份与可比性

| 后端 | benchmark class | 权重来源 | 模型范围 | Server 状态 |
|---|---|---|---|---|
| KTransformers | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5-122B text model 端到端全量微调 | 7/8 独立进程成功；seq=4096 失败 |
| MegaTrain | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5-122B text model 端到端全量微调 | 8/8 独立进程成功 |
| DeepSpeed | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5-122B text model 端到端全量微调 | 8/8 均在训练启动后被杀死，无稳定 timing |

## Server（8 GPU / global batch size 8）

### KTransformers

原始汇总：[summary.md](../Qwen3.5-122B-A10B/test_log/20260801_175712_KTRANSFORMERS_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../Qwen3.5-122B-A10B/test_log/20260801_175712_KTRANSFORMERS_BF16_FULL_SWEEP/sweep_results.csv)

| Seq | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 24.78 | 2458.52 | - | - | - | - | - | FAILED |
| 2048 | 20.25 | 2307.02 | 40.950 | 400.09 | 9.938 | 23.400 | 7.403 | SUCCESS |
| 1024 | 19.22 | 2230.22 | 27.775 | 294.94 | 6.688 | 13.430 | 7.432 | SUCCESS |
| 512 | 16.02 | 2192.52 | 20.958 | 195.43 | 5.135 | 9.063 | 6.556 | SUCCESS |
| 256 | 16.34 | 2173.45 | 18.804 | 108.91 | 4.335 | 7.262 | 7.005 | SUCCESS |
| 128 | 16.21 | 2163.98 | 17.060 | 60.02 | 3.730 | 6.433 | 6.699 | SUCCESS |
| 64 | 16.20 | 2159.36 | 16.719 | 30.62 | 4.012 | 5.579 | 6.926 | SUCCESS |
| 32 | 16.27 | 2158.10 | 15.505 | 16.51 | 3.227 | 4.651 | 7.425 | SUCCESS |

seq=32–2048 退出码均为 0，每档 10 个稳定 step，`contract=OK`（`logical_total=122111526912`，`kt_wrappers=48`）。进程树 RSS 峰值约 2158–2307 GB，均超过 1 TiB；对应 `host_used` 约 1705–1853 GB。

seq=4096 在进入 `***** Running training *****`、注入 fused expert LoRA 参数之后失败：rank 0 收到 `Signal 9 (SIGKILL)`（`ChildFailedError`，exitcode -9）。单卡任务显存峰值约 24.78 GiB，未见 CUDA OOM 文本；进程树 RSS 峰值约 2458.52 GB，`host_used` 峰值约 1963.72 GB。更符合主机内存压力下的 OOM killer，而不是显存打满。

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
- 在可比的成功档中，KTransformers 在 seq≤2048 的 TPS 明显高于 MegaTrain（例如 seq=2048：400.09 vs 219.12）；MegaTrain 是唯一跑通 seq=4096 的后端（TPS 428.81），但计时模式不同，不能当作严格同口径名次。
- 三组进程树 RSS 峰值都超过 1 TiB。MegaTrain 的进程树峰值约 5.6 TB，但其 `host_used`（约 1.69–1.70 TB）反而低于 KTransformers 成功档和 DeepSpeed 失败档，不能把进程树 RSS 直接读成整机物理占用。
- DeepSpeed 八档全部在训练启动后被系统信号杀死，当前配置下没有可用 throughput 数字；若要继续对比，需要降低 CPU offload 压力或扩充主机可用内存后再测。
- KTransformers seq=4096 同样以 SIGKILL 失败，GPU 峰值未打满；若要补齐该点，优先从主机内存与激活占用入手，而不是单纯增大 GPU 显存。
- 若要得到可发表的严格后端对比，仍应统一 CUDA 同步计时口径、确认每个后端确实更新全部参数，并补齐 consumer 剖面或在相同主机内存预算下重跑 DeepSpeed / KT@4096。
