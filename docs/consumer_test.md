# Qwen3.5-35B-A3B Consumer 测试结果分析

本文分析以下四组 consumer 测试日志：

- [APTMOE_BF16_FULL_CONSUMER](../Qwen3.5-35B-A3B/test_log/APTMOE_BF16_FULL_CONSUMER/summary.md)
- [DS_BF16_FULL_CONSUMER](../Qwen3.5-35B-A3B/test_log/DS_BF16_FULL_CONSUMER/summary.md)
- [20260728_122231_KTRANSFORMERS_BF16_FULL_SWEEP](../Qwen3.5-35B-A3B/test_log/20260728_122231_KTRANSFORMERS_BF16_FULL_SWEEP/summary.md)
- [MEGATRAIN_BF16_FULL_CONSUMER](../Qwen3.5-35B-A3B/test_log/MEGATRAIN_BF16_FULL_CONSUMER/summary.md)

## 测量口径

- 四组测试均为 BF16、text-only、配置为 full finetuning、LoRA rank 0；使用 2 张 consumer GPU，global batch size 2，per-device batch size 1，gradient accumulation steps 1。
- 每档执行 15 step，前 5 step 为 warmup，表中时间均为后 10 个稳定 step 的均值。
- `TPS = global batch size × sequence length / 稳定 step 平均时间`，单位为 token/s。它按配置长度计算 token 数，不会自动扣除 padding、mask 或被跳过的 batch。
- CPU 内存是训练进程树的 RSS 求和峰值，单位为十进制 GB。多进程共享页可能被重复计入，因此它适合比较同一采集方法下的进程树占用，不等同于整机实际新增物理内存。
- GPU 0/1 是该测试任务进程在对应 consumer GPU 上的显存峰值，单位为 GiB；不包含其他任务和驱动基线占用。
- APTMoE 和 MegaTrain 使用持久进程，从 sequence length 2048 依次测试到 16，因此后续档位的 CPU/GPU 峰值可能包含前面档位保留的模型、缓存或 allocator 状态。更新后的 KTransformers 使用原始独立进程模式，从 16 测到 2048，每个 sequence 都重新启动训练进程；DeepSpeed 的现有产物也按 sequence 分别记录。
- APTMoE、DeepSpeed、KTransformers 的计时模式为 `coarse_host_wall_no_cuda_sync`，没有在每个阶段强制 CUDA 同步。MegaTrain 为 `megatrain_host_wall_with_backend_cuda_sync`：forward/backward 使用 CUDA event，optimizer 和 step total 使用 host wall time。不同计时模式的阶段时间不宜直接当作完全相同的测量量。
- forward、backward、optimizer 是各自计时边界的平均值；由于异步执行、同步点和阶段交叠，它们不保证严格相加等于 step time。
- 1 TiB 的判断阈值是 1099.51 GB。日志只采样并可视化内存，没有因超过阈值自动终止，也没有自动把超过阈值判成 OOM。

## 测试身份与可比性

| 后端 | benchmark class | 权重来源 | 模型范围 | 结果状态 |
|---|---|---|---|---|
| APTMoE | `deployment_proxy` | 确定性随机初始化 | Qwen3.5 component-isomorphic APTMoE proxy | 仅代表部署代理吞吐，不作 exact-model 或模型质量声明 |
| DeepSpeed | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5 text model 端到端全量微调 | 正常完成 |
| KTransformers | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5 text model 端到端全量微调 | 更新后的 8 档独立进程测试均正常完成 |
| MegaTrain | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5 text model 端到端全量微调 | 64–2048 可用；32 部分 batch 异常；16 无有效训练 batch |

## APTMoE

原始汇总：[summary.md](../Qwen3.5-35B-A3B/test_log/APTMOE_BF16_FULL_CONSUMER/summary.md) · [sweep_results.csv](../Qwen3.5-35B-A3B/test_log/APTMOE_BF16_FULL_CONSUMER/sweep_results.csv)

| Seq | GPU 0 峰值 (GiB) | GPU 1 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 13.57 | 15.96 | 303.10 | 94.932 | 43.15 | 14.918 | 72.675 | 9.029 |
| 1024 | 14.53 | 20.24 | 348.24 | 65.103 | 31.46 | 11.559 | 45.902 | 8.962 |
| 512 | 15.41 | 22.86 | 343.61 | 45.075 | 22.72 | 8.343 | 29.324 | 8.033 |
| 256 | 15.41 | 22.86 | 343.01 | 33.345 | 15.35 | 6.368 | 20.223 | 7.239 |
| 128 | 15.91 | 24.69 | 343.55 | 26.621 | 9.62 | 5.171 | 15.517 | 6.535 |
| 64 | 16.48 | 26.19 | 343.81 | 18.973 | 6.75 | 3.572 | 11.084 | 4.780 |
| 32 | 16.91 | 27.42 | 343.81 | 13.938 | 4.59 | 2.836 | 7.899 | 3.737 |
| 16 | 17.30 | 28.79 | 343.81 | 9.853 | 3.25 | 1.974 | 5.533 | 2.663 |


## DeepSpeed

原始汇总：[summary.md](../Qwen3.5-35B-A3B/test_log/DS_BF16_FULL_CONSUMER/summary.md) · [sweep_results.csv](../Qwen3.5-35B-A3B/test_log/DS_BF16_FULL_CONSUMER/sweep_results.csv)

| Seq | GPU 0 峰值 (GiB) | GPU 1 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 14.04 | 14.04 | 822.75 | 17.966 | 227.99 | 3.548 | 11.395 | 3.009 |
| 1024 | 11.11 | 11.11 | 822.76 | 17.823 | 114.91 | 3.530 | 11.257 | 3.022 |
| 512 | 10.16 | 10.16 | 822.75 | 17.670 | 57.95 | 3.525 | 11.108 | 3.024 |
| 256 | 10.20 | 10.20 | 822.75 | 17.613 | 29.07 | 3.508 | 11.035 | 3.057 |
| 128 | 10.19 | 10.19 | 822.76 | 17.515 | 14.62 | 3.530 | 10.959 | 3.013 |
| 64 | 10.17 | 10.17 | 822.75 | 17.478 | 7.32 | 3.504 | 10.902 | 3.059 |
| 32 | 10.17 | 10.17 | 822.69 | 17.456 | 3.67 | 3.540 | 10.842 | 3.062 |
| 16 | 10.16 | 10.16 | 822.69 | 17.362 | 1.84 | 3.487 | 10.817 | 3.046 |


## KTransformers

更新数据：[summary.md](../Qwen3.5-35B-A3B/test_log/20260728_122231_KTRANSFORMERS_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../Qwen3.5-35B-A3B/test_log/20260728_122231_KTRANSFORMERS_BF16_FULL_SWEEP/sweep_results.csv)

| Seq | GPU 0 峰值 (GiB) | GPU 1 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 21.04 | 21.08 | 617.57 | 7.422 | 551.88 | 1.825 | 3.596 | 1.898 |
| 1024 | 19.38 | 19.38 | 607.18 | 5.969 | 343.11 | 1.405 | 2.561 | 1.900 |
| 512 | 19.00 | 19.00 | 602.08 | 5.431 | 188.56 | 1.238 | 2.125 | 1.965 |
| 256 | 19.04 | 19.04 | 599.44 | 5.088 | 100.63 | 1.137 | 1.886 | 1.959 |
| 128 | 18.60 | 18.60 | 598.31 | 4.768 | 53.69 | 1.064 | 1.670 | 1.929 |
| 64 | 18.88 | 18.88 | 597.53 | 4.563 | 28.05 | 1.037 | 1.454 | 1.969 |
| 32 | 18.84 | 18.84 | 597.11 | 4.260 | 15.02 | 0.969 | 1.258 | 1.931 |
| 16 | 18.81 | 18.81 | 596.82 | 4.157 | 7.70 | 0.976 | 1.120 | 1.950 |

本次 KTransformers 测试的 8 个 `exit_code.txt` 均为 0，聚合状态均为 `OK`，日志中没有 CUDA OOM。各 sequence 使用独立训练进程，`persistent_profile_process=false`、`gpu_lifecycle_guard_enabled=false`，所以没有 `gpu_peak_hold.json` 或 `gpu_lifecycle.json`；对应列为空是当前设计，而不是驻留检查失败。

该 run 的聚合 `summary.md` 顶部仍含有“每个 profile 只启动一次持久训练进程”的通用旧说明，但这与 8 份 `run_config.json` 及逐 sequence 独立的 `monitor.csv`、`train.log` 产物不符。本节以这些逐档原始产物为准。

取消跨 sequence 显存保护后，CPU RSS 峰值从旧测试的累计 617.63–1971.19 GB 变为稳定的 596.82–617.57 GB，所有档位均低于 1 TiB。两卡任务显存也由旧测试的累计最高约 47.08 GiB 变为每卡约 18.60–21.08 GiB，GPU 0/1 基本均衡。


## MegaTrain

原始汇总：[summary.md](../Qwen3.5-35B-A3B/test_log/MEGATRAIN_BF16_FULL_CONSUMER/summary.md) · [sweep_results.csv](../Qwen3.5-35B-A3B/test_log/MEGATRAIN_BF16_FULL_CONSUMER/sweep_results.csv) · [train.log](../Qwen3.5-35B-A3B/test_log/MEGATRAIN_BF16_FULL_CONSUMER/consumer_2gpu_batch2/train.log)

| Seq | GPU 0 峰值 (GiB) | GPU 1 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2048 | 16.37 | 15.93 | 557.16 | 11.580 | 353.71 | 2.704 | 7.668 | 1.017 |
| 1024 | 16.37 | 15.93 | 557.16 | 11.284 | 181.49 | 2.702 | 7.409 | 0.982 |
| 512 | 16.57 | 16.13 | 557.18 | 11.226 | 91.22 | 2.701 | 7.329 | 0.986 |
| 256 | 16.58 | 16.14 | 557.18 | 11.085 | 46.19 | 2.701 | 7.192 | 0.959 |
| 128 | 16.58 | 16.14 | 557.18 | 10.964 | 23.35 | 2.700 | 7.118 | 0.959 |
| 64 | 16.58 | 16.14 | 557.18 | 10.924 | 11.72 | 2.700 | 7.053 | 1.017 |
| 32 | 16.58 | 16.14 | 557.18 | 7.760 | 8.25 | 1.080 | 2.783 | 0.965 |
| 16 | 16.58 | 16.14 | 557.19 | 0.920 | 34.78 | 0.000 | 0.000 | 0.918 |


## 横向对比注意事项

- 只有 DeepSpeed、KTransformers、MegaTrain 被日志声明为 exact-model；APTMoE 必须单列为 deployment proxy。
- exact-model 的正常区间内，更新后的 KTransformers 原始 TPS 最高，CPU RSS 和逐卡显存也不再跨 sequence 累计；MegaTrain 次之，但短序列数据集截断问题使 32 和 16 两档失效；DeepSpeed TPS 最低，不过逐卡 GPU 显存最低。
- 如果要得到可发表的严格后端对比，下一轮应统一 CUDA 同步计时口径、保证每个 sequence 的 label 中都有有效 token、验证每个后端确实更新全部参数，并明确区分“持久进程 sweep 峰值”和“KTransformers 独立进程单档峰值”。
