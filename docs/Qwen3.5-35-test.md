# Qwen3.5-35B-A3B Consumer / Server 测试结果分析

本文分析 `Qwen3.5-35B-A3B/test_log` 中四个后端的 consumer 和 server 测试日志：

| 后端 | Consumer | Server |
|---|---|---|
| APTMoE | [summary](../Qwen3.5-35B-A3B/test_log/APTMOE_BF16_FULL_CONSUMER/summary.md) | [summary](../Qwen3.5-35B-A3B/test_log/APTMOE_BF16_FULL_SERVER/summary.md) |
| DeepSpeed | [summary](../Qwen3.5-35B-A3B/test_log/DS_BF16_FULL_CONSUMER/summary.md) | [summary](../Qwen3.5-35B-A3B/test_log/DS_BF16_FULL_SERVER/summary.md) |
| KTransformers | [summary](../Qwen3.5-35B-A3B/test_log/KT_BF16_FULL_CONSUMER/summary.md) | [summary](../Qwen3.5-35B-A3B/test_log/KT_BF16_FULL_SERVER/summary.md) |
| MegaTrain | [summary](../Qwen3.5-35B-A3B/test_log/MEGATRAIN_BF16_FULL_BOTH_NEW/summary.md) | [summary](../Qwen3.5-35B-A3B/test_log/MEGATRAIN_BF16_FULL_BOTH_NEW/summary.md) |

## 测量口径

- 八组测试均为 BF16、text-only、配置为 full finetuning、LoRA rank 0，per-device batch size 1，gradient accumulation steps 1。
- Consumer 使用 2 张 GPU、global batch size 2，测试 sequence length 16–2048；server 使用 8 张 GPU、global batch size 8，测试 sequence length 32–4096。
- 每档执行 15 step，前 5 step 为 warmup，表中时间均为后 10 个稳定 step 的均值。
- `TPS = global batch size × sequence length / 稳定 step 平均时间`，单位为 token/s。它按配置长度计算 token 数，不会自动扣除 padding、mask 或被跳过的 batch。
- CPU 内存是训练进程树的 RSS 求和峰值，单位为十进制 GB。多进程共享页可能被重复计入，因此它适合比较同一采集方法下的进程树占用，不等同于整机实际新增物理内存。
- GPU 0/1 是该测试任务进程在对应 consumer GPU 上的显存峰值，单位为 GiB；不包含其他任务和驱动基线占用。
- Server 表中的“单卡 GPU 峰值”取该档 8 张任务 GPU 的 `task_peak_gib` 最大值，不是 8 张卡的显存之和。
- Consumer 的 APTMoE 使用持久进程，从 sequence length 2048 依次测试到 16，因此后续档位的 CPU/GPU 峰值可能包含前面档位保留的模型、缓存或 allocator 状态。更新后的 KTransformers 和 MegaTrain 使用独立进程模式，每个 sequence 都重新启动训练进程；DeepSpeed 的现有产物也按 sequence 分别记录。
- Server 的 APTMoE 使用持久进程，从 sequence length 4096 测到 32；DeepSpeed、KTransformers 和 MegaTrain 均为逐 sequence 独立进程。
- APTMoE、DeepSpeed、KTransformers 的计时模式为 `coarse_host_wall_no_cuda_sync`，没有在每个阶段强制 CUDA 同步。MegaTrain 为 `megatrain_host_wall_with_backend_cuda_sync`：forward/backward 使用 CUDA event，optimizer 和 step total 使用 host wall time。不同计时模式的阶段时间不宜直接当作完全相同的测量量。
- forward、backward、optimizer 是各自计时边界的平均值；由于异步执行、同步点和阶段交叠，它们不保证严格相加等于 step time。
- 1 TiB 的判断阈值是 1099.51 GB。日志只采样并可视化内存，没有因超过阈值自动终止，也没有自动把超过阈值判成 OOM。
- 本文中的 MegaTrain 数据已更新为修复后目录 `MEGATRAIN_BF16_FULL_BOTH_NEW`：consumer 启用了保留 response token 的截断，sequence 16 和 32 均保留有效 label；server 丢弃不完整尾批次。Consumer 和 server 共 16 档的退出码均为 0，每档都有 10 个完整稳定 step，日志中不再出现 `No valid tokens in entire batch! Skipping...`。

## 测试身份与可比性

| 后端 | benchmark class | 权重来源 | 模型范围 | Consumer 状态 | Server 状态 |
|---|---|---|---|---|---|
| APTMoE | `deployment_proxy` | 确定性随机初始化 | Qwen3.5 component-isomorphic APTMoE proxy | 8 档完成 | 8 档完成（OK；显存峰值自动释放） |
| DeepSpeed | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5 text model 端到端全量微调 | 8 档完成 | 8 档完成 |
| KTransformers | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5 text model 端到端全量微调 | 8 档独立进程测试完成 | 8 档独立进程测试完成 |
| MegaTrain | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5 text model 端到端全量微调 | 8 档独立进程测试完成，短序列 label 修复已验证 | 8 档独立进程测试完成，无稳定 step 跳过 |

## Consumer（2 GPU / global batch size 2）

### APTMoE

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

### DeepSpeed

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

### KTransformers

更新数据：[summary.md](../Qwen3.5-35B-A3B/test_log/KT_BF16_FULL_CONSUMER/summary.md) · [sweep_results.csv](../Qwen3.5-35B-A3B/test_log/KT_BF16_FULL_CONSUMER/sweep_results.csv)

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

### MegaTrain

修复后汇总：[summary.md](../Qwen3.5-35B-A3B/test_log/MEGATRAIN_BF16_FULL_BOTH_NEW/summary.md) · [sweep_results.csv](../Qwen3.5-35B-A3B/test_log/MEGATRAIN_BF16_FULL_BOTH_NEW/sweep_results.csv)

| Seq | GPU 0 峰值 (GiB) | GPU 1 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2048 | 16.37 | 15.93 | 556.74 | 11.553 | 354.53 | 2.708 | 7.706 | 0.988 | OK |
| 1024 | 16.25 | 15.81 | 556.60 | 11.269 | 181.74 | 2.705 | 7.415 | 0.966 | OK |
| 512 | 15.98 | 15.54 | 556.57 | 11.009 | 93.02 | 2.693 | 7.214 | 0.957 | OK |
| 256 | 15.84 | 15.40 | 556.80 | 11.051 | 46.33 | 2.696 | 7.211 | 0.967 | OK |
| 128 | 15.78 | 15.34 | 553.36 | 11.064 | 23.14 | 2.701 | 7.206 | 0.995 | OK |
| 64 | 15.73 | 15.29 | 556.58 | 10.924 | 11.72 | 2.689 | 7.077 | 1.001 | OK |
| 32 | 15.71 | 15.27 | 553.30 | 10.822 | 5.91 | 2.691 | 6.960 | 1.002 | OK |
| 16 | 15.70 | 15.26 | 556.54 | 10.716 | 2.99 | 2.693 | 6.904 | 0.974 | OK |

八档均以 `OK_BACKEND_SYNC`、exit code 0 完成，每档包含 10 个有效稳定 step。修复后的 sequence 16 和 32 均执行了真实 forward/backward，不再出现零有效 label 或跳过 batch。每个 sequence 使用独立训练进程，因此本表显存峰值不包含前一档保留的 allocator 状态。

## Server（8 GPU / global batch size 8）

### APTMoE

原始汇总：[summary.md](../Qwen3.5-35B-A3B/test_log/APTMOE_BF16_FULL_SERVER/summary.md) · [sweep_results.csv](../Qwen3.5-35B-A3B/test_log/APTMOE_BF16_FULL_SERVER/sweep_results.csv)

| Seq | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 35.94 | 581.93 | 136.187 | 240.61 | 38.562 | 111.138 | 6.201 | OK |
| 2048 | 47.29 | 633.77 | 126.148 | 129.88 | 43.625 | 96.918 | 5.019 | OK |
| 1024 | 47.08 | 663.09 | 125.593 | 65.23 | 43.631 | 96.745 | 5.112 | OK |
| 512 | 31.26 | 679.00 | 105.301 | 38.90 | 37.038 | 82.864 | 4.601 | OK |
| 256 | 31.27 | 682.86 | 67.596 | 30.30 | 22.357 | 52.845 | 3.747 | OK |
| 128 | 31.35 | 682.23 | 42.335 | 24.19 | 14.631 | 31.844 | 3.192 | OK |
| 64 | 31.50 | 682.26 | 26.932 | 19.01 | 9.848 | 19.750 | 2.459 | OK |
| 32 | 31.52 | 682.91 | 16.208 | 15.79 | 6.371 | 11.502 | 1.681 | OK |

八档的训练退出码均为 0，稳定 step 完整，最终 GPU release 也得到确认，因此训练状态标记为 OK。持久进程运行期间观察到最长 sequence 的显存峰值被 allocator 自动清理；该现象保留为 `status=AUTO_RELEASED`、`Peak held=no`、`NOT_OOM_GPU_AUTO_RELEASED` 观测项，不再覆盖正常训练状态，也不属于训练 OOM。

### DeepSpeed

原始汇总：[summary.md](../Qwen3.5-35B-A3B/test_log/DS_BF16_FULL_SERVER/summary.md) · [sweep_results.csv](../Qwen3.5-35B-A3B/test_log/DS_BF16_FULL_SERVER/sweep_results.csv)

| Seq | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 21.88 | 810.31 | 16.714 | 1960.52 | 4.066 | 8.652 | 3.980 | OK |
| 2048 | 11.79 | 809.19 | 16.642 | 984.53 | 3.834 | 8.500 | 4.292 | OK |
| 1024 | 8.62 | 809.19 | 16.205 | 505.51 | 3.847 | 8.177 | 4.164 | OK |
| 512 | 7.72 | 809.25 | 16.428 | 249.34 | 3.832 | 8.213 | 4.365 | OK |
| 256 | 8.26 | 809.19 | 16.026 | 127.79 | 3.763 | 8.049 | 4.197 | OK |
| 128 | 7.25 | 809.24 | 15.959 | 64.17 | 3.795 | 8.103 | 4.046 | OK |
| 64 | 6.73 | 809.55 | 16.117 | 31.77 | 3.798 | 8.196 | 4.109 | OK |
| 32 | 6.85 | 811.69 | 15.799 | 16.20 | 3.748 | 8.004 | 4.033 | OK |

### KTransformers

原始汇总：[summary.md](../Qwen3.5-35B-A3B/test_log/KT_BF16_FULL_SERVER/summary.md) · [sweep_results.csv](../Qwen3.5-35B-A3B/test_log/KT_BF16_FULL_SERVER/sweep_results.csv)

| Seq | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 20.41 | 815.23 | 30.512 | 1073.95 | 7.803 | 20.460 | 2.107 | OK |
| 2048 | 15.11 | 732.72 | 17.717 | 924.76 | 4.613 | 10.856 | 2.114 | OK |
| 1024 | 11.71 | 691.51 | 11.361 | 721.03 | 2.910 | 6.173 | 2.149 | OK |
| 512 | 11.71 | 671.28 | 7.700 | 531.97 | 2.007 | 3.635 | 1.928 | OK |
| 256 | 12.46 | 660.72 | 6.244 | 327.97 | 1.572 | 2.624 | 1.920 | OK |
| 128 | 12.21 | 655.54 | 5.682 | 180.22 | 1.416 | 2.188 | 1.945 | OK |
| 64 | 12.20 | 653.18 | 5.232 | 97.86 | 1.313 | 1.792 | 1.994 | OK |
| 32 | 12.19 | 651.72 | 4.758 | 53.81 | 1.124 | 1.504 | 1.998 | OK |

八档均为独立训练进程，`persistent_profile_process=false`、`gpu_lifecycle_guard_enabled=false`。聚合 `summary.md` 顶部的持久进程说明是通用旧文案，与逐档 `run_config.json` 不符，本节以逐档配置和 CSV 为准。

### MegaTrain

修复后汇总：[summary.md](../Qwen3.5-35B-A3B/test_log/MEGATRAIN_BF16_FULL_BOTH_NEW/summary.md) · [sweep_results.csv](../Qwen3.5-35B-A3B/test_log/MEGATRAIN_BF16_FULL_BOTH_NEW/sweep_results.csv)

| Seq | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 18.07 | 1589.98 | 21.480 | 1525.51 | 2.707 | 12.028 | 2.368 | OK |
| 2048 | 16.37 | 1589.92 | 23.545 | 695.86 | 2.704 | 13.774 | 2.299 | OK |
| 1024 | 16.25 | 1586.53 | 24.314 | 336.93 | 2.703 | 14.717 | 2.299 | OK |
| 512 | 15.98 | 1589.94 | 23.871 | 171.59 | 2.702 | 14.342 | 2.080 | OK |
| 256 | 15.84 | 1589.98 | 24.385 | 83.99 | 2.701 | 14.547 | 2.354 | OK |
| 128 | 15.78 | 1589.94 | 23.511 | 43.55 | 2.702 | 13.046 | 2.473 | OK |
| 64 | 15.73 | 1589.93 | 20.557 | 24.91 | 2.697 | 11.229 | 2.279 | OK |
| 32 | 15.71 | 1589.70 | 22.152 | 11.56 | 2.698 | 12.707 | 2.243 | OK |

八档均以 `OK_BACKEND_SYNC`、exit code 0 完成，每档包含 10 个完整稳定 step，日志中没有 `No valid tokens in entire batch! Skipping...`。丢弃不完整尾批次后，TPS 不再受跳过 step 人为抬高。所有档位的进程树 RSS 求和峰值仍超过 1 TiB；这不等价于整机新增物理内存超过 1 TiB，也未被自动归类为 OOM。

## 横向对比注意事项

- 只有 DeepSpeed、KTransformers、MegaTrain 被日志声明为 exact-model；APTMoE 必须单列为 deployment proxy。
- Consumer 的 exact-model 八档有效数据中，更新后的 KTransformers TPS 最高，MegaTrain 次之；MegaTrain 的 sequence 16 和 32 已修复并纳入比较。DeepSpeed TPS 最低，不过逐卡 GPU 显存最低。
- Server 的修复后有效数据中，DeepSpeed 在 sequence 4096 和 2048 的 TPS 最高，KTransformers 在 1024 及以下最高。MegaTrain 已无稳定 step 跳过，但因计时模式不同，仍不能把各后端 TPS 当作完全同口径的严格名次。
- Server 的 MegaTrain 进程树 RSS 求和峰值约 1587–1590 GB，八档都超过 1 TiB；APTMoE、DeepSpeed 和 KTransformers 均未超过。APTMoE 训练状态为 OK，但跨 sequence 期间显存峰值被自动清理，因此各档显存峰值可作为该档观测值，不能解释为持续驻留到 profile 结束的显存量。
- MegaTrain 的短序列 label 和 server 尾批次问题已在本轮修复。如果要得到可发表的严格后端对比，仍应统一 CUDA 同步计时口径、验证每个后端确实更新全部参数，并统一使用逐 sequence 独立进程或通过验证的持久进程峰值保持方案。
