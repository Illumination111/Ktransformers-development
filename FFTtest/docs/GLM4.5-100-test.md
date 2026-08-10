# GLM-4.5-Air 100 样本全量微调测试结果分析

本文分析 `GLM-4.5-Air/test_log` 中 KTransformers、DeepSpeed 与 MegaTrain 的 Server 测试日志：


| 后端                                 | Server 汇总                                                                               |
| ---------------------------------- | --------------------------------------------------------------------------------------- |
| KTransformers                      | [32–2048 当前 sweep](../GLM-4.5-Air/test_log/20260730_185400_KTRANSFORMERS_BF16_FULL_SWEEP/summary.md) · [4096 成功复测](../GLM-4.5-Air/test_log/KT_BF16_FULL_4k/summary.md) |
| DeepSpeed ZeRO-3（仅优化器 CPU offload） | [summary](../GLM-4.5-Air/test_log/20260730_121704_DEEPSPEED_BF16_FULL_SWEEP/summary.md) |
| MegaTrain（CPU master / layer streaming） | [summary](../GLM-4.5-Air/test_log/MEGATRAIN_BF16_FULL_CONSUMER/summary.md)              |


## 测量口径

- 所有测试均为 BF16、text-only、full finetuning，per-device batch size 1，gradient accumulation steps 1，使用数据集 `fft_real_100`。
- Server 使用 8 张 GPU、global batch size 8，测试 sequence length 32–4096。
- DeepSpeed 使用 ZeRO-3，优化器 offload 到 CPU，参数不 offload；启用 BF16 master weights/gradients 和 BF16 optimizer states。
- 配置术语需要特别区分：本轮不是“将优化器 offload 到 GPU”，而是保留 `offload_optimizer.device=cpu` 并删除 `offload_param`。因此优化器状态和计算位于 CPU，参数分片则常驻 GPU。
- 每档计划执行 15 step，前 5 step 为 warmup。成功档位的表中时间均为后 10 个稳定 step 的均值。KTransformers Server sequence 4096 的旧测试受其他 GPU 进程干扰而 CUDA OOM，当前 full sweep 又因 NUMA node 0 内存策略约束下的主机 OOM 导致 CPU owner rank 0 被内核 `SIGKILL`；独立清洁环境复测完整执行，因此主结果表采用该成功复测。三个后端的 sequence 4096 均有有效稳定结果。
- `TPS = global batch size × sequence length / 稳定 step 平均时间`，单位为 token/s。它按配置长度计算 token 数，不会自动扣除 padding、mask 或被跳过的 batch。
- CPU 内存是训练进程树的 RSS 求和峰值，单位为十进制 GB。多进程共享页可能被重复计入，因此适合比较同一采集方法下的进程树占用，不等同于整机实际新增物理内存。
- DeepSpeed 与 MegaTrain 表额外列出 `host_used_peak_gb_decimal`，它是监控期间的整机实际已用内存峰值；判断约 2 TB 主机内存是否耗尽，应优先使用该值，而不是进程树 RSS 求和。
- Server 的“单卡 GPU 峰值”取该档 8 张任务 GPU 的 `task_peak_gib` 最大值，不是 8 张卡的显存之和，也不是整卡所有进程的显存峰值。
- 每个 sequence 均使用独立训练进程，`persistent_profile_process=false`，显存峰值不包含前一档保留的模型、缓存或 allocator 状态。
- KTransformers 与 DeepSpeed 的计时模式为 `coarse_host_wall_no_cuda_sync`，没有在每个阶段强制 CUDA 同步；MegaTrain 使用 `megatrain_host_wall_with_backend_cuda_sync`，保留后端执行所需的 CUDA event 与同步。forward、backward、optimizer 是各自计时边界的平均值，不保证严格相加等于 step time；MegaTrain 的阶段时间尤其不应与另外两种无强制同步的计时边界直接等同。
- 1 TiB 的判断阈值为 1099.51 GB。日志只采样并可视化内存，没有因超过阈值自动终止，也没有自动把超过阈值判成 OOM。



## 测试身份与完成状态


| 后端            | benchmark class             | 权重来源           | 模型范围                           | Server 状态                    |
| ------------- | --------------------------- | -------------- | ------------------------------ | ---------------------------- |
| KTransformers | `exact_model_full_finetune` | 预训练 checkpoint | GLM-4.5-Air text model 端到端全量微调 | 8 档全部完成；4096 来自独立复测          |
| DeepSpeed     | `exact_model_full_finetune` | 预训练 checkpoint | GLM-4.5-Air text model 端到端全量微调 | 8 档全部完成                      |
| MegaTrain     | `exact_model_full_finetune` | 预训练 checkpoint | GLM-4.5-Air text model 端到端全量微调 | 8 档全部完成                      |


各轮 `dataset_validation.json` 均为 `OK`：数据集包含 100 条样本，token 长度为 7053–7284，覆盖要求的最大测试长度。

## Ktransformers Server（8 GPU / global batch size 8）

Sequence 32–2048 当前 sweep：[summary.md](../GLM-4.5-Air/test_log/20260730_185400_KTRANSFORMERS_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../GLM-4.5-Air/test_log/20260730_185400_KTRANSFORMERS_BF16_FULL_SWEEP/sweep_results.csv)

Sequence 4096 独立复测：[summary.md](../GLM-4.5-Air/test_log/KT_BF16_FULL_4k/summary.md) · [sweep_results.csv](../GLM-4.5-Air/test_log/KT_BF16_FULL_4k/sweep_results.csv)


| Seq  | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | Step (s) | TPS    | Forward (s) | Backward (s) | Optimizer (s) |
| ---- | --------------- | --------------- | -------- | ------ | ----------- | ------------ | ------------- |
| 4096 | 27.38           | 2224.88         | 85.351   | 383.92 | 20.795      | 58.428       | 5.859         |
| 2048 | 27.38           | 2030.56         | 48.589   | 337.20 | 11.777      | 30.179       | 6.413         |
| 1024 | 27.38           | 1933.08         | 30.599   | 267.72 | 7.396       | 16.552       | 6.451         |
| 512  | 27.38           | 1884.49         | 21.623   | 189.43 | 5.160       | 10.095       | 6.175         |
| 256  | 27.38           | 1860.14         | 17.591   | 116.42 | 4.068       | 7.338        | 5.992         |
| 128  | 27.38           | 1848.12         | 16.269   | 62.94  | 3.591       | 6.494        | 5.991         |
| 64   | 27.38           | 1841.85         | 15.326   | 33.41  | 3.300       | 5.735        | 6.098         |
| 32   | 27.38           | 1838.60         | 11.714   | 21.85  | 2.770       | 2.988        | 5.764         |


当前 full sweep 中 sequence 32–2048 的退出码均为 0，每档都有 10 个完整稳定 step。其 sequence 4096 在首个有效 step 计时写出前因 NUMA node 0 局部 OOM 导致 rank 0 被内核 `SIGKILL`，所以不用于性能汇总。独立的 sequence 4096 复测退出码为 0，模型契约确认 `logical_trainable=logical_total=106852245504`，15 个 step 全部执行，表中使用后 10 个稳定 step 的均值。该档稳定 step 为 85.351 秒，按每 step 32768 个配置 token 计算得到 383.92 TPS。

首次 sequence 4096 测试的多个 rank 曾在 FSDP 申请 2.31 GiB 时发生 CUDA OOM。当时整卡峰值最高为 47.97 GiB，异常日志显示 GPU 1–6 上另一个 PID 499802 各占用约 16.73 GiB，本测试进程约占 29.62 GiB。清洁环境复测中，8 张卡的任务峰值均为 27.38 GiB、整卡峰值均为 28.00 GiB，并完整完成训练，证明首次失败来自同卡外部进程造成的可用显存不足，而不是 KTransformers 在独占约 48 GiB GPU 时无法运行 sequence 4096。

当前 full sweep 的 sequence 4096 是另一类失败。内核日志在 `2026-07-30 12:24:47 UTC` 明确记录 `numa_0_t_31 invoked oom-killer`、`constraint=CONSTRAINT_MEMORY_POLICY`、`nodemask=0`，随后杀死 CPU owner rank 0（PID 3308779）。当时 node 0 Normal 区只剩约 1.76 GB，而整机仍有约 395 GB 空闲内存，说明空闲页主要位于 node 1，但 KTransformers CPU worker 使用严格 node 0 内存绑定，不能回退到 node 1；这不是整机 2 TB 物理内存全部耗尽。调用栈停在 `wp_page_copy`，表明直接触发点是写时复制。该配置每个 rank 启动 2 个 persistent DataLoader worker；内核任务表中恰有两个 `pt_data_worker` 各映射约 373 GiB RSS，极可能是从 CPU owner rank fork 后继承了大模型映射并放大 COW 压力。终端中的 `ChildFailedError` 和其余 rank 的 `SIGTERM` 均是 rank 0 被 OOM killer 杀死后的连带结果。

Server 各档的任务进程树 RSS 求和峰值为 1838.60–2224.88 GB，全部超过 1 TiB。Sequence 4096 复测的整机实际内存峰值为 1847.62 GB，低于监控器报告的 2164.13 GB 总内存；2224.88 GB 的进程树 RSS 求和包含多 rank 共享页重复统计，不能解释为真实物理内存占用。当前 32–2048 sweep 与 4096 成功复测的任务显存峰值均为 27.38 GiB。当前脚本设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，而旧 sweep 未启用该 allocator 配置，因此不再使用旧 sweep 中 32.78–32.83 GiB 的显存峰值作为主结果。

## DeepSpeed Server（8 GPU / global batch size 8）

原始汇总：[summary.md](../GLM-4.5-Air/test_log/20260730_121704_DEEPSPEED_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../GLM-4.5-Air/test_log/20260730_121704_DEEPSPEED_BF16_FULL_SWEEP/sweep_results.csv)


| Seq  | 单卡任务 GPU 峰值 (GiB) | 整机实际内存峰值 (GB) | CPU RSS 求和峰值 (GB) | Step (s) | TPS    | Forward (s) | Backward (s) | Optimizer (s) |
| ---- | ----------------- | ------------- | ----------------- | -------- | ------ | ----------- | ------------ | ------------- |
| 4096 | 43.29             | 1655.05       | 3216.07           | 52.864   | 619.85 | 11.642      | 27.808       | 13.393        |
| 2048 | 37.72             | 1653.59       | 3216.41           | 52.004   | 315.06 | 11.272      | 26.377       | 14.337        |
| 1024 | 37.72             | 1652.20       | 3216.39           | 49.420   | 165.76 | 11.111      | 25.836       | 12.456        |
| 512  | 37.72             | 1652.96       | 3216.41           | 48.482   | 84.48  | 11.057      | 25.312       | 12.096        |
| 256  | 37.72             | 1659.17       | 3216.43           | 50.499   | 40.56  | 11.053      | 25.780       | 13.649        |
| 128  | 37.77             | 1652.99       | 3216.16           | 49.803   | 20.56  | 10.981      | 25.463       | 13.343        |
| 64   | 37.75             | 1657.27       | 3216.47           | 51.611   | 9.92   | 10.971      | 25.258       | 15.364        |
| 32   | 37.73             | 1663.65       | 3217.98           | 51.272   | 4.99   | 11.009      | 24.825       | 15.418        |


八档退出码均为 0，每档都有 10 个完整稳定 step。完整 Server sweep 从数据校验开始到 sequence 4096 写出退出码约耗时 2 小时 30 分钟。

本轮成功与 offload 布局直接相关。此前参数和优化器同时 CPU offload 的 DeepSpeed 对照运行，在所有 sequence length 上都会因主机 RAM 耗尽而 OOM，无法得到有效训练结果。本轮只保留优化器 CPU offload，并将参数分片从主机内存移回 8 张 GPU，降低了主机物理内存压力，因而 sequence 32–4096 均能完成。换言之，成功的关键是**取消参数 CPU offload**，不是将优化器移到 GPU。

Sequence 32–2048 的单卡任务显存峰值稳定在 37.72–37.77 GiB；sequence 4096 增至 43.29 GiB，整卡峰值为 44.06 GiB，仍在约 48 GiB 容量内。整机实际内存峰值为 1652.20–1663.65 GB，未耗尽监控器报告的 2164.13 GB 主机内存。进程树 RSS 求和约 3216 GB 是共享页在多 rank 间重复计数的结果，不能解释为真实物理内存用量。

DeepSpeed 的稳定 step 时间在 48.48–52.86 秒之间，随 sequence 增长变化较小；因此按配置 token 数计算的 TPS 基本随 sequence 增长。CPU optimizer 阶段约 12.10–15.42 秒，是小 sequence 下吞吐较低的重要固定开销。

## MegaTrain Server（8 GPU / global batch size 8）

原始汇总：[summary.md](../GLM-4.5-Air/test_log/MEGATRAIN_BF16_FULL_CONSUMER/summary.md) · [sweep_results.csv](../GLM-4.5-Air/test_log/MEGATRAIN_BF16_FULL_CONSUMER/sweep_results.csv)

结果目录名称虽然包含 `CONSUMER`，但目录内的 `summary.md`、各档 `run_config.json` 和子目录均明确记录为 `profile=server`、8 GPU、global batch size 8。因此本报告按测试产物的权威元数据将其归类为 MegaTrain Server，而不是 2 GPU Consumer。

| Seq  | 单卡任务 GPU 峰值 (GiB) | 整机实际内存峰值 (GB) | CPU RSS 求和峰值 (GB) | Step (s) | TPS    | Forward (s) | Backward (s) | Optimizer (s) |
| ---- | ----------------- | ------------- | ----------------- | -------- | ------ | ----------- | ------------ | ------------- |
| 4096 | 34.73             | 1596.78       | 4975.94           | 64.440   | 508.51 | 8.489       | 39.271       | 6.098         |
| 2048 | 34.62             | 1596.84       | 4975.76           | 63.192   | 259.28 | 8.466       | 38.222       | 6.283         |
| 1024 | 34.63             | 1595.41       | 4975.89           | 61.173   | 133.91 | 8.465       | 36.568       | 5.996         |
| 512  | 30.81             | 1594.05       | 4975.83           | 68.576   | 59.73  | 8.461       | 41.707       | 7.103         |
| 256  | 30.71             | 1596.24       | 4979.08           | 68.129   | 30.06  | 8.462       | 42.284       | 5.823         |
| 128  | 30.66             | 1592.60       | 4975.81           | 73.122   | 14.00  | 8.459       | 43.969       | 7.123         |
| 64   | 30.63             | 1597.97       | 4975.85           | 66.694   | 7.68   | 8.461       | 41.103       | 5.900         |
| 32   | 30.60             | 1597.34       | 4975.79           | 68.233   | 3.75   | 8.454       | 41.610       | 6.900         |

八档退出码均为 0，每档都有 10 个完整稳定 step；sequence 4096 的日志也记录了模型契约通过和 `TRAINING COMPLETE`。稳定 step 时间为 61.17–73.12 秒，forward 基本稳定在 8.45–8.49 秒，主要波动来自 backward 和未单列的后端同步、调度及数据搬运开销。TPS 从 sequence 32 的 3.75 增至 sequence 4096 的 508.51。

单卡任务显存峰值在 sequence 32–512 为 30.60–30.81 GiB，在 sequence 1024–4096 增至 34.62–34.73 GiB；sequence 4096 的整卡峰值为 35.94 GiB。整机实际内存峰值为 1592.60–1597.97 GB，没有耗尽监控器报告的约 2164 GB 主机内存。约 4976 GB 的进程树 RSS 求和来自多 worker 对共享页的重复统计，不能解释为真实物理内存占用。

## 结果解读与注意事项

- KTransformers Server 具有 sequence 32–4096 全部 8 个档位的有效结果：32–2048 采用当前 full sweep，4096 采用清洁环境独立成功复测。当前 full sweep 的 4096 因严格 NUMA node 0 内存绑定触发局部 OOM，CPU owner rank 0 被内核 `SIGKILL`，故不用于性能汇总；成功复测的 TPS 为 383.92，单卡任务显存峰值为 27.38 GiB。
- DeepSpeed 在仅将优化器 offload 到 CPU、参数分片保留在 GPU 的配置下完成全部 8 个 Server 档位，包括 sequence 4096；此前参数和优化器同时 CPU offload 时，各 sequence 均会因主机 RAM 耗尽而 OOM。
- MegaTrain 完成 sequence 32–4096 全部 8 个 Server 档位；sequence 4096 的 TPS 为 508.51，单卡任务显存峰值为 34.73 GiB。
- KTransformers Server 的 TPS 从 sequence 32 的 21.85 增至 sequence 4096 的 383.92。
- 三个后端在 sequence 4096 上均有有效结果：DeepSpeed、MegaTrain 和 KTransformers 的 TPS 分别为 619.85、508.51 和 383.92。KTransformers/DeepSpeed TPS 比值从 sequence 32 的 4.38 倍逐步下降到 sequence 2048 的 1.07 倍，并在 sequence 4096 降至 0.62，即 DeepSpeed 此时约为 KTransformers 的 1.61 倍。
- DeepSpeed Server 的整机实际内存峰值约 1.66 TB，说明当前“仅优化器 CPU offload”配置没有耗尽约 2 TB 主机内存；sequence 4096 的单卡任务显存峰值为 43.29 GiB，是该配置更接近的容量边界。
- MegaTrain Server 的整机实际内存峰值约 1.60 TB、sequence 4096 单卡任务显存峰值为 34.73 GiB，均低于本轮 DeepSpeed 对应峰值；但 MegaTrain sequence 4096 的 TPS 也比 DeepSpeed 低约 18%。
- KTransformers、DeepSpeed 和 MegaTrain Server 所有档位的进程树 RSS 求和峰值均超过 1 TiB。该结果没有触发自动终止，且 RSS 求和可能重复统计共享页，不应直接解释为同等规模的整机新增物理内存。
- 三个后端使用相同模型、数据、GPU 数量、batch 和 step 数；KTransformers 与 DeepSpeed 使用无强制 CUDA 同步的 host-wall 计时，MegaTrain 保留后端所需同步。端到端 TPS 可作为本轮实测参考，但阶段时间边界和内部专家执行、内存布局不同，结论不应外推到其他模型或配置。
