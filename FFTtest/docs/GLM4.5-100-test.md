# GLM-4.5-Air 100 样本全量微调测试结果分析

本文分析 `GLM-4.5-Air/test_log` 中 KTransformers、DeepSpeed、MegaTrain、PyTorch TORCH 与 APTMoE deployment proxy 的 Server 测试日志：


| 后端                                 | Server 汇总                                                                               |
| ---------------------------------- | --------------------------------------------------------------------------------------- |
| KTransformers                      | [32–2048 当前 sweep](../GLM-4.5-Air/test_log/20260730_185400_KTRANSFORMERS_BF16_FULL_SWEEP/summary.md) · [4096 成功复测](../GLM-4.5-Air/test_log/KT_BF16_FULL_4k/summary.md) |
| DeepSpeed ZeRO-3（仅优化器 CPU offload） | [summary](../GLM-4.5-Air/test_log/20260730_121704_DEEPSPEED_BF16_FULL_SWEEP/summary.md) |
| MegaTrain（CPU master / layer streaming） | [summary](../GLM-4.5-Air/test_log/MEGATRAIN_BF16_FULL_CONSUMER/summary.md)              |
| PyTorch TORCH                      | [32–64](../GLM-4.5-Air/test_log/20260914_115933_PYTORCH_TORCH_BF16_FULL_SWEEP/summary.md) · [128–512](../GLM-4.5-Air/test_log/20260914_141236_PYTORCH_TORCH_BF16_FULL_SWEEP/summary.md) · [1024](../GLM-4.5-Air/test_log/20260914_194233_PYTORCH_TORCH_BF16_FULL_SWEEP/summary.md) |
| APTMoE deployment proxy            | [32、64、256、512、1024](../GLM-4.5-Air/test_log/20260915_060758_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY/summary.md) |


## 测量口径

- 所有测试均为 BF16、text-only、full finetuning，per-device batch size 1，gradient accumulation steps 1，使用数据集 `fft_real_100`。其中 KTransformers、DeepSpeed、MegaTrain 与 PyTorch TORCH 是预训练 checkpoint 的 exact-model 测试；APTMoE 是组件同构 deployment proxy，不是 exact-model 测试。
- Server 使用 8 张 GPU、global batch size 8。KTransformers / DeepSpeed / MegaTrain 测试 sequence length 32–4096；PyTorch TORCH 本轮只测 32–1024；APTMoE 本轮显式测试 32、64、256、512、1024，启动命令没有包含 128，因此没有 128 结果。
- DeepSpeed 使用 ZeRO-3，优化器 offload 到 CPU，参数不 offload；启用 BF16 master weights/gradients 和 BF16 optimizer states。
- 配置术语需要特别区分：本轮不是“将优化器 offload 到 GPU”，而是保留 `offload_optimizer.device=cpu` 并删除 `offload_param`。因此优化器状态和计算位于 CPU，参数分片则常驻 GPU。
- 每档计划执行 15 step，前 5 step 为 warmup。成功档位的表中时间均为后 10 个稳定 step 的均值。KTransformers Server sequence 4096 的旧测试受其他 GPU 进程干扰而 CUDA OOM，当前 full sweep 又因 NUMA node 0 内存策略约束下的主机 OOM 导致 CPU owner rank 0 被内核 `SIGKILL`；独立清洁环境复测完整执行，因此主结果表采用该成功复测。KTransformers / DeepSpeed / MegaTrain 的 sequence 4096 均有有效稳定结果。
- PyTorch TORCH 使用只读 `/mnt/data2/xmy/venv`，经 `PYTHONPATH` 注入 `torch-moe-qwen-current` 的 transformers 5.6 / accelerate / llamafactory-vendor。Routed expert 常驻 CPU（`kt_backend=TORCH`，`kt_num_gpu_experts=0`），按 rank 分片；关闭 gradient checkpointing；每 rank 64 个 CPU 线程。32–1024 因中途停扫、改序重跑而拆成三次独立进程扫频，协议（15/5、dataset、batch、计时）相同；主表只采用这三次中的成功档，不含 3-step smoke，也不含被 SIGTERM 打断的未完成 1024。
- APTMoE 使用参数量为 106,852,245,504 的 GLM-4.5-Air 组件同构代理，执行真实 forward、backward 和 optimizer update，但权重来自确定性随机 BF16 初始化、不是 checkpoint-compatible。由于本轮显式允许 synthetic routing 和 unprofiled placement，五档结果均标记为 `SMOKE_ONLY`，只用于修复回归和代理路径性能观察，不能当作 GLM-4.5-Air 正式吞吐或模型质量结果。
- `TPS = global batch size × sequence length / 稳定 step 平均时间`，单位为 token/s。它按配置长度计算 token 数，不会自动扣除 padding、mask 或被跳过的 batch。
- CPU 内存是训练进程树的 RSS 求和峰值，单位为十进制 GB。多进程共享页可能被重复计入，因此适合比较同一采集方法下的进程树占用，不等同于整机实际新增物理内存。
- DeepSpeed、MegaTrain、PyTorch TORCH 与 APTMoE 表额外列出 `host_used_peak_gb_decimal`，它是监控期间的整机实际已用内存峰值；判断约 2 TB 主机内存是否耗尽，应优先使用该值，而不是进程树 RSS 求和。PyTorch TORCH 还为每个 case 建立独立 cgroup v2 scope，并以 `memory.current` 峰值作为该任务的主内存口径；该 scope 的 `memory.max` 和 `memory.swap.max` 均为 `max`，只改进记账隔离，不限制或节省内存。APTMoE 本轮的 `cgroup_memory_peak_gb` 为空，仍以进程树 RSS 与整机 `host_used` 辅助判断。
- Server 的“单卡 GPU 峰值”取该档 8 张任务 GPU 的 `task_peak_gib` 最大值，不是 8 张卡的显存之和，也不是整卡所有进程的显存峰值。
- 每个 sequence 均使用独立训练进程，`persistent_profile_process=false`，显存峰值不包含前一档保留的模型、缓存或 allocator 状态。
- KTransformers、DeepSpeed、PyTorch TORCH 与 APTMoE 的计时模式为 `coarse_host_wall_no_cuda_sync`，没有在每个阶段强制 CUDA 同步；MegaTrain 使用 `megatrain_host_wall_with_backend_cuda_sync`，保留后端执行所需的 CUDA event 与同步。forward、backward、optimizer 是各自计时边界的平均值，不保证严格相加等于 step time；MegaTrain 的阶段时间尤其不应与另外四种无强制同步的计时边界直接等同。
- 1 TiB 的判断阈值为 1099.51 GB。日志只采样并可视化内存，没有因超过阈值自动终止，也没有自动把超过阈值判成 OOM。



## 测试身份与完成状态


| 后端            | benchmark class             | 权重来源           | 模型范围                           | Server 状态                    |
| ------------- | --------------------------- | -------------- | ------------------------------ | ---------------------------- |
| KTransformers | `exact_model_full_finetune` | 预训练 checkpoint | GLM-4.5-Air text model 端到端全量微调 | 8 档全部完成；4096 来自独立复测          |
| DeepSpeed     | `exact_model_full_finetune` | 预训练 checkpoint | GLM-4.5-Air text model 端到端全量微调 | 8 档全部完成                      |
| MegaTrain     | `exact_model_full_finetune` | 预训练 checkpoint | GLM-4.5-Air text model 端到端全量微调 | 8 档全部完成                      |
| PyTorch TORCH | `exact_model_full_finetune` | 预训练 checkpoint | GLM-4.5-Air text model 端到端全量微调；routed expert 在 CPU | 6 档 32–1024 全部成功；未测 2048 / 4096 |
| APTMoE        | `deployment_proxy`          | 确定性随机 BF16 初始化 | GLM-4.5-Air 组件同构代理；真实 forward/backward/update | 5 档全部完成；均为 `SMOKE_ONLY`；未测 128 |


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

## PyTorch TORCH Server（8 GPU / global batch size 8）

正式结果拆在三次独立扫频里，协议均为 BF16 full FT、15 step / 5 warmup、`coarse_host_wall_no_cuda_sync`、每 rank 64 线程、`kt_num_gpu_experts=0`、gradient checkpointing 关闭：

- seq=32、64：[summary.md](../GLM-4.5-Air/test_log/20260914_115933_PYTORCH_TORCH_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../GLM-4.5-Air/test_log/20260914_115933_PYTORCH_TORCH_BF16_FULL_SWEEP/sweep_results.csv)
- seq=128、256、512：[summary.md](../GLM-4.5-Air/test_log/20260914_141236_PYTORCH_TORCH_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../GLM-4.5-Air/test_log/20260914_141236_PYTORCH_TORCH_BF16_FULL_SWEEP/sweep_results.csv)
- seq=1024：[summary.md](../GLM-4.5-Air/test_log/20260914_194233_PYTORCH_TORCH_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../GLM-4.5-Air/test_log/20260914_194233_PYTORCH_TORCH_BF16_FULL_SWEEP/sweep_results.csv)

| Seq  | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | CPU cgroup (GB) | Host used (GB) | Step (s) | TPS  | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
| ---- | --------------- | --------------- | ------------- | ------------- | -------- | ---- | ----------- | ------------ | ------------- | ---- |
| 1024 | 30.62           | 1363.39         | 1124.62       | 1205.87       | 874.939  | 9.36 | 83.334      | 767.552      | 4.316         | SUCCESS |
| 512  | 26.24           | 1359.23         | 1119.99       | 1203.11       | 560.176  | 7.31 | 60.306      | 483.190      | 4.390         | SUCCESS |
| 256  | 26.14           | 1347.75         | 1106.20       | 1187.77       | 389.679  | 5.26 | 37.060      | 339.336      | 4.360         | SUCCESS |
| 128  | 24.04           | 1352.05         | 1108.51       | 1190.82       | 294.937  | 3.47 | 26.850      | 256.504      | 4.355         | SUCCESS |
| 64   | 24.68           | 1337.29         | 1092.19       | 1174.12       | 213.567  | 2.40 | 12.388      | 190.500      | 4.363         | SUCCESS |
| 32   | 24.61           | 1331.54         | 1086.94       | 1168.86       | 138.705  | 1.85 | 7.039       | 122.574      | 4.262         | SUCCESS |

六档退出码均为 0，每档 10 个稳定 step。日志中 `glm45_model_contract` 为 `contract=OK`：`logical_trainable=logical_total=106852245504`，`kt_wrappers=45`，`registered_trainable=7195582464`，`kt_managed=11072962560`，其余 routed expert 以 placeholder 计。Trainer 打印的 GPU 可训练参数为 7.20B；每个 rank 另向 optimizer 注入 15 个 owner-local expert 参数。

CPU 主口径是独立 cgroup v2 `memory.current`，约 1086.94–1124.62 GB。seq=32 / 64 低于 1 TiB 阈值 1099.51 GB；seq=128–1024 略超该阈值（1106.20–1124.62 GB），但监控未自动终止，也没有 CUDA OOM 或内核 OOM。整机 `host_used` 约 1168.86–1205.87 GB，相对监控器报告的 2164.13 GB 总内存仍有约 0.96–1.00 TB 余量。进程树 RSS 约 1331.54–1363.39 GB，高于 cgroup / `host_used`，说明多 rank 共享页被 RSS 求和重复计入。Optimizer 几乎不随长度变化（4.26–4.39 s）。Backward 占稳定步时的约 86%–89%（seq=32：122.574 / 138.705；seq=1024：767.552 / 874.939），是吞吐瓶颈。tokens/step 从 256 增到 8192（32×）时，TPS 只从 1.85 增到 9.36（5.1×）。这与 CPU-resident expert 的反向计算一致，且本后端关闭了 gradient checkpointing，不能把与 KTransformers / DeepSpeed / MegaTrain 的 TPS 差距直接读成实现质量名次。

单卡任务显存峰值 24.04–30.62 GiB，seq=1024 整卡峰值约 31.24 GiB，仍在约 48 GiB 容量内；本轮长度上限不是 GPU。seq=128 的 24.04 GiB 略低于 seq=32 / 64 的 24.61 / 24.68 GiB，来自不同扫频目录的独立进程，属于采样波动，不表示更长序列反而更省显存。

未纳入主表的实验：`20260914_005019` 的 seq=32 只有 3 step / 1 warmup（稳定 2 step，TPS 1.66），不是 15/5 协议；`20260914_115933` 的 seq=128 在启动阶段被中断，无 timing；`20260914_141236` 的 seq=1024 在写出 `***** Running training *****` 之前被 SIGTERM 停掉（exit 143），以便单独重跑，该目录不能当失败性能点。正式 seq=1024 以 `20260914_194233` 为准。

## APTMoE deployment proxy Server（8 GPU / global batch size 8）

原始汇总：[summary.md](../GLM-4.5-Air/test_log/20260915_060758_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY/summary.md) · [sweep_results.csv](../GLM-4.5-Air/test_log/20260915_060758_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY/sweep_results.csv)

| Seq  | 单卡任务 GPU 峰值 (GiB) | 整机实际内存峰值 (GB) | CPU RSS 求和峰值 (GB) | Step (s) | TPS   | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
| ---- | ----------------- | ------------- | ----------------- | -------- | ----- | ----------- | ------------ | ------------- | ---- |
| 1024 | 44.34             | 1342.85       | 1254.82           | 90.023   | 91.00 | 24.660      | 64.807       | 13.967        | `SMOKE_ONLY` |
| 512  | 36.81             | 1338.48       | 1250.36           | 63.201   | 64.81 | 19.093      | 41.754       | 13.856        | `SMOKE_ONLY` |
| 256  | 31.15             | 1275.07       | 1187.25           | 49.088   | 41.72 | 15.837      | 30.132       | 14.715        | `SMOKE_ONLY` |
| 64   | 27.40             | 1247.34       | 1159.00           | 40.643   | 12.60 | 11.397      | 23.406       | 12.558        | `SMOKE_ONLY` |
| 32   | 27.06             | 1186.66       | 1098.52           | 33.435   | 7.66  | 10.278      | 19.143       | 12.485        | `SMOKE_ONLY` |

稳定 step 的离散度如下。主表 TPS 使用 `tokens_per_step / mean(step)`；下表的 TPS mean/P50/P95/std 则先逐 step 计算 TPS 再做统计，因此 TPS mean 与主表值会有轻微差异。P95 使用线性插值，std 为总体标准差。

| Seq | Step P50 (s) | Step P95 (s) | Step std (s) | TPS mean | TPS P50 | TPS P95 | TPS std |
| --- | ------------ | ------------ | ------------ | -------- | ------- | ------- | ------- |
| 1024 | 88.216 | 100.707 | 6.176 | 91.375 | 92.863 | 95.972 | 5.500 |
| 512  | 60.183 | 76.439  | 7.817 | 65.581 | 68.060 | 69.187 | 6.256 |
| 256  | 46.580 | 60.789  | 7.614 | 42.446 | 43.967 | 44.785 | 4.676 |
| 64   | 39.208 | 46.599  | 3.495 | 12.681 | 13.058 | 13.679 | 0.974 |
| 32   | 32.466 | 38.238  | 2.828 | 7.703  | 7.885  | 8.142  | 0.550 |

五档均退出码为 0，各有 10 个稳定 step。每档审计均满足 `valid_full_update=true`、`after_backward_drop_cpu_only=true`、`final_optimizer_residency_cpu_only=true`、`co_location_repair_count=0`，Adam 状态设备集合仅为 `cpu`。这说明“历史预测专家未卸载，导致 Adam 状态混合驻留 CPU/CUDA”的生命周期修复已通过本轮五长度回归，不是靠 optimizer 阶段临时搬运状态得到的伪成功。

本轮性能点仍不是正式 APTMoE/GLM 吞吐：`proxy_manifest.json` 明确记录 `benchmark_class=deployment_proxy`、`weight_source=deterministic_random_bf16_initialization`、`checkpoint_compatible=false`，并使用 `synthetic_router_smoke_only` 和 `unprofiled_fraction_smoke_only`。正式性能结论仍需 exact route trace、当前主机对应的 placement lookup 和非空的独立 cgroup 内存采集。此次命令显式列出的长度是 `1024,512,256,64,32`，所以 seq=128 没有被调度，也没有可补入表格的结果。

## 结果解读与注意事项

- KTransformers Server 具有 sequence 32–4096 全部 8 个档位的有效结果：32–2048 采用当前 full sweep，4096 采用清洁环境独立成功复测。当前 full sweep 的 4096 因严格 NUMA node 0 内存绑定触发局部 OOM，CPU owner rank 0 被内核 `SIGKILL`，故不用于性能汇总；成功复测的 TPS 为 383.92，单卡任务显存峰值为 27.38 GiB。
- DeepSpeed 在仅将优化器 offload 到 CPU、参数分片保留在 GPU 的配置下完成全部 8 个 Server 档位，包括 sequence 4096；此前参数和优化器同时 CPU offload 时，各 sequence 均会因主机 RAM 耗尽而 OOM。
- MegaTrain 完成 sequence 32–4096 全部 8 个 Server 档位；sequence 4096 的 TPS 为 508.51，单卡任务显存峰值为 34.73 GiB。
- PyTorch TORCH 完成 sequence 32–1024 全部 6 个档位，未测 2048 / 4096。seq=1024 的 TPS 为 9.36，单卡任务显存峰值 30.62 GiB；cgroup 峰值 1124.62 GB，`host_used` 1205.87 GB。
- APTMoE deployment proxy 完成显式请求的 32、64、256、512、1024 五档，TPS 分别为 7.66、12.60、41.72、64.81、91.00；五档全更新审计和 CPU-only 驻留审计均通过。该结论确认目标生命周期缺陷的修复回归，但 synthetic routing、unprofiled placement、随机权重与 linear/component proxy 口径决定了这些 TPS 不能与四个 exact-model 后端直接排名。
- KTransformers Server 的 TPS 从 sequence 32 的 21.85 增至 sequence 4096 的 383.92。
- 三个旧后端在 sequence 4096 上均有有效结果：DeepSpeed、MegaTrain 和 KTransformers 的 TPS 分别为 619.85、508.51 和 383.92。KTransformers/DeepSpeed TPS 比值从 sequence 32 的 4.38 倍逐步下降到 sequence 2048 的 1.07 倍，并在 sequence 4096 降至 0.62，即 DeepSpeed 此时约为 KTransformers 的 1.61 倍。
- 在已测重叠长度上，PyTorch TORCH 明显更慢。seq=32：TORCH 1.85 vs KT 21.85 vs DeepSpeed 4.99 vs MegaTrain 3.75；seq=1024：TORCH 9.36 vs KT 267.72 vs DeepSpeed 165.76 vs MegaTrain 133.91。计时模式与 KT / DeepSpeed 相同，但 expert 在 CPU、且未开 gradient checkpointing，只适合作为 CPU-expert TORCH 路径的吞吐/显存基线，不宜直接排成第四名实现。
- DeepSpeed Server 的整机实际内存峰值约 1.66 TB，说明当前“仅优化器 CPU offload”配置没有耗尽约 2 TB 主机内存；sequence 4096 的单卡任务显存峰值为 43.29 GiB，是该配置更接近的容量边界。
- MegaTrain Server 的整机实际内存峰值约 1.60 TB、sequence 4096 单卡任务显存峰值为 34.73 GiB，均低于本轮 DeepSpeed 对应峰值；但 MegaTrain sequence 4096 的 TPS 也比 DeepSpeed 低约 18%。
- PyTorch TORCH 的约束与 Qwen3.5-122B 同后端相反：这里 GPU 仍有余量（seq=1024 仅 30.62 / 47.99 GiB），主机 cgroup 在长序列上略超 1 TiB 记账阈值，但 `host_used` 约 1.17–1.21 TB，没有耗尽约 2 TB 主机内存。吞吐限制在 CPU expert 反向，不在显存。
- KTransformers、DeepSpeed、MegaTrain、PyTorch TORCH 和 APTMoE Server 所有已测档位的进程树 RSS 求和峰值均接近或超过 1 TiB。该结果没有触发自动终止，且 RSS 求和可能重复统计共享页，不应直接解释为同等规模的整机新增物理内存。
- 四个 exact-model 后端使用相同模型、数据、GPU 数量、batch 和 step 数；KTransformers、DeepSpeed 与 PyTorch TORCH 使用无强制 CUDA 同步的 host-wall 计时，MegaTrain 保留后端所需同步。端到端 TPS 可作为本轮实测参考，但阶段时间边界和内部专家执行、内存布局不同，结论不应外推到其他模型或配置。PyTorch TORCH 若要进入同一张 32–4096 对比表，还需补 2048 / 4096 以及与 KT 对齐的 checkpointing 设置。APTMoE 虽使用相同 GPU、batch、step 和 TPS 公式，但属于随机权重代理，必须独立阅读。
