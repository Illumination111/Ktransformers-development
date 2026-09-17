# Qwen3.5-122B-A10B Server 测试结果分析

本文分析 `Qwen3.5-122B-A10B/test_log` 中四个后端的 **server** 全量扫描日志（本轮未跑 consumer，也未包含 APTMoE）：

| 后端 | Server |
|---|---|
| KTransformers | [summary](../Qwen3.5-122B-A10B/test_log/20260803_115140_KTRANSFORMERS_BF16_FULL_SWEEP/summary.md) |
| MegaTrain | [summary](../Qwen3.5-122B-A10B/test_log/20260802_094707_MEGATRAIN_BF16_FULL_SWEEP/summary.md) |
| DeepSpeed | [summary](../Qwen3.5-122B-A10B/test_log/20260802_133440_DEEPSPEED_BF16_FULL_SWEEP/summary.md) |
| PyTorch TORCH | [summary](../Qwen3.5-122B-A10B/test_log/20260914_011039_PYTORCH_TORCH_BF16_FULL_SWEEP/summary.md) |

## 测量口径

- 四组测试均为 BF16、text-only、配置为 full finetuning、LoRA rank 0，per-device batch size 1，gradient accumulation steps 1。
- Server 使用 8 张 GPU、global batch size 8。KTransformers / MegaTrain / DeepSpeed 测试 sequence length 32–4096；PyTorch TORCH 本轮只测 32–1024。每个 sequence 均为独立训练进程（`persistent_profile_process=false`）。
- PyTorch TORCH 使用只读 `/mnt/data2/xmy/venv`，经 `PYTHONPATH` 注入 `torch-moe-qwen-current` 的 transformers 5.6 / accelerate / llamafactory-vendor。Routed expert 常驻 CPU（`kt_backend=TORCH`，`kt_num_gpu_experts=0`），按 rank 分片；关闭 gradient checkpointing；每 rank 64 个 CPU 线程。
- 每档配置 15 step，前 5 step 为 warmup；成功档的时间均为后 10 个稳定 step 的均值。
- `TPS = global batch size × sequence length / 稳定 step 平均时间`，单位为 token/s。它按配置长度计算 token 数，不会自动扣除 padding、mask 或被跳过的 batch。
- 表中 CPU 内存是训练进程树的 RSS 求和峰值，单位为十进制 GB。多进程共享页可能被重复计入，因此它适合比较同一采集方法下的进程树占用，不等同于整机实际新增物理内存。同目录 `memory_summary.json` 另给出 `host_used_peak_gb_decimal`（整机 used 采样峰值）。KTransformers 与 PyTorch TORCH 还为每个 case 建立了独立 cgroup v2 scope，并以 `memory.current` 峰值作为该任务的主内存口径；该 scope 的 `memory.max` 和 `memory.swap.max` 均为 `max`，只改进记账隔离，不限制或节省内存。PyTorch TORCH 结果表额外列出 cgroup 与 `host_used`，避免只看 RSS。
- Server 表中的“单卡 GPU 峰值”取该档 8 张任务 GPU 的 `task_peak_gib` 最大值，不是 8 张卡的显存之和；单位为 GiB，不包含其他任务和驱动基线占用。
- DeepSpeed、KTransformers、PyTorch TORCH 的计时模式为 `coarse_host_wall_no_cuda_sync`，没有在每个阶段强制 CUDA 同步。MegaTrain 为 `megatrain_host_wall_with_backend_cuda_sync`：forward/backward 使用 CUDA event，optimizer 和 step total 使用 host wall time。不同计时模式的阶段时间不宜直接当作完全相同的测量量。
- forward、backward、optimizer 是各自计时边界的平均值；由于异步执行、同步点和阶段交叠，它们不保证严格相加等于 step time。
- 1 TiB 的判断阈值是 1099.51 GB。日志只采样并可视化内存，没有因超过阈值自动终止，也没有自动把超过阈值判成 OOM。
- 本轮 KTransformers 已在 shared-expert 参数契约修复之后运行（见 [Qwen3.5-KTransformers-shared-expert-fix.md](./Qwen3.5-KTransformers-shared-expert-fix.md)）；成功档日志中 `logical_total=122111526912`、`contract=OK`。

## 测试身份与可比性

| 后端 | benchmark class | 权重来源 | 模型范围 | Server 状态 |
|---|---|---|---|---|
| KTransformers | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5-122B text model 端到端全量微调 | 8/8 独立进程成功 |
| MegaTrain | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5-122B text model 端到端全量微调 | 8/8 独立进程成功 |
| DeepSpeed | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5-122B text model 端到端全量微调 | 8/8 均在训练启动后被杀死，无稳定 timing |
| PyTorch TORCH | `exact_model_full_finetune` | 预训练 checkpoint | Qwen3.5-122B text model 端到端全量微调；routed expert 在 CPU | 6 档中 32–512 成功，1024 CUDA OOM |

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

### PyTorch TORCH

原始汇总：[summary.md](../Qwen3.5-122B-A10B/test_log/20260914_011039_PYTORCH_TORCH_BF16_FULL_SWEEP/summary.md) · [sweep_results.csv](../Qwen3.5-122B-A10B/test_log/20260914_011039_PYTORCH_TORCH_BF16_FULL_SWEEP/sweep_results.csv)

本轮扫频目录为 `20260914_011039_PYTORCH_TORCH_BF16_FULL_SWEEP`，长度列表为 1024、512、256、128、64、32（与 KT/MegaTrain/DeepSpeed 的 32–4096 不完全重合）。计时模式与 KTransformers / DeepSpeed 相同，为 `coarse_host_wall_no_cuda_sync`。

| Seq | 单卡 GPU 峰值 (GiB) | CPU RSS 峰值 (GB) | CPU cgroup (GB) | Host used (GB) | Step (s) | TPS | Forward (s) | Backward (s) | Optimizer (s) | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 47.36 | 279.39 | 93.70 | 173.99 | - | - | - | - | - | FAILED |
| 512 | 42.39 | 555.71 | 381.37 | 458.51 | 321.420 | 12.74 | 49.123 | 269.742 | 1.953 | SUCCESS |
| 256 | 31.72 | 541.95 | 364.93 | 446.24 | 214.750 | 9.54 | 33.208 | 179.024 | 1.891 | SUCCESS |
| 128 | 28.90 | 528.44 | 359.42 | 441.09 | 151.398 | 6.76 | 19.410 | 129.467 | 1.892 | SUCCESS |
| 64 | 24.76 | 506.82 | 356.13 | 437.59 | 103.288 | 4.96 | 10.616 | 90.156 | 1.901 | SUCCESS |
| 32 | 24.34 | 481.64 | 351.90 | 433.45 | 68.068 | 3.76 | 6.940 | 58.649 | 1.873 | SUCCESS |

成功五档退出码均为 0，每档 10 个稳定 step。seq=512 日志中 `qwen35_122b_model_contract` 为 `contract=OK`：`logical_trainable=logical_total=122111526912`，`kt_wrappers=48`，`registered_trainable=6147409920`，`kt_managed=14495514624`，其余 routed expert 以 placeholder 计。Trainer 打印的 GPU 可训练参数为 6.15B；每个 rank 另向 optimizer 注入 12 个 owner-local expert 参数。

成功档的 CPU 主口径是独立 cgroup v2 `memory.current`，约 351.90–381.37 GB，远低于 1 TiB 阈值 1099.51 GB。进程树 RSS 约 481.64–555.71 GB，`host_used` 约 433.45–458.51 GB，三者都不构成这轮失败原因。Optimizer 几乎不随长度变化（1.87–1.95 s）。Backward 占稳定步时的 83%–87%（seq=32：58.649 / 68.068；seq=512：269.742 / 321.420），是吞吐瓶颈。tokens/step 从 256 增到 4096（16×）时，TPS 只从 3.76 增到 12.74（3.4×），明显低于 KTransformers 同长度（seq=512：189.68；seq=32：16.14）。这与 CPU-resident expert 的反向计算一致，不能把差距直接读成实现质量差异；本后端也关闭了 gradient checkpointing，激活显存路径与 KT 不同。

seq=1024 在打印 `***** Running training *****` 之后、写出任何稳定 step 之前 CUDA OOM：各 rank 申请 2–20 MiB 时，单卡已占用约 47.36 GiB / 47.37 GiB。该档 cgroup / RSS / `host_used` 峰值分别为 93.70 / 279.39 / 173.99 GB，是加载阶段被打断后的值，不能与成功档的稳态内存比较。seq=512 单卡任务显存已达 42.39 GiB（约 48 GiB 卡的 88%），256→512 再增加约 10.7 GiB；按这个斜率 1024 放不进激活。因此本后端在当前 8×48 GiB、无 gradient checkpointing、global batch 8 的配置下，可用长度是 32–512，上限是 GPU 而不是主机内存。

## 横向对比注意事项

- 现有 exact-model server 数据为 DeepSpeed、KTransformers、MegaTrain、PyTorch TORCH；没有 APTMoE，也没有 consumer 剖面。PyTorch TORCH 本轮未测 2048 / 4096。
- 在可比的成功档中，KTransformers 在 seq≤2048 的 TPS 明显高于 MegaTrain（例如 seq=2048：397.78 vs 219.12）；本次 KTransformers 也跑通了 seq=4096（TPS 490.42）。但两者计时模式不同，不能当作严格同口径名次。
- PyTorch TORCH 在已测重叠长度上明显更慢（seq=512：12.74 vs KT 189.68 vs MegaTrain 54.78；seq=32：3.76 vs KT 16.14 vs MegaTrain 3.29）。计时模式与 KT 相同，但 expert 在 CPU、且未开 gradient checkpointing，只适合作为 CPU-expert TORCH 路径的吞吐/显存基线，不宜直接排成第四名实现。
- 三组旧后端的进程树 RSS 峰值都超过 1 TiB。MegaTrain 的进程树峰值约 5.6 TB，但其 `host_used`（约 1.69–1.70 TB）反而低于 KTransformers 成功档和 DeepSpeed 失败档，不能把进程树 RSS 直接读成整机物理占用。PyTorch TORCH 成功档 cgroup 仅约 352–381 GB、`host_used` 约 433–459 GB，主机内存压力明显更低。
- DeepSpeed 八档全部在训练启动后被系统信号杀死，当前配置下没有可用 throughput 数字；若要继续对比，需要降低 CPU offload 压力或扩充主机可用内存后再测。
- KTransformers seq=4096 已补齐；该点的单卡 GPU 峰值仅 27.68/47.99 GiB，真正接近上限的资源仍是主机内存。对 2164.13 GB 主机而言，160.19 GB 最低可用量可以跑通本次 15 step，但不应视为充足的长时稳态安全余量。
- PyTorch TORCH 的约束相反：CPU 仍有大量余量，seq=1024 死在单卡 47.36/47.37 GiB 的 CUDA OOM。若要测更长序列，需要 gradient checkpointing、减小 microbatch，或接受该后端在 48 GiB 卡上止于 512。
- 若要得到可发表的严格后端对比，仍应统一 CUDA 同步计时口径、确认每个后端确实更新全部参数，并补齐 consumer 剖面或在相同主机内存预算下重跑 DeepSpeed。PyTorch TORCH 若要进入同一张 32–4096 对比表，还需补 2048 / 4096（在显存允许的前提下）以及与 KT 对齐的 checkpointing 设置。
