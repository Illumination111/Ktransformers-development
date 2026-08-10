# Qwen3.5-35B-A3B 全量微调测试（KT / DeepSpeed / APTMoE / MegaTrain）

## 测试口径

- BF16、text-only、全参数微调。KTransformers（KT）和 DeepSpeed 使用真实模型；APTMoE 因暂不支持该架构，仅在 consumer 模式运行等运算量 deployment proxy，其 TPS 不代表真实模型端到端 TPS。
- Server：8×RTX 4090 48G、global batch 8；consumer：2×RTX 4090 48G、
  global batch 2。当前脚本不再设置 1 TiB 主机内存硬上限，而是在完整训练期间
  采样 CPU/GPU 内存并在结束后人工判断是否按 OOM 记录。
- KT/DeepSpeed consumer 采用现有旧档位 `32～4096`；APTMoE consumer 采用 `16～2048`。`—` 表示未测试。

下表是取消硬上限之前的历史结果，不能把其中由旧 cgroup 触发的 OOM 直接外推到新
脚本。MegaTrain 尚无端到端实测值。

## TPS 结果

### Server

| Sequence length | KT TPS | DeepSpeed TPS |
|---:|---:|---:|
| 32 | 54.71 | 13.60 |
| 64 | 99.89 | 26.60 |
| 128 | 182.83 | 54.03 |
| 256 | 324.61 | 108.59 |
| 512 | 517.98 | 217.68 |
| 1024 | 701.19 | 432.02 |
| 2048 | 929.25 | 847.48 |
| 4096 | 1045.89 | 1689.07 |

### Consumer

| Sequence length | KT TPS | DeepSpeed | APTMoE proxy TPS |
|---:|---:|---:|---:|
| 16 | — | — | 3.67 |
| 32 | 15.77 | OOM | 5.65 |
| 64 | 29.47 | OOM | 9.36 |
| 128 | 55.43 | OOM | 15.09 |
| 256 | 102.53 | OOM | 26.13 |
| 512 | 193.27 | OOM | 43.45 |
| 1024 | 339.28 | OOM | 70.94 |
| 2048 | 550.39 | OOM | 100.38 |
| 4096 | 755.16 | OOM | — |

## 简要分析

### DeepSpeed 选择性 offload

- 在旧的 1 TiB cgroup 测试中，参数和优化器均 offload 到 CPU，DeepSpeed 估算主机
  内存约 871.57 GiB；叠加 pinned/staging buffer 后所有档位均在首个 step 前被旧
  cgroup OOM killer 终止。当前脚本不再自动终止，必须根据 `memory_summary.md` 和
  CPU 曲线重新人工定性。
- 对 48G GPU，建议只 offload 优化器：估算主机内存降至 774.73 GiB，每卡显存约 35.28 GiB，理论可行但最长序列仍需实测。只 offload 参数会让每卡承担约 258 GiB 的优化器相关状态，不可行。

### KTransformers 长序列 TPS 增长受限

- KT 的多卡 CPU MoE 是 rank0 集中式执行：各 rank 的 token 汇聚到 rank0，server 中 rank0 使用 80 个 CPU 核，其余 rank 各 2 核，因此 CPU expert 计算无法随 8 张 GPU 横向扩展。
- Server 从 seq=2048 增至 4096 时，step time 从 17.631 s 增至 31.330 s（1.78×），TPS 仅从 929.25 增至 1045.89（1.13×）；4096 档 forward+backward 占 29.27/31.33 s，而 optimizer 仅 1.92 s，瓶颈位于 rank0 的 CPU expert 前反向计算。
