# APTMoE Baseline Benchmark

> APTMoE pipeline-parallel offloading baseline 对比数据
>
> 测试日期：2026-03-18~19，服务器：sapphire4
>
> 交接文档（完整改动清单 + 测试方法 + 可应用的 patch）：`aptmoe_changes_and_testing.md` / `aptmoe_changes.diff`

## 1. 概述

APTMoE 是一种面向带宽受限 GPU 节点的 MoE 训练系统，使用 pipeline parallelism + CPU offloading。
本文档记录在 sap4 上用 APTMoE 的 benchmark 模式跑合成模型的速度数据，用于与 KT MoE 和 ZeRO-3 对比。

**仓库**：https://github.com/JimmyPeilinLi/APTMoE-baseline → `/home/star/APTMoE-baseline/` (sap4)

## 2. 环境

| 项目 | 值 |
|------|-----|
| 服务器 | sapphire4 (star@117.74.64.224:15016) |
| GPU | 8 × NVIDIA RTX 4090 (48GB VRAM each) |
| CPU | Intel Xeon Platinum 8488C, 2 sockets × 48 cores |
| RAM | 2016 GB |
| Python | 3.12.12 (sft 环境) |
| PyTorch | 2.10.0+cu128 |
| 代码路径 | `/home/star/APTMoE-baseline/` |
| 环境 | `/mnt/data2/hxx/mini/envs/sft/` |
| 日志 | `/tmp/aptmoe_logs/` (sap4) |

## 3. 对 APTMoE 的修改

原始 APTMoE benchmark 模式使用合成权重的 TransformerLM 模型。为更公平地对比，做了以下修改：

### 3.1 Expert FFN: 2-linear → 3-linear SwiGLU

**文件**: `model/transformer_lm.py` — `InnerFeedForwardLayer`

原始用 `Linear(H,I) → ReLU → Linear(I,H)`（2 个 linear），真实 MoE 模型用 SwiGLU（3 个 linear）。

```python
# 修改后
class InnerFeedForwardLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, dim_feedforward, bias=False)
        self.up_proj = nn.Linear(d_model, dim_feedforward, bias=False)
        self.down_proj = nn.Linear(dim_feedforward, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**影响**：参数量从 262B → 391B (Qwen3.5)，从 471B → 700B (DSV3)，与真实模型对齐。

### 3.2 Top-k Routing

**文件**: `model/top2gate.py`, `model/moe_layer.py`, `model/transformer_lm.py`

原始 `RandomGate` 只支持 top-1 routing（每 token 分配 1 个 expert）。真实模型用 top-k（Qwen3.5: top-10, DSV3: top-8）。

修改内容：
- `RandomGate` 新增 `num_experts_per_tok` 参数
- `random_gating()` 新增 topk 路径：top-k 选择 → 按 expert 排序 → 返回 (counts, gather_indices)
- `MoELayer.forward()` 新增 `_expand_for_topk()` / `_reduce_from_topk()` 处理 token 展开和归约
- 删除 `assert input[0].shape[1] % len(self.experts) == 0`（top-k 下不需要整除）
- `num_experts_per_tok` 通过 `TransformerDecoderLayer` → `TransformerLM` → `_build_stage` 穿透
- CLI 新增 `--num_experts_per_tok` 参数覆盖 model_config 默认值

### 3.3 LoRA 适配 SwiGLU

**文件**: `model/lora.py` — `_apply_lora_to_transformer()`

原始 LoRA 代码引用 `inner.linear1` / `inner.linear2`，改为检测 `gate_proj` / `up_proj` / `down_proj`：

```python
if hasattr(inner, 'gate_proj'):
    inner.gate_proj = LoRALinear(inner.gate_proj, ...)
    inner.up_proj = LoRALinear(inner.up_proj, ...)
    inner.down_proj = LoRALinear(inner.down_proj, ...)
else:
    inner.linear1 = LoRALinear(inner.linear1, ...)
    inner.linear2 = LoRALinear(inner.linear2, ...)
```

### 3.4 Qwen3.5-397B 模型配置

**文件**: `utils.py`, `main.py`, `Static/lookup_table.py`, `Runtime/OffloadRuntime/R_solver.py`

新增 `QWEN35_397B` 配置：

```python
# utils.py model_config()
elif model_conf == 'QWEN35_397B':
    _set('embedding_dim', 4096)
    _set('hidden_dim', 1024)       # moe_intermediate_size
    _set('num_heads', 32)
    _set('num_layers', 60)
    _set('num_stages', 60)
    _set('num_experts', 512)
    _set('num_experts_per_tok', 10)
    _set('batch_size', 1)
    _set('num_chunks', 1)
    _set('seq_length', 512)
```

来源：`/mnt/data3/models/Qwen3.5-397B-A17B/config.json` (sap4)

Lookup table 基于 QWEN3_235B 按 intermediate_size 比例 (1024/1536) 缩放。

## 4. Benchmark 模式说明

### 4.1 运行方式

```bash
cd /home/star/APTMoE-baseline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /mnt/data2/hxx/mini/envs/sft/bin/torchrun --nproc_per_node 8 --master_port 29xxx \
  ./main.py --is_moe=True --num_training_steps=3 --num_warmup_steps=1 \
  --model_config=QWEN35_397B --seq_length=512 --num_experts_per_tok=2 \
  --gini=0.3 --topo=C1+G2 --pipeline=APTMoE \
  [--lora --lora_rank=8 --lora_target=all]
```

### 4.2 计时逻辑

- Warmup: `num_warmup_steps` 步（默认 1），不计时
- Measure: `num_training_steps` 步（默认 3），前后 `torch.cuda.synchronize()`
- 每步包含完整 forward + backward + `optimizer.step()`
- `training_time = wall_clock / num_training_steps`
- `TPS = batch_size × seq_length / training_time`

### 4.3 模型结构 vs 真实模型

| 维度 | Benchmark (TransformerLM) | 真实模型 |
|------|--------------------------|---------|
| Expert FFN | SwiGLU 3-linear (修改后) | SwiGLU 3-linear ✓ |
| Attention | `nn.MultiheadAttention` (dense MHA) | GQA/MLA + RoPE + Flash Attn |
| Normalization | `nn.LayerNorm` | `RMSNorm` |
| Shared Expert | 无 | 模型相关 |
| Routing | `RandomGate` (power law 模拟) | 真实 softmax/sigmoid gate |
| 数据 | `torch.rand` 随机 tensor | 真实 tokenized text |
| Loss | 有（随机数据上的 CE loss） | 有 |
| 权重 | 随机初始化 BF16 | 预训练权重 |

### 4.4 已知限制

1. **seq_length 必须整除 num_experts**：`RandomGate` 的 top-1 路径有 `assert num_tokens % num_experts == 0`。Qwen3.5 (512 experts) 最小 seqlen=512，DSV3 (256 experts) 最小 seqlen=256。
2. **_user_specified 陷阱**：`--seq_length=X` 如果 X 等于 argparse 默认值 (128)，会被 model_config 的 `_set()` 覆盖。
3. **Pipeline 不支持 grad accumulation**：每次 `run_pipeline()` 结束后 experts 被 offload 到 CPU，第二轮调用需要完整的 action list 重新 load。

## 5. 结果

### 5.1 Qwen3.5-397B (391B 合成) — 全量训练 (无 LoRA)

8×4090, `--pipeline=APTMoE --model_config=QWEN35_397B --gini=0.3 --topo=C1+G2`

| TopK | SeqLen | Time/Step (s) | TPS (tok/s) | Peak GPU (GB) | Status |
|------|--------|--------------|-------------|---------------|--------|
| 1 | 512 | 23.7 | 21.6 | 15.1 | OK |
| 2 | 512 | 17.8 | 28.8 | — | OK |
| 2 | 1024 | 24.1 | 42.5 | — | OK |
| 2 | 2048 | — | — | — | killed (CPU RAM) |
| 4+ | 512 | — | — | — | killed (CPU RAM) |

全量训练的 optimizer state (Adam m/v) 消耗大量 CPU RAM，限制了 topk 和 seqlen 范围。

### 5.2 Qwen3.5-397B (391B 合成) — LoRA

8×4090, `--lora --lora_rank=8 --lora_target=all`

| TopK | SeqLen | Time/Step (s) | TPS (tok/s) | Status |
|------|--------|--------------|-------------|--------|
| **1** | 512 | 26.5 | 19.4 | OK |
| **1** | 1024 | 26.6 | 38.5 | OK |
| **1** | 2048 | 27.2 | 75.2 | OK |
| **1** | 4096 | 28.0 | 146.4 | OK |
| **1** | 8192 | 29.9 | 273.7 | OK |
| **1** | 16384 | — | — | GPU OOM |
| **2** | 512 | 26.6 | 19.3 | OK |
| **2** | 1024 | 27.1 | 37.8 | OK |
| **2** | 2048 | 27.9 | 73.3 | OK |
| **2** | 4096 | 29.1 | 140.6 | OK |
| **2** | 8192 | — | — | GPU OOM |
| **4** | 512 | 26.7 | 19.2 | OK |
| **4** | 1024 | 27.5 | 37.2 | OK |
| **4** | 2048 | 28.2 | 72.7 | OK |
| **4** | 4096 | 37.4 | 109.5 | OK |
| **4** | 8192 | — | — | GPU OOM |

**关键特征**：
- step 时间与 seqlen 基本无关（~27s），pipeline offload 是绝对瓶颈
- TPS 随 seqlen 线性增长（因 step 时间恒定）
- topk 对速度影响很小（LoRA 下 expert 权重不需要梯度，offload 搬运量不变）
- topk=1 可到 8192，topk=2/4 可到 4096（GPU OOM 限制）
- 模型参数 391B，LoRA 可训练参数 ~1.9B

### 5.3 DeepSeek-V3 (700B 合成) — LoRA

8×4090, `--model_config=DSV3 --lora --lora_rank=8 --lora_target=all`

| TopK | SeqLen | Time/Step (s) | TPS (tok/s) | Status |
|------|--------|--------------|-------------|--------|
| 1 | 256 | 54.6 | 4.7 | OK |
| 1 | 512 | 41.9 | 12.2 | OK |
| 1 | 1024 | 52.0 | 19.7 | OK |
| 1 | 2048 | — | — | CPU RAM killed |
| 2 | 256 | 61.7 | 4.1 | OK |
| 4 | 256 | 44.4 | 5.8 | OK |
| 4 | 512 | — | — | GPU OOM |
| 8 | 256 | — | — | GPU OOM |

**关键特征**：
- 700B 参数，基础 RSS ~170GB/进程 × 8 = 1360GB
- step 时间 ~42-62s（比 Qwen3.5 慢 ~2x，因模型更大）
- topk=4 只能在 seq=256 上跑，seq=512 GPU OOM
- topk=8 直接 GPU OOM（expert load 到 GPU 超 48GB）
- LoRA 可训练参数 ~3.5B

### 5.4 DeepSeek-V3 (700B 合成) — 全量训练 (无 LoRA)

全量训练时 optimizer state 需要 ~5.2TB，远超 2016GB RAM。连初始化都过不了。

### 5.5 对比参考

#### vs ZeRO-3 (Qwen3.5-397B 真实权重, sap4)

来源：`migrating/docs/qwen35_397b_seqlen_sweep.md`

| SeqLen | ZeRO-3 TPS | APTMoE top-1 LoRA TPS | APTMoE/ZeRO-3 |
|--------|-----------|----------------------|---------------|
| 512 | 18.9 | 19.4 | 1.0x |
| 1024 | 38.8 | 38.5 | 1.0x |
| 2048 | 70.0 | 75.2 | 1.1x |
| 4096 | 69.9 | 146.4 | 2.1x |
| 8192 | 104.1 | 273.7 | 2.6x |

注意：ZeRO-3 是真实权重 + 真实数据，APTMoE 是合成权重 + 合成数据 + top-1 routing。

#### vs KT INT8 (DeepSeek-V3 671B, sap4)

来源：`docs/bench_seqlen_sweep/README.md`

| SeqLen | KT INT8 TPS | APTMoE top-4 LoRA TPS | KT/APTMoE |
|--------|------------|----------------------|-----------|
| 256 | 95.4 | 5.8 | **16x** |
| 512 | 129.9 | — (OOM) | — |

注意：KT 是 DSV3 真实权重 + INT8 + 8GPU data parallel，APTMoE 是 DSV3 合成权重 + top-4 + pipeline parallel。

## 6. 运行命令参考

### 生成模型

```bash
# Qwen3.5-397B 全量训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29xxx \
  ./main.py --is_moe=True --num_training_steps=3 --num_warmup_steps=1 \
  --model_config=QWEN35_397B --seq_length=1024 --num_experts_per_tok=2 \
  --gini=0.3 --topo=C1+G2 --pipeline=APTMoE

# Qwen3.5-397B LoRA
# 加 --lora --lora_rank=8 --lora_target=all

# DSV3 671B LoRA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29xxx \
  ./main.py --is_moe=True --num_training_steps=3 --num_warmup_steps=1 \
  --model_config=DSV3 --seq_length=256 --num_experts_per_tok=4 \
  --gini=0.3 --topo=C1+G2 --pipeline=APTMoE \
  --lora --lora_rank=8 --lora_target=all
```

### 重要注意事项

1. **跑前必清理**：`killall -9 python3.12; sleep 5; nvidia-smi` 确认 GPU 全释放 + `free -g` 确认 RAM available > 1800GB
2. **日志必存文件**：`> /tmp/aptmoe_logs/xxx.log 2>&1`，不要管道 grep
3. **seqlen 必须整除 num_experts**：Qwen3.5 min=512, DSV3 min=256
4. **LoRA 是必须的**：DSV3 700B 全量训练 optimizer state 需 5TB+

## 7. 原始日志文件 (sap4:/tmp/aptmoe_logs/)

| 文件 | 内容 |
|------|------|
| `qwen35_397b_topk10_sweep.log` | Qwen3.5 全量 top-10 sweep (2-linear 旧版) |
| `qwen35_3linear_topk_probe.log` | Qwen3.5 3-linear 全量 topk=1,2 seq=512 |
| `qwen35_3linear_topk_clean.log` | Qwen3.5 3-linear 全量 topk=4,6,8,10 seq=512 (全 fail) |
| `qwen35_topk2_seqsweep.log` | Qwen3.5 全量 topk=2 seq=512,1024,2048 |
| `qwen35_lora_sweep.log` | **Qwen3.5 LoRA topk=1,2,4 × seq=512-16384** |
| `dsv3_3linear_lora_v2.log` | DSV3 LoRA topk=1,2,4,8 seq=256 |
| `dsv3_lora_top4_sweep.log` | DSV3 LoRA topk=4 seq=256,512,1024 |
| `dsv3_lora_top1_sweep.log` | DSV3 LoRA topk=1 seq=512,1024,2048 |
| `dsv3_3linear_probe.log` | DSV3 全量 topk=1,2 seq=256 (fail, RAM OOM) |
