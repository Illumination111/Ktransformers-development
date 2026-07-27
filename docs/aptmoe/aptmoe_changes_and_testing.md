# APTMoE Baseline：改动清单与测试方法（交接文档）

> 面向接手同事：如何从原始 APTMoE 仓库复现我们的 benchmark 环境、所有代码改动的内容与原因、以及标准测试流程。
>
> Benchmark 日期：2026-03-18~19，服务器：sapphire4。结果数据详见同目录 `README.md`。

## 1. 背景

APTMoE 是一种面向带宽受限 GPU 节点的 MoE 训练系统（pipeline parallelism + CPU offloading）。我们用它的 benchmark 模式（合成权重 TransformerLM + 随机数据）测训练速度，作为 KT MoE 和 ZeRO-3 的对比 baseline。

- **上游仓库**：https://github.com/JimmyPeilinLi/APTMoE-baseline
- **修改后代码**：sap4 `/home/star/APTMoE-baseline/`（不是 git 仓库，改动只存在于工作目录）
- **完整 diff**：同目录 `aptmoe_changes.diff`（9 个文件，可在上游 fresh clone 里 `patch -p1 < aptmoe_changes.diff` 一键复现）

改动的核心动机：原始 benchmark 模型结构与真实 MoE 模型差距太大（2-linear ReLU expert、top-1 routing），跑出来的数字对 KT 不公平。我们把 expert 结构和 routing 对齐真实模型，并新增 Qwen3.5-397B 配置。

## 2. 环境

| 项目 | 值 |
|------|-----|
| 服务器 | sapphire4（SSH 别名 `sapphire4`） |
| GPU | 8 × RTX 4090 48GB |
| CPU / RAM | 2 × Xeon 8488C（96 核）/ 2016 GB |
| Python 环境 | `/mnt/data2/hxx/mini/envs/sft/`（Py 3.12, torch 2.10.0+cu128） |
| 代码路径 | `/home/star/APTMoE-baseline/` |
| 日志目录 | sap4 `/tmp/aptmoe_logs/` |

## 3. 改动总览

共改 9 个文件，无新增/删除文件。按目的分四组：

| 组 | 文件 | 改动 |
|----|------|------|
| A. Expert 结构对齐 | `model/transformer_lm.py` | Expert FFN 从 2-linear ReLU 改为 3-linear SwiGLU |
| B. Top-k routing | `model/top2gate.py`<br>`model/moe_layer.py`<br>`model/transformer_lm.py`<br>`utils.py`<br>`main.py` | `RandomGate` 支持 top-k；`MoELayer` token 展开/归约；`num_experts_per_tok` 参数全链路穿透 + CLI 覆盖 |
| C. LoRA 适配 | `model/lora.py` | Expert LoRA 适配 SwiGLU 三投影 |
| D. Qwen3.5-397B 配置 | `utils.py`<br>`main.py`<br>`Static/lookup_table.py`<br>`Runtime/OffloadRuntime/R_solver.py` | 新增 `QWEN35_397B` model_config + lookup table |
| （杂项） | `Runtime/OffloadRuntime/offload.py`<br>`main.py` | `prefetch_portion` 0.5→0.1；`--grad_accum` 死参数（见 4.5） |

## 4. 改动详解

### 4.1 Expert FFN：2-linear ReLU → 3-linear SwiGLU

**文件**：`model/transformer_lm.py` — `InnerFeedForwardLayer`

原始实现是 `Linear(H,I) → ReLU → Dropout → Linear(I,H) → Dropout`（2 个 linear）。真实 MoE 模型（Qwen3.5 / DSV3）的 expert 是 SwiGLU（3 个 linear，无 bias 无 dropout）：

```python
class InnerFeedForwardLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, dim_feedforward, bias=False)
        self.up_proj = nn.Linear(d_model, dim_feedforward, bias=False)
        self.down_proj = nn.Linear(dim_feedforward, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**影响**：expert 参数量 ×1.5，总参数量从 262B → 391B（Qwen3.5 配置）、471B → 700B（DSV3 配置），与真实模型对齐。offload 搬运量也随之 ×1.5，这正是要公平对比的部分。

### 4.2 Top-k Routing

原始 `RandomGate` 只支持 top-1（每 token 1 个 expert），真实模型是 top-k（Qwen3.5: 10, DSV3: 8）。涉及 5 个文件：

**`model/top2gate.py`**：
- `random_gating(logits, layer_id, topk=1)`：`topk==1` 走原路径不变（保留 `num_tokens % num_experts == 0` 断言）；`topk>1` 新路径：`torch.topk` 选 k 个 expert → `randperm` 置换 expert id（保持原有的 popularity 随机化语义）→ 构建 per-expert token 列表 → 返回 `(expert_selection_counts, gather_indices)` 二元组（top-1 只返回 counts）。
- `RandomGate.__init__` 新增 `num_experts_per_tok` 参数，`forward` 透传给 `random_gating`。

**`model/moe_layer.py`**：
- 删除 `assert input[0].shape[1] % len(self.experts) == 0`（top-k 下 token 数不需要整除 expert 数）。
- 新增两个 helper：
  - `_expand_for_topk()`：按 `gather_indices` 把输入展开成 `(num_tokens × k, d_model)`，每 token 复制 k 份、按 expert 排序；
  - `_reduce_from_topk()`：`scatter_add_` 把 expert 输出加回每 token，再除以 k 取平均（RandomGate 无真实 gate 权重，只能等权平均）。
- `forward()` 通过 `getattr(self.gate, 'num_experts_per_tok', 1)` 判断是否 top-k 模式；GPipe/GPipeOffload/Mobius 路径和 APTMoE 路径都做了适配。
- ⚠️ **APTMoE 路径的回退逻辑**：APTMoE 用 `generate_similiar_list()` 模拟"预测的 expert selection"覆盖真实 selection，此时 `gather_indices` 与新 counts 不再对应，代码回退为不展开 token、直接按 counts split 原始输入。计算量（token 分配总数）仍正确，但 token 复制不精确——对速度 benchmark 无影响，不能用于精度验证。

**`model/transformer_lm.py` / `utils.py`**：`num_experts_per_tok` 沿 `_build_stage` → `TransformerLM` → `TransformerDecoderLayer` → `RandomGate` 穿透，默认 1。

**`main.py`**：新增 `--num_experts_per_tok` CLI 参数（默认 `None`），显式指定时优先于 model_config 默认值（走 `_user_specified` 机制）。

### 4.3 LoRA 适配 SwiGLU

**文件**：`model/lora.py` — `_apply_lora_to_transformer()`

原始代码只认 `inner.linear1 / linear2`。改为检测结构，兼容两种 expert：

```python
if hasattr(inner, 'gate_proj'):
    inner.gate_proj = LoRALinear(inner.gate_proj, ...)
    inner.up_proj   = LoRALinear(inner.up_proj, ...)
    inner.down_proj = LoRALinear(inner.down_proj, ...)
else:
    inner.linear1 = LoRALinear(inner.linear1, ...)
    inner.linear2 = LoRALinear(inner.linear2, ...)
```

LoRA 可训练参数：Qwen3.5 ~1.9B，DSV3 ~3.5B（rank=8, target=all）。

### 4.4 新增 Qwen3.5-397B 模型配置

**`utils.py`** — `model_config()` 新增 `QWEN35_397B` 分支，参数来自 sap4 `/mnt/data3/models/Qwen3.5-397B-A17B/config.json`：

```python
elif model_conf == 'QWEN35_397B':
    _set('embedding_dim', 4096)
    _set('hidden_dim', 1024)        # moe_intermediate_size
    _set('num_heads', 32)
    _set('num_layers', 60)
    _set('num_stages', 60)          # 1 layer per stage
    _set('num_experts', 512)
    _set('num_experts_per_tok', 10)
    _set('batch_size', 1)
    _set('num_chunks', 1)
    _set('seq_length', 512)
    _set('prefetch_portion', 0.6)
```

**`main.py`**：`--model_config` choices 加入 `QWEN35_397B`。

**`Static/lookup_table.py`**：新增 `LookupTable_QWEN35_397B`（APTMoE 的 R_solver 需要 expert 加载/CPU 计算耗时表来做 GPU/CPU 分配决策）。没有实测 profile，基于 `QWEN3_235B` 的表按 `moe_intermediate_size` 比例（1024/1536 ≈ 0.667）缩放 `load_expert` 和 `comp_cpu`；`load_MHA` 不变（hidden_size 同为 4096）；`load_Gate_512_experts` 从 384 外推为 0.0009。**这是估算值，只影响 APTMoE 内部的 expert 放置决策，不影响计时的正确性。**

**`Runtime/OffloadRuntime/R_solver.py`**：import 并接入新 lookup table；顺带删了几行注释掉的 debug print。

### 4.5 杂项改动（README 未记录，注意）

1. **`Runtime/OffloadRuntime/offload.py`：模块级 `prefetch_portion = 0.5 → 0.1`**
   这个全局变量控制 offload runtime 每个 stage 随机预载到 GPU 的 expert 比例（`random_stageload_list` 用），与 `args.prefetch_portion`（model_config 里的 0.6）是**两个独立变量**。降到 0.1 是因为 512-expert 模型预载 50% 会撑爆 GPU 显存。该文件修改时间（4/1）晚于 benchmark 日期（3/18~19），README 中的数据是否在此值下跑出未确认——重跑对比时留意。
2. **`main.py`：`--grad_accum` 是死参数**
   argparse 存到 `args.grad_accum`，但代码实际读的是 YAML config 里的 `gradient_accumulation_steps`，两者没接上。且 APTMoE pipeline 本身不支持 grad accumulation（见 6.5 限制 #3）。**不要用这个 flag。**

## 5. 在 fresh clone 上复现改动

```bash
git clone https://github.com/JimmyPeilinLi/APTMoE-baseline
cd APTMoE-baseline
patch -p1 < /path/to/aptmoe_changes.diff   # 本目录的 aptmoe_changes.diff
```

## 6. 测试方法

### 6.1 跑前检查清单（必做）

```bash
# 1. 杀干净残留进程（pipeline 异常退出常留僵尸进程占 GPU/RAM）
ssh sapphire4 "killall -9 python3.12; sleep 5"
# 2. 确认 GPU 全释放、RAM available > 1800GB
ssh sapphire4 "nvidia-smi | tail -20; free -g"
```

RAM 不干净直接跑会被 earlyoom/OOM killer 杀掉，且症状不明显（进程 silently killed）。

### 6.2 运行命令

```bash
cd /home/star/APTMoE-baseline

# Qwen3.5-397B（391B 合成），LoRA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /mnt/data2/hxx/mini/envs/sft/bin/torchrun --nproc_per_node 8 --master_port 29501 \
  ./main.py --is_moe=True --num_training_steps=3 --num_warmup_steps=1 \
  --model_config=QWEN35_397B --seq_length=1024 --num_experts_per_tok=2 \
  --gini=0.3 --topo=C1+G2 --pipeline=APTMoE \
  --lora --lora_rank=8 --lora_target=all \
  > /tmp/aptmoe_logs/qwen35_lora_top2_seq1024.log 2>&1

# DSV3（700B 合成），LoRA：把 --model_config=DSV3 --seq_length=256
# 全量训练：去掉 --lora 三个参数（仅 Qwen3.5 topk≤2 可跑，见 6.5）
```

参数说明：
- `--gini=0.3`：expert popularity 的 power-law 偏斜度（RandomGate 模拟真实 routing 的负载不均）
- `--topo=C1+G2`：APTMoE 的 CPU/GPU 拓扑配置
- `--num_warmup_steps=1 --num_training_steps=3`：1 步预热不计时 + 3 步计时
- 日志**必须重定向到文件**，不要管道 grep（pipeline 多进程输出会乱序/丢失）

### 6.3 计时与指标

- 计时段前后有 `torch.cuda.synchronize()`；每步 = 完整 forward + backward + `optimizer.step()`
- `training_time = wall_clock / num_training_steps`
- `TPS = batch_size × seq_length / training_time`（batch_size 恒为 1）
- 结果从日志尾部读 `training_time`；Peak GPU 看日志中的 memory 输出或另开 `nvidia-smi` 采样

### 6.4 测试矩阵

标准 sweep 维度（已跑过的组合与结果见 `README.md` 第 5 节）：

| 维度 | 取值 |
|------|------|
| 模型 | `QWEN35_397B` / `DSV3` |
| 训练方式 | LoRA（rank=8, target=all）/ 全量 |
| topk | 1 / 2 / 4 /（8, 仅 DSV3 会 OOM） |
| seq_length | 512 → 16384 翻倍递增（DSV3 从 256 起） |

每个组合跑到 OOM 为止。上限参考：Qwen3.5 LoRA top-1 到 8192、top-2/4 到 4096；DSV3 LoRA top-1 到 1024、top-4 只有 256。

### 6.5 已知限制与坑

1. **seq_length 必须整除 num_experts**（top-1 路径的断言）：Qwen3.5（512 experts）最小 seqlen=512，DSV3（256 experts）最小 256。
2. **`_user_specified` 陷阱**：`--seq_length=128`（等于 argparse 默认值）会被判定为"未指定"而被 model_config 覆盖。要用非默认值显式传参。
3. **Pipeline 不支持 grad accumulation**：每次 `run_pipeline()` 结束 experts 已 offload 回 CPU，第二轮需要完整 action list 重新 load。`--grad_accum` 参数也没接线（见 4.5）。
4. **DSV3 全量训练跑不了**：Adam optimizer state 需 ~5.2TB CPU RAM（机器只有 2016GB），初始化即失败。DSV3 只能 LoRA。
5. **Qwen3.5 全量训练范围有限**：optimizer state 占大量 RAM，topk≥4 或 seq≥2048 会被 OOM kill。
6. **失败模式区分**：CPU RAM 不足 → 进程被 kill（日志戛然而止，`dmesg` 可见 oom-kill）；GPU 显存不足 → 日志有 CUDA OOM traceback。记录结果时注明是哪种。

## 7. 结果摘要

完整数据见 `README.md` 第 5 节，关键结论：

- **APTMoE 的 step 时间几乎与 seqlen 无关**（Qwen3.5 LoRA 恒 ~27s，DSV3 ~42-62s）——offload 搬运是绝对瓶颈，TPS 随 seqlen 线性涨。topk 对速度影响也很小（LoRA 下 expert 权重不需要梯度，搬运量不变）。
- **vs ZeRO-3**（Qwen3.5 真实权重）：短 seq 打平，seq=4096/8192 时 APTMoE 快 2.1x/2.6x。
- **vs KT INT8**（DSV3）：seq=256 时 KT 95.4 TPS vs APTMoE top-4 5.8 TPS，**KT 快 16 倍**；seq=512 APTMoE 已 OOM。
- 注意对比口径：APTMoE 是合成权重+合成数据+简化结构（dense MHA、LayerNorm、无 shared expert），KT/ZeRO-3 是真实权重真实数据。APTMoE 的数字是它的**乐观估计**。

## 8. 原始日志索引（sap4:/tmp/aptmoe_logs/）

| 文件 | 内容 |
|------|------|
| `qwen35_lora_sweep.log` | **主结果**：Qwen3.5 LoRA topk=1,2,4 × seq=512-16384 |
| `qwen35_3linear_topk_probe.log` | Qwen3.5 全量 topk=1,2 seq=512 |
| `qwen35_3linear_topk_clean.log` | Qwen3.5 全量 topk=4,6,8,10 seq=512（全 fail） |
| `qwen35_topk2_seqsweep.log` | Qwen3.5 全量 topk=2 seq=512,1024,2048 |
| `qwen35_397b_topk10_sweep.log` | 旧版（2-linear）top-10 sweep，仅存档 |
| `dsv3_3linear_lora_v2.log` | DSV3 LoRA topk=1,2,4,8 seq=256 |
| `dsv3_lora_top4_sweep.log` | DSV3 LoRA topk=4 seq=256,512,1024 |
| `dsv3_lora_top1_sweep.log` | DSV3 LoRA topk=1 seq=512,1024,2048 |
| `dsv3_3linear_probe.log` | DSV3 全量 topk=1,2 seq=256（fail, RAM OOM） |

⚠️ `/tmp` 重启会清空，日志要长期保留的话尽快拷出来。
