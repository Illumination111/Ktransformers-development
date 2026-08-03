# Qwen3.5 KTransformers Shared Expert 缺失问题与最小修复方案

## 当前状态

截至 2026-07-31，本问题已完成定位，但修复尚未应用到
`/mnt/data2/wbw/ktransformers`。在完成下述修复和回归验证前：

- Qwen3.5-122B-A10B 的参数审计会在训练前终止；
- 现有 Qwen3.5-35B-A3B KTransformers Full-FT TPS 漏掉了 shared expert，
  不能作为完整 Qwen3.5 模型的有效 TPS；
- 不能通过降低期望参数量或关闭参数审计来绕过。

## 1. 122B 测试报错

失败运行：

```text
FFTtest/Qwen3.5-122B-A10B/test_log/
20260731_173222_KTRANSFORMERS_BF16_FULL_SWEEP/
server_8gpu_batch8/seq_32
```

训练尚未进入 optimizer step，七个非 owner rank 在模型加载后的 Full-FT 参数审计中
报告：

```text
Unexpected Qwen3.5-122B-A10B text-model parameter count:
got=121658394624
expected=122111526912
```

差值为：

```text
122,111,526,912 - 121,658,394,624 = 453,132,288
```

该差值精确等于 48 层全部 shared expert 和 shared-expert gate：

```text
每层 shared MLP = 3 × hidden_size × intermediate_size
                    = 3 × 3072 × 1024
                    = 9,437,184

每层 shared gate = hidden_size = 3,072
每层合计         = 9,440,256

48 × 9,440,256 = 453,132,288
```

日志中没有 CUDA OOM 或主机 OOM；这是模型结构被错误替换导致的确定性参数覆盖失败。

## 2. 根因

Transformers 的 `Qwen3_5MoeSparseMoeBlock` 使用 singular 属性：

```python
self.shared_expert
self.shared_expert_gate
```

其正确 shared 分支为：

```python
shared_output = self.shared_expert(hidden_states)
shared_output = (
    torch.sigmoid(self.shared_expert_gate(hidden_states))
    * shared_output
)
```

当前 KTransformers SFT wrapper 只识别 plural 属性：

```python
if moe_config.has_shared_experts and hasattr(original_moe, "shared_experts"):
    self.shared_experts = original_moe.shared_experts
else:
    self.shared_experts = None
```

Qwen3.5 不存在 `shared_experts`，所以原始 MoE 被 `KTMoELayerWrapper` 替换后：

- `shared_expert` 和 `shared_expert_gate` 没有注册到新 wrapper；
- 相关参数从模型树中消失；
- forward 不再计算 shared expert；
- backward 和 optimizer 也不会覆盖这些参数。

现有 Qwen3.5-35B-A3B 日志也显示同一问题：

```text
记录的 all params:     34,534,699,648
完整 text parameters:  34,660,610,688
差值:                    125,911,040
```

35B 配置的差值同样满足：

```text
40 × (3 × 2048 × 512 + 2048) = 125,911,040
```

因此旧 35B KTransformers TPS 是“不含 shared expert”的路径，不能直接用于完整模型的
性能结论。

## 3. 最小正确修复

最小修复只涉及 KTransformers 的两个 Python 文件，不需要修改 C++/AMX kernel。

### 3.1 保留两种 shared expert 命名

文件：

```text
ktransformers/kt-kernel/python/sft/layer.py
```

在 `KTMoELayerWrapper.__init__()` 中同时支持 DeepSeek/GLM 的 plural 命名和
Qwen 的 singular 命名：

```python
self.shared_experts = None
self.shared_expert = None
self.shared_expert_gate = None

if moe_config.has_shared_experts:
    if hasattr(original_moe, "shared_experts"):
        # DeepSeek/GLM
        self.shared_experts = original_moe.shared_experts
    elif hasattr(original_moe, "shared_expert"):
        # Qwen2/Qwen3/Qwen3.5
        self.shared_expert = original_moe.shared_expert
        self.shared_expert_gate = getattr(
            original_moe, "shared_expert_gate", None
        )
```

不要把 Qwen 的 `shared_expert` 同时注册成 `shared_experts` 别名，否则 state dict
可能出现重复或不兼容的参数名。

### 3.2 统一 shared expert 前向

在同一 wrapper 中增加：

```python
def _compute_shared_expert(
    self, hidden_states: torch.Tensor
) -> torch.Tensor | None:
    if self.shared_experts is not None:
        return self.shared_experts(hidden_states)

    if self.shared_expert is None:
        return None

    output = self.shared_expert(hidden_states)
    if self.shared_expert_gate is not None:
        output = (
            torch.sigmoid(self.shared_expert_gate(hidden_states))
            * output
        )
    return output
```

把分布式和单卡 `_submit_and_compute_gpu()` 中的：

```python
if self.shared_experts is not None:
    gpu_output = self.shared_experts(hidden_states)
```

统一替换为：

```python
gpu_output = self._compute_shared_expert(hidden_states)
```

shared 分支仍在每个 rank 的 GPU 上计算，并与 rank 0 CPU routed-expert 计算尽量并行。

### 3.3 补充设备迁移

文件：

```text
ktransformers/kt-kernel/python/sft/arch.py
```

在 `move_non_experts_to_gpu()` 中覆盖三种属性：

```python
for name in (
    "shared_experts",
    "shared_expert",
    "shared_expert_gate",
):
    module = getattr(moe_module, name, None)
    if module is not None:
        module.to(device)
```

修复后这些参数保持普通 PyTorch/FSDP 参数，不需要加入 KT CPU authoritative expert
buffer，也不需要修改 `get_kt_trainable_params()`。

### 3.4 不应采用的绕过方式

以下操作会让程序继续运行，但得到错误模型：

```python
EXPECTED_LOGICAL_PARAMETERS = 121_658_394_624
```

同样不能删除参数覆盖检查或把 shared expert 标记为非训练参数。正确期望值必须保持：

```python
EXPECTED_LOGICAL_PARAMETERS = 122_111_526_912
```

## 4. TPS 影响估算

### 4.1 参数和显存

缺失参数占完整文本模型：

```text
453,132,288 / 122,111,526,912 = 0.371%
```

但它们是每个 token 都执行的参数，不能按 0.371% 估算计算影响。

当前错误路径中，GPU/FSDP 管理的真实参数约为：

```text
121,658,394,624 - 115,964,116,992 = 5,694,277,632
```

补回 shared expert 后，GPU/FSDP 参数增加：

```text
453,132,288 / 5,694,277,632 = 7.96%
```

453,132,288 个 BF16 参数的原始容量约为 0.844 GiB；8 卡 FSDP 仅参数分片约为
0.106 GiB/卡。计入梯度和 optimizer states 后，预计增加约 0.4～0.8 GiB/卡，
具体取决于 FSDP master parameter 和 optimizer state dtype。

### 4.2 计算量

Qwen3.5 每层、每 token 执行：

```text
8 个 routed experts：Rank 0 CPU
1 个 shared expert：每张 GPU 本地执行
```

shared expert 与单个 routed expert 的 hidden/intermediate 维度相同，因此：

```text
相对错误路径的 MoE MLP FLOPs 增加：1 / 8 = 12.5%
shared expert 占正确 MoE MLP 计算：1 / 9 = 11.1%
```

122B 模型 shared expert 的前向计算约为：

```text
48 × 2 × 3 × 3072 × 1024 ≈ 0.906 GFLOP/token
```

按 backward 约为 forward 两倍估算，完整训练约增加 2.7 GFLOP/token。

### 4.3 端到端 TPS 范围

shared GPU 分支与 Rank 0 CPU routed-expert 分支在
`_submit_and_compute_gpu()` 中并行启动。如果 CPU routed experts 仍是关键路径，
大量 shared GPU 计算会被覆盖，因此端到端 TPS 不应直接下降 12.5%。

当前合理估计为：

| 场景 | 修复后相对错误路径的 TPS 下降 |
|---|---:|
| 8 卡 server，中长序列 | 约 1%～5% |
| 8 卡 server，短序列 | 约 3%～8% |
| 2 卡 consumer | 约 2%～8% |
| GPU/FSDP 通信无法有效重叠的保守上限 | 约 10%～15% |

这些数字是基于 FLOPs、参数规模和 CPU/GPU 重叠关系的估算，不是 122B 实测。正式
TPS 必须在修复后重新运行；旧 35B TPS 只能作为错误路径的诊断对照，不能作为正式
baseline。

## 5. 修复验收标准

至少完成以下检查后，才可恢复 Qwen3.5 TPS 测试：

1. CPU-only 单元测试确认 Qwen singular shared 分支输出与 Transformers 原实现一致；
2. backward 后 `shared_expert.{gate,up,down}_proj` 和
   `shared_expert_gate.weight` 均有非空梯度；
3. DeepSeek/GLM plural `shared_experts` 路径回归通过；
4. 122B 空模型参数量仍为 `122,111,526,912`；
5. KTransformers 包装后参数审计得到相同 logical total；
6. 48 个 KT wrapper 和 `115,964,116,992` 个 routed-expert placeholder 参数完整；
7. 8 卡 `seq=32, steps=1` smoke test 完成真实 optimizer step；
8. 修复前后用相同机器、序列、batch 和 timing 口径做 A/B，记录 TPS 和显存差异。
