# Progress of MLS：适配 KTransformers 的 Multi-LoRA Serving

日期：2026-08-06  
项目目录：`/mnt/data2/wbw/MLStest`  
代号：**MLS**（Multi-LoRA Serving）

本文说明如何为 **KTransformers / sglang-kt** 设计并落地同进程多块 composite LoRA 服务能力，以及配套的训练、转换与验证流水线。实现细节对照：

- 设计原文：[`kt-agent/doc/kt-multi-lora-serving/2026-08-06-MLS-design.md`](../../kt-agent/doc/kt-multi-lora-serving/2026-08-06-MLS-design.md)
- 前序背景：[`kt-agent/doc/kt-multi-lora-serving/previous-progress.md`](../../kt-agent/doc/kt-multi-lora-serving/previous-progress.md)
- 本仓库起服手册：[`task_bash_Qwen3.5-397B-A17B.md`](task_bash_Qwen3.5-397B-A17B.md)

---

## 1. 目标与范围

### 1.1 要解决的问题

SGLang 原生 Multi-LoRA 已覆盖 **GPU / non-expert** 路径（registry → manager → memory pool → CSGMV/Triton）。  
对 KTransformers 的 **CPU routed expert**，旧路径是：

```text
启动时绑定唯一一块 merged KT composite
  → kt_expert_lora_path 静态写入
  → 同进程无法按请求切换 expert LoRA
```

Macaron / Qwen3.5-MoE 类模型的「完整」adapter 同时包含：

| 部分 | 驻留 | 作用 |
| --- | --- | --- |
| non-expert LoRA（attention / shared 等） | GPU | 已有请求级 multi-LoRA |
| expert LoRA（routed MoE gate/up/down） | CPU（AMX） | 原先仅静态单块 |

MLS 要补齐的是：**同进程按请求在多块 KT composite 间切换**，使 `model="base:L0"` / `lora_path="L1"` 对 GPU 与 CPU expert **同时生效**。

### 1.2 里程碑切分

| 里程碑 | 能力 | 状态 |
| --- | --- | --- |
| **M1** | 同进程多 composite；**同批仅 1 种** adapter；batch 边界 `activate` | **已实现** |
| **M2** | 同批最多 N 种 adapter；Python grouped CPU expert；支撑 N 路 sub-agent 并发 | **已实现（grouped 路径）** |
| fused | AMX 多 slot、一次 forward 按 `token_slot` 选 A/B | 更远期 |

首版共同约束：

- `disable_cuda_graph`
- `kt_num_gpu_experts=0`
- 各 adapter 同 `rank` / `alpha` / `target_modules`
- `dp_size=1`；关闭 DeepEP / TBO / spec / overlap
- **不改 C++/AMX**（M1/M2 grouped 均可零改动 kernel）

### 1.3 首测模型

```text
Qwen3.5-397B-A17B
  architecture: Qwen3_5MoeForConditionalGeneration
  layers: 60, hidden: 4096
  routed experts / layer: 512, top-k: 10
  moe intermediate: 1024
```

底座路径：`/mnt/data2/models/Qwen3.5-397B-A17B`  
Serving 环境：conda **`kt-kernel`**  
Training 环境：conda **`Kllama`**（LLaMA-Factory editable + accelerate KT）

---

## 2. 设计原则

三条不可破坏约束：

1. **SGLang `LoRARegistry` 是唯一逻辑身份源**  
   请求里的 adapter 名 → `lora_id`；GPU 与 KT 都从同一身份解析，禁止旁路第二套命名。
2. **GPU slot 与 KT slot 独立，由 composite manager 原子协调**  
   GPU `LoRAManager` 继续管 non-expert；KT 侧新建 pool/manager；load/unload 按「KT 预备 → GPU load → KT commit」事务，失败可回滚。
3. **M2 起 token→adapter slot 必须是显式 forward metadata**  
   M1 不需要 token 级元数据；M2 再引入 `kt_lora_token_slots`，禁止用模块全局变量偷传。

M1 故意收窄语义：

```text
一个 batch 内 distinct adapters ≤ 1
  → 只需在 batch 边界 activate 一次 KT slot
  → 现有 submit_forward_inference / sync_forward_inference 签名不变
  → 零改动 AMX
```

```text
M1 一个 batch：
  [req_A(L0), req_B(L0)]  → activate L0 → 一次 CPU expert forward
  下一 batch 才能跑 L2

M2 一个 batch：
  [req_A(L0), req_B(L2)]
    → 按 slot 分组 → 多次 activate + forward → scatter
```

---

## 3. M1 架构

### 3.1 数据流

```text
请求 adapter 名（model="base:Lx" 或 lora_path="Lx"）
        │
        ▼
LoRARegistry
  · 唯一 lora_id
  · LoRARef 携带 source_lora_path / kt_expert_lora_path / adapter_kind / adapter_hash
        │
        ▼
Scheduler admission
  · GPU: distinct ≤ max_loras_per_batch
  · KT : distinct ≤ 1（M1）
  · 组合条件：gpu_ok ∧ kt_ok
        │
        ▼
┌──────────────────────────────────────────┐
│ GPU：LoRAManager.prepare_lora_batch      │  已有
│ KT ：KTCompositeLoRAManager.prepare_batch│  新增：解析 active kt_slot 并 activate
└──────────────────────────────────────────┘
        │
        ▼
KTEPWrapperMethod.submit / sync（签名不变）
```

不扩展模型里旧的静态 `qwen3_5.load_kt_lora` / `_kt_lora_*` 单路径绑定；多 adapter 时由 manager 拥有专家 LoRA 生命周期。

### 3.2 KT pool 与 activate

```text
slot 0：base-only（A/B 全零 / 无 expert delta）
slot 1..N：已加载 composite 的 expert 权重（CPU 常驻）

prepare_batch(forward_batch):
  unique = {id for id in lora_ids if id is not None}
  assert len(unique) ≤ 1
  target = 0 if empty else pool.get_slot(unique.pop())
  if target != active:
    wait CPUInfer idle
    for each KTEP layer: activate_kt_lora_slot(target)
    active = target
```

动态 load 事务：

```text
KT stage → GPU load_lora_adapter → KT commit
失败 → abort KT / unload GPU
```

卸载：registry 注销 → 等 refcount / CPUInfer → GPU unload → 回收 KT slot（`generation++` 防串台）。

### 3.3 代码落点（sglang-kt fork）

代码根：`/mnt/data2/wbw/ktransformers/third_party/sglang`

| 模块 | 改动摘要 |
| --- | --- |
| `srt/lora/lora_registry.py` | `LoRARef` 扩展 KT 路径与种类元数据 |
| `srt/server_args.py` | 允许多块 composite；强制 `max_loras_per_batch=1`；`--kt-max-loaded-loras` |
| `srt/lora/kt_composite_lora_manager.py` | **新建** `KTExpertLoRAPool` + `KTCompositeLoRAManager` |
| `srt/layers/moe/kt_ep_wrapper.py` | `register_kt_lora_slot` / `activate_kt_lora_slot`；managed 模式下延迟绑定 |
| `srt/model_executor/model_runner.py` | 初始化 KT manager；组合 load/unload |
| `srt/model_executor/forward_batch_info.py`（及 cuda graph runner） | GPU prepare 后调用 `kt_manager.prepare_batch` |
| `srt/managers/scheduler.py` | 组合 admission |

### 3.4 目标启动形态

```bash
python -m sglang.launch_server \
  --model-path /mnt/data2/models/Qwen3.5-397B-A17B \
  --enable-lora \
  --lora-paths L0=/path/L0 L1=/path/L1 \
  --max-loaded-loras 2 \
  --max-loras-per-batch 1 \
  --kt-max-loaded-loras 2 \
  --disable-cuda-graph \
  --kt-num-gpu-experts 0 \
  --kt-weight-path <verified-kt-weights> \
  --kt-method AMXBF16 \
  --lora-backend triton
```

请求侧：

```text
model = "Qwen3.5-397B-A17B:L0"
或
lora_path = "L0"
```

### 3.5 M1 验收清单

1. ≥2 块 composite 同进程加载，不再触发 “Only one merged KT composite”。
2. L0 / L1 交替请求与各自独立起服行为对齐。
3. base-only（无 lora / slot 0）无 expert delta。
4. 同批塞 2 个 adapter 时 admission 拒绝。
5. unload / reload 不串台（generation 递增）。
6. 首版不要求吞吐；不要求同批 mixed-token。

> 注：单元级 import / pool 逻辑已在 `kt-kernel` 环境验证；397B 全量 e2e 依赖真实 KT 权重包与至少两块 composite adapter。

---

## 4. M2：N 路 Sub-agent 并行（已实现代码路径）

目标语义：

```text
用户任务 → Orchestrator 拆成 N 条子请求（N≥2）
  → 并发 API，每条绑定一个 lora_path
  → 允许进入同一 forward batch
  → 各请求各自返回；serving 不合并多路输出
```

相对 M1：解除 `distinct adapters ≤ 1`；KT CPU expert 走 **token-grouped activate + forward + scatter**。

### 4.1 关键参数

```bash
--kt-lora-dispatch grouped          # M2；single 保持 M1
--kt-max-loras-per-batch N          # 默认 4；同批最多 N 种 adapter
--max-loras-per-batch >= N
--kt-max-loaded-loras >= N
```

### 4.2 实现落点

| 模块 | 改动 |
| --- | --- |
| `server_args.py` | `kt_lora_dispatch` / `kt_max_loras_per_batch`；grouped 时取消强制 `max_loras_per_batch=1` |
| `forward_batch_info.py` | `kt_lora_req_slots` / `kt_lora_token_slots` / `kt_lora_slot_generations` |
| `kt_composite_lora_manager.py` | `validate_batch` 允许多 distinct；`build_token_slots`；grouped `prepare_batch` |
| `kt_ep_wrapper.py` | `_cpu_forward_grouped`；`apply` 多 slot 走分组路径 |
| `scheduler.py` | 复用更新后的 `validate_batch`（无需改 admission 结构） |

### 4.3 验收与脚本

- 单元：[`scripts/test_m2_kt_manager_unit.py`](../Qwen3.5-397B-A17B/scripts/test_m2_kt_manager_unit.py)
- 并发 client：[`run_multi_lora_m2_client_concurrent.sh`](../Qwen3.5-397B-A17B/run_multi_lora_m2_client_concurrent.sh)
- 起服示例：`KT_LORA_DISPATCH=grouped` 或 `--kt-lora-dispatch grouped`

M2 之后才考虑 AMX fused multi-slot；grouped 为正确性路径。

---

## 5. Adapter 契约（bf16 composite）

Serving 侧每个 `--lora-paths` 条目必须是 **merged KT composite**，精度为 **bf16**：

```text
<ADAPTER>/
  adapter_config.json
  adapter_model.safetensors   # non-expert + 已展开的 expert LoRA（bf16）
```

训练侧 KT SFT 原始产物通常还带：

```text
<RUN>/
  adapter_config.json
  adapter_model.safetensors      # 主要是 non-expert
  fused_expert_lora.safetensors  # expert A/B，训练默认 bf16
```

转换：

```text
kt-kernel/scripts/convert_kt_to_sglang_adapter.py
  输入 run 目录 → 输出 merged composite
  只做 key 映射与合并，不转 int8；dtype 随训练权重透传（bf16）
```

注意区分：

| 配置 | 影响对象 | 与 adapter 精度关系 |
| --- | --- | --- |
| `accelerate ... AMXBF16` / `AMXINT8` | **底座** expert 训练/推理权重包 | 无关 |
| LoRA `bf16: true` + fused save | **adapter** | 决定 serving composite 为 bf16 |
| convert 脚本 | key 布局 | 不改变精度 |

错误底座示例（不可直接用于本 397B MLS）：Macaron GLM 系 L0–L3、缺 `fused_expert_lora` 的纯 PEFT 公开权重。

---

## 6. MLStest 工程布局

```text
/mnt/data2/wbw/MLStest/
├── docs/
│   ├── progress-of-MLS.md          # 本文
│   └── task_bash_Qwen3.5-397B-A17B.md
├── dataset/                        # HF 原始数据（可 gitignore）
├── lora-adapter/Qwen3.5-397B-A17B/ # convert 后的 serving composite
└── Qwen3.5-397B-A17B/
    ├── configs/default_env.sh
    ├── run_multi_lora_m1_serve.sh
    ├── run_multi_lora_m1_client.sh
    ├── run_multi_lora_m1_e2e.sh
    └── train/
        ├── run_prepare_data.sh
        ├── run_train_lora.sh
        ├── configs/                # yaml + accelerate 8gpu
        ├── data/                   # dataset_info + prepared jsonl
        └── scripts/                # prepare / convert
```

### 6.1 Serving 验证（M1）

| 脚本 | 作用 |
| --- | --- |
| `run_multi_lora_m1_serve.sh` | 按 M1 硬约束起服 |
| `run_multi_lora_m1_client.sh` | base / 各 adapter / 交替请求 smoke |
| `run_multi_lora_m1_e2e.sh` | 起服 → 就绪 → client → 停服 |

硬约束与默认：`max_loras_per_batch=1`、`kt_num_gpu_experts=0`、`disable_cuda_graph`、`lora-backend=triton`、`chunked-prefill-size=2048`。

### 6.2 训练 → 三块任务 LoRA

为 MLS 准备三块可切换的任务 adapter（同一 base、同 r=8）：

| 任务 | 数据集 | adapter 名 |
| --- | --- | --- |
| CUDA 编程 | `nvidia/Nemotron-SFT-CUDA-v1` | `cuda` |
| 软件工程 | `nvidia/Nemotron-SFT-SWE-v3` | `swe` |
| C++ 竞赛 | `Nemotron-SFT-Competitive-Programming-v2`（仅 cpp jsonl） | `cpp` |

流水线：

```text
HF 原始数据
  → prepare_nemotron_datasets.py（openai messages，tool→observation）
  → LLaMA-Factory KT LoRA SFT（8×GPU，use_kt: true，bf16）
  → convert_kt_to_sglang_adapter.py
  → lora-adapter/Qwen3.5-397B-A17B/{cuda,swe,cpp}
  → run_multi_lora_m1_serve.sh --lora-paths cuda=...,swe=...,cpp=...
```

默认训练入口：

```bash
cd /mnt/data2/wbw/MLStest/Qwen3.5-397B-A17B/train
bash run_prepare_data.sh          # 或 cuda / swe / cpp
bash run_train_lora.sh cuda       # 8 卡，整份 CUDA 集 1 epoch，产出 bf16 composite
```

---

## 7. 当前进度与待办

### 已完成

- [x] M1/M2 设计文档与前序进度归档  
- [x] M1 代码：registry / server_args / KT manager / KTEP activate / scheduler admission  
- [x] MLStest serving 脚本与环境约定（`kt-kernel`）  
- [x] 三数据集 → KT LoRA 训练 / convert 脚本包（`train/`）  
- [x] CUDA 数据集准备完成（约 2276 条 openai jsonl）
- [x] M2 grouped：server_args / token slots / manager / kt_ep_wrapper / 并发 client / 单元测试

### 进行中 / 阻塞

- [ ] SWE-v3、Competitive C++ 原始数据下载收尾  
- [ ] 三任务 LoRA 实训与 convert 产物落盘  
- [ ] 397B + 真实 KT 权重包的 M1/M2 e2e（≥2 adapter 交替与并发）  
- [ ] AMX fused multi-slot（M2 之后）
- [ ] **阻塞**：`kt_composite_lora_manager.py` 的 `tp_rank == 0` 限制导致 `--tp-size > 1` 时非 0 号 rank 从未激活 KT expert LoRA slot，首个请求即崩（见下方「已知 Bug」）；35B/397B 的 `--tp-size 8` M1 e2e 均无法验证，需 KT 团队修复或改用 `--tp-size 1` 绕过

### 风险备忘

- 397B KT LoRA 训练与起服均吃满 8×GPU + 大 CPU RAM；需避开冲突任务。  
- `--kt-weight-path` 必须与 base 对齐的已验证包；脚本不猜测路径。  
- Client smoke 目前是固定短 prompt，不是正式评测集。  
- 历史「只传 `model=L2`」不会触发 registry；必须用 `base:adapter` 或 `lora_path`。

### 已修复 Bug 记录（曾阻塞 M1 e2e，含 `--tp-size 8`；2026-08-07 三个 bug 已全部修复，e2e 已跑通）

**2026-08-07，Qwen3.5-35B-A3B `run_multi_lora_m1_e2e.sh --tp-size 8` 复现**：server 起服、`/health` 就绪探测均通过，但第一个真实请求（甚至不带 adapter 的 base 请求）在 forward 阶段崩溃，只有 **TP0** 一个进程抛异常（其余 7 个 TP 进程随之被 SIGQUIT 带崩，但本身没有报错）：

```text
RuntimeError: LoRA weights not initialized. Call init_lora_weights() first.
  kt_kernel/sft/base.py:500 _validate_forward_inputs
  <- kt_ep_wrapper.py:2920 _submit_cpu_forward
  <- kt_ep_wrapper.py:3009 _submit_with_staged_input
  <- kt_ep_wrapper.py:3225 apply
```

**排查记录（含一次已回滚的错误尝试）**：

1. 最初怀疑是 `kt_composite_lora_manager.py` 里 `initialize_from_server_args` / `activate_slot` / `stage` / `abort` / `_ensure_resident` 的 `tp_rank == 0` 限制导致非 0 号 rank 从未初始化 LoRA。据此改过一版（去掉全部 `tp_rank` 限制），改完复测发现 **7 个非 0 号 rank 反而在启动阶段就崩了**：
   ```
   RuntimeError: Managed KT expert LoRA layers must expose template rank/shape before base slot registration.
   ```
   顺藤摸瓜确认：`kt_ep_wrapper.py::create_weights()` 里 `self.wrapper`（真正的 CPU AMX 计算引擎）和 `self.kt_expert_lora_weights` 模板**本来就只在 `tp_rank == 0` 时创建**（`kt_kernel/sft/layer.py` 里也有明确注释 "Skip if wrapper is None (non-rank-0 processes)"）——即 CPU 侧 MoE 专家计算**设计上只由 TP0 一个进程承担**，其余 7 个 rank 的 `self.wrapper` 本来就是 `None`，天然不需要（也不能）注册/激活。也就是说 `kt_composite_lora_manager.py` 原来的 `tp_rank == 0` 限制是**符合设计的**，不是 bug。**已把这版改动完整回滚**，`kt_composite_lora_manager.py` 现在与本节最初发现问题时的状态一致。
2. 重新用原始（未改动）代码复测，日志证实：8 个 rank 在启动阶段都打印了 `KTCompositeLoRAManager ready: ...`（这条日志本来就在 `tp_rank` 判断之外，无条件打印，不代表真的做了注册/激活），随后**只有 TP0** 在处理第一个请求时抛出 `LoRA weights not initialized`，其余 7 个 rank 没有自己的错误日志，是被 TP0 崩溃后的 SIGQUIT 级联带走的。这与"只有 TP0 拥有真实 CPU 计算引擎"的架构完全吻合。
3. 真正的疑点缩小到：TP0 在启动阶段的 `initialize_from_server_args()` 里已经执行过 `_register_base_slot()` + `activate_slot(0)`（对应会调用到 `self.wrapper.init_lora_weights(...)`，把 `_lora_initialized` 设为 `True`），且期间没有抛异常、正常走到了 "Server ready"；但约 28 秒后处理第一个请求时，同一个 TP0 又在 `_validate_forward_inputs` 里读到 `_lora_initialized=False`。也就是说**初始化时明明成功设置过的标志位，到请求到来时又变回了未初始化**——这更像是 KT 侧（`kt_ep_wrapper.py` 与编译好的 `kt_kernel` C++ 扩展之间）一个尚未定位到具体行的时序/状态同步问题，而不是单纯的 Python 逻辑漏洞。受限于 `kt_kernel` 里 `TP_MOE_SFT` / `AMX_SFT_MOE_TP`（日志里能看到这两个类名）是编译产物，静态读代码走不下去了，需要跑起来加日志才能继续定位。

**根因确认（已修复）**：加 `SGLANG_KT_EXPERT_LORA_DEBUG=1` 复测后，server.log / stdout 里 `[KT expert LoRA debug]` 一次都没出现过——说明真正调用 `init_lora_weights()` 的代码路径从启动到崩溃从未被执行。定位到 `kt_composite_lora_manager.py::__init__`：

```136:kt_composite_lora_manager.py（修复前）
        self._active_slot: int = BASE_KT_LORA_SLOT   # = 0
```

`activate_slot()` 的短路逻辑 `if slot_id == self._active_slot: return` 会在**第一次**调用 `activate_slot(BASE_KT_LORA_SLOT)`（即 `activate_slot(0)`）时，因为 `0 == 0` 恒成立而直接跳过，根本不会执行 `method.activate_kt_lora_slot(slot_id)`（真正触发 `init_lora_weights()` 的地方）。结果：无论是全零的 base slot 还是后来加载的 `cuda` slot，都从未被真正初始化过——这解释了为什么连不带 adapter 的 `01_base` 请求都会崩。

**修复**：把 `_active_slot` 的初始值从 `BASE_KT_LORA_SLOT`（0，一个合法 slot 号）改成 `-1`（不会与任何真实 slot 冲突的哨兵值），确保第一次 `activate_slot()` 调用一定会真正执行：

```python
self._active_slot: int = -1
```

已确认此字段只在本文件内部使用（`activate_slot()` 的短路判断、赋值、以及 `initialize_from_server_args()` 末尾日志的 `%d` 格式化），没有被其它模块引用，改动是安全的、局部的。

- 影响范围：所有走 M1 composite LoRA（`--lora-paths` 指定了至少一个 `kt_composite` adapter）的起服都会命中——不区分 `--tp-size`、不区分 397B/35B，只是因为之前测试一直卡在更早的问题（CuDNN 检查、gate/shared_expert_gate 模块）上，这个 bug 直到今天下午才第一次被真正跑到。

**同一次调试还顺带修了两个已解决的问题**（详见 git 历史 / 本文件其余章节）：CuDNN 兼容性检查（`SGLANG_DISABLE_CUDNN_CHECK=1`）、composite adapter 里混入 sglang-kt 不支持的 `mlp.gate` / `mlp.shared_expert_gate` LoRA 模块（已从 397B/35B 的 `cuda` adapter 与训练 yaml 的 `lora_target` 中剔除）。

#### Bug 2（已修复）：GPU 侧 `down_proj` / `gate_up_proj` LoRA buffer 用错了 `intermediate_size` 字段

修完 `_active_slot` 后重测（`--tp-size 8`），`01_base` 请求成功返回；但带 `cuda` adapter 的第二个请求（`02_adapter_cuda`）让全部 8 个 TP 进程同时崩溃：

```text
AssertionError: LoRA buffer shape torch.Size([8, 704]) does not match weight shape torch.Size([8, 512]).
  sglang/srt/lora/mem_pool.py:395 load_lora_weight_tensor
```

排查过程：

1. 给 `mem_pool.py` 的 assert 临时加了 `module_name` / `layer_id` 调试信息，定位到出问题的是 **`down_proj`，`layer_id=0`**。
2. 手工按 `Qwen3.5-35B-A3B` 的 `config.json`（`text_config`）算 `get_hidden_dim("down_proj", ...)` 应有的各种候选公式，怎么算都凑不出 704；给 `get_lora_A_shape` 加调试打印后发现真相：`self.base_hf_config.intermediate_size` 实际读出来是 **`5632`**（`tp_size=8` 时 `5632/8=704`，正好对上），而不是 JSON 里没写、我以为会是 `None` 的值——说明 transformers 的配置类给 `intermediate_size` 塞了一个和这个 MoE 架构无关的 **默认值**。
3. 检查composite adapter 里 `nonexpert` 分支的真实张量名，确认 `down_proj` / `gate_proj` / `up_proj` 全部来自 `model.layers.N.mlp.shared_expert.*`（单独一份、不按 expert 编号区分的"共享专家" MLP，和放在 CPU 侧走 KT expert LoRA 路径的 256 个 routed experts 是两码事）。
4. 顺藤摸瓜到 `sglang/srt/models/qwen2_moe.py::Qwen2MoeSparseMoeBlock.__init__`：`self.shared_expert = Qwen2MoeMLP(..., intermediate_size=config.shared_expert_intermediate_size, ...)`——真正决定 `shared_expert.down_proj` 形状的字段是 **`shared_expert_intermediate_size`**（本模型里 = 512，和权重文件里的 512 完全吻合），根本不是 `config.intermediate_size`（那是给非 MoE 的 `qwen3_5_text` 稠密变体准备的、本模型用不到的默认值 5632）。

**修复**（`sglang/srt/models/qwen3_5.py::Qwen3_5ForCausalLM.get_hidden_dim`）：按 `config.model_type` 区分取值来源——MoE 变体（`qwen3_5_moe_text`）优先用 `shared_expert_intermediate_size`（回退 `moe_intermediate_size`），非 MoE 稠密变体（`qwen3_5_text`）保持原来的 `intermediate_size`（回退 `moe_intermediate_size`）不变：

```python
if config.model_type == "qwen3_5_moe_text":
    intermediate_size = getattr(config, "shared_expert_intermediate_size", None)
    if not intermediate_size:
        intermediate_size = getattr(config, "moe_intermediate_size", None)
else:
    intermediate_size = getattr(config, "intermediate_size", None)
    if intermediate_size is None:
        intermediate_size = getattr(config, "moe_intermediate_size", None)
```

影响范围：所有给 `qwen3_5_moe_text`（397B / 35B 皆同一套配置模式）训练了 `down_proj` / `gate_up_proj`（即 shared-expert）LoRA、且以 `--tp-size > 1` 起服的场景都会命中（`tp_size=1` 时 `5632` 和 `704`/`512` 的除法巧合不会触发——但形状仍然是错的，只是不一定会在 assert 里显形，需要一并排查）。

#### Bug 3（已修复）：非 0 号 TP rank 从未注册 KT expert LoRA 的 slot 映射

修完 Bug 2 后再测，`02_adapter_cuda` 请求把崩溃点往后推了一步，这次是 7 个非 0 号 TP rank（TP0 之外全部）在 `prepare_batch` 阶段报错：

```text
KeyError: 'KT expert LoRA id 04d36e65695d43139c14e273d00fa917 is not resident in the pool'
```

定位：`kt_composite_lora_manager.py::initialize_from_server_args()` 里把 `_register_base_slot()` / `_load_composite_into_pool()` / `activate_slot(BASE_KT_LORA_SLOT)` 整体包在 `if self.tp_rank == 0:` 里——这对"CPU expert 权重只由 TP0 计算"是对的，但 `KTExpertLoRAPool.lora_id_to_slot`（`lora_id → slot 号` 的轻量映射表，纯 Python 字典，不含任何权重数据）也被一起挡在了 TP0 之外。而 `build_token_slots()` / `_ensure_resident()` 是**每个 rank 每次前向都会跑**的路径（用于把请求的 `lora_id` 换算成 per-token 的 slot 号，写进 `forward_batch.kt_lora_token_slots`），非 0 号 rank 一读 `self.pool.lora_id_to_slot` 发现是空的，直接 `KeyError`。

**修复**：把"纯 slot 号 bookkeeping"和"CPU 端真实权重加载/激活"解耦——

- `_register_base_slot()` / `_load_composite_into_pool()` 现在**无条件在所有 TP rank 上调用**；内部改为在“这个 rank 没有真实 KT expert LoRA layer / 模板”时（`not self.layers` 或 `template is None`，非 0 号 rank 的常态，因为 `kt_ep_wrapper.py::create_weights()` 本来就只在 `tp_rank==0` 建真实模板）优雅跳过逐层加权重，只做 `pool.alloc_slot()` / `pool.mark_ready()` 这类轻量注册，不再 `raise RuntimeError`。
- `activate_slot(BASE_KT_LORA_SLOT)`（真正触发 CPU 端 `init_lora_weights()`）继续只在 `tp_rank == 0` 调用，未改动。
- `_ensure_resident()` 里原来 `if self.tp_rank == 0: self._load_composite_into_pool(ref)` 的判断也去掉了（现在所有 rank 都能安全调用，理由同上），避免运行期动态挂载 adapter 时非 0 号 rank 重蹈覆辙。

因为每个 rank 的 `composite_refs` 字典（顺序一致）和 `KTExpertLoRAPool` 构造参数完全相同，`alloc_slot()` 按"下一个空 slot"分配是确定性的，所有 rank 分配到的 slot 号天然保持一致，不需要额外的跨进程同步。

**验证**：清空 `cache/sglang_kt_lora/` 后重跑 `run_multi_lora_m1_e2e.sh --tp-size 8`，`01_base` / `02_adapter_cuda` / `alt_r01_cuda` 三个请求全部 200 返回，`E2E PASSED`。

- 影响范围：与 Bug 1 一样，所有走 M1 composite LoRA 的多卡（`--tp-size > 1`）起服都会命中；`--tp-size 1`（单卡）不受影响，因为那种情况下只有一个 TP rank，天然就是 `tp_rank==0`。

---

## 8. 一句话总结

MLS 在 **不改 AMX** 的前提下，把 KT CPU expert LoRA 从「启动绑死一块」提升为「同进程多块、批间切换（M1）」与「同批最多 N 块、token-grouped 正确混算（M2）」；GPU 侧继续复用 SGLang multi-LoRA。配套 MLStest 用同一 Qwen3.5-397B base 训出多块 **bf16 composite** adapter，再经 serving harness 验证请求级切换与 N 路并发。fused AMX 留到更后。
