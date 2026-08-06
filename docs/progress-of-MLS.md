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

### 风险备忘

- 397B KT LoRA 训练与起服均吃满 8×GPU + 大 CPU RAM；需避开冲突任务。  
- `--kt-weight-path` 必须与 base 对齐的已验证包；脚本不猜测路径。  
- Client smoke 目前是固定短 prompt，不是正式评测集。  
- 历史「只传 `model=L2`」不会触发 registry；必须用 `base:adapter` 或 `lora_path`。

---

## 8. 一句话总结

MLS 在 **不改 AMX** 的前提下，把 KT CPU expert LoRA 从「启动绑死一块」提升为「同进程多块、批间切换（M1）」与「同批最多 N 块、token-grouped 正确混算（M2）」；GPU 侧继续复用 SGLang multi-LoRA。配套 MLStest 用同一 Qwen3.5-397B base 训出多块 **bf16 composite** adapter，再经 serving harness 验证请求级切换与 N 路并发。fused AMX 留到更后。
