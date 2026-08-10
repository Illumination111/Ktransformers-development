# Qwen3.5-35B-A3B Multi-LoRA Serving（M1）测试

测试目录：

```text
/mnt/data2/wbw/MLStest/Qwen3.5-35B-A3B
```

当前覆盖 **sglang-kt Multi-LoRA Serving M1**（同批 1 种 adapter）与可选 **M2**（`--kt-lora-dispatch grouped`，同批最多 N 种 adapter，支撑 N 路并发 sub-agent）。  
Conda 环境固定为 **`kt-kernel`**。

| 脚本 | 作用 |
|---|---|
| `run_multi_lora_m1_serve.sh` | 启动 multi-LoRA server（可用 `--kt-lora-dispatch grouped` 开 M2） |
| `run_multi_lora_m1_client.sh` | 对已启动服务做 base / 各 adapter / 交替请求 smoke |
| `run_multi_lora_m2_client_concurrent.sh` | N 路不同 adapter **并发**请求 smoke（不合并输出） |
| `run_multi_lora_m1_e2e.sh` | 起服 → 就绪探测 → client → 停服 |
| `configs/default_env.sh` | 默认路径与参数 |

设计对照：[`kt-agent/doc/kt-multi-lora-serving/2026-08-06-m1-m2-design.md`](/mnt/data2/wbw/kt-agent/doc/kt-multi-lora-serving/2026-08-06-m1-m2-design.md)。

> **已修复（2026-08-07，共 3 个 bug，`--tp-size 8` M1 e2e 现已跑通）**：
> 1. `kt_composite_lora_manager.py::__init__` 里 `self._active_slot` 初始值被错误地设成了 `BASE_KT_LORA_SLOT`（0），导致启动时第一次 `activate_slot(0)` 被自身的短路逻辑跳过，`init_lora_weights()` 从未被真正调用过——不管是 base slot 还是任何 composite adapter slot，第一个请求（哪怕不带 adapter）都会崩 `RuntimeError: LoRA weights not initialized`。已改成哨兵值 `-1` 修复。
> 2. `sglang/srt/models/qwen3_5.py::get_hidden_dim` 对 MoE 变体的 `down_proj` / `gate_up_proj`（实际对应 `mlp.shared_expert`）错误地用了 `config.intermediate_size`（本模型里是个无关默认值 5632），而不是真正决定 shared-expert 形状的 `config.shared_expert_intermediate_size`（512），导致 `--tp-size 8` 时 LoRA buffer 算出 704、和权重实际的 512 对不上，assert 崩溃。已按 `model_type` 区分取值来源修复。
> 3. `kt_composite_lora_manager.py` 里 `KTExpertLoRAPool.lora_id_to_slot`（`lora_id → slot` 号的轻量映射）被和"CPU 端真实权重加载"一起挡在了 `tp_rank == 0` 后面，导致非 0 号 TP rank 在处理带 adapter 的请求时 `KeyError: ... is not resident in the pool`。已把"slot 号 bookkeeping"和"CPU 端真实权重加载/激活"解耦，前者在所有 rank 上都跑（没有真实权重时优雅跳过），后者仍只在 `tp_rank==0` 执行。
>
> 三个 bug 的完整排查记录与修复代码见 [`progress-of-MLS.md`](progress-of-MLS.md#已修复-bug-记录曾阻塞-m1-e2e含---tp-size-82026-08-07-三个-bug-已全部修复e2e-已跑通)。

## 1. 模型与加载契约

默认模型入口：

```text
/mnt/data3/models/Qwen3.5-35B-A3B
```

训练 adapter 的 `base_model_name_or_path` 与 serving `--model-path` 均使用该目录。

关键结构参数：

```text
architecture: Qwen3_5MoeForConditionalGeneration (text MoE)
decoder layers: 40
hidden size: 2048
routed experts / layer: 256
experts / token: 8
moe intermediate size: 512
```

`--kt-weight-path` 必须指向**已验证**的该 base 对应 KT CPU expert 权重包；脚本不会猜测路径。  
`--lora-paths` 中每个 adapter 必须是 **merged KT composite** 目录，且包含：

```text
<ADAPTER>/
  adapter_config.json
  adapter_model.safetensors
```

M1 硬约束：

```text
--max-loras-per-batch 1
--kt-num-gpu-experts 0
--disable-cuda-graph
--lora-backend triton          # 397B dense LoRA 正确性基线（与 397B 案例相同后端）
--chunked-prefill-size 2048    # 首轮建议，勿一次灌入超大 prefill
```

## 2. 环境

```text
Conda:            /mnt/data2/wbw/conda/envs/kt-kernel
KTransformers:    /mnt/data2/wbw/ktransformers
sglang-kt python: .../third_party/sglang/python
kt-kernel python: .../kt-kernel/python
```

脚本会设置：

```bash
PYTHONPATH=<kt-kernel/python>:<third_party/sglang/python>:$PYTHONPATH
```

并优先使用 `/mnt/data2/wbw/conda/envs/kt-kernel/bin/python`。

可用环境变量覆盖：`MLS_CONDA_ENV`、`KTRANSFORMERS_ROOT`、`MODEL_PATH`、`KT_WEIGHT_PATH`、`LORA_PATHS`、`PORT`、`LOG_BASE` 等（见 `configs/default_env.sh`）。

## 3. 完整测试命令

命令可从任意目录执行。将下面的路径替换为真实 KT 权重与至少两块 composite adapter。

### 3.1 Dry run（不加载模型）

```bash
bash /mnt/data2/wbw/MLStest/Qwen3.5-35B-A3B/run_multi_lora_m1_serve.sh \
  --kt-weight-path /path/to/verified-qwen35-397b-kt-weights \
  --lora-paths L0=/path/to/L0,L1=/path/to/L1 \
  --devices 0,1,2,3,4,5,6,7 \
  --tp-size 8 \
  --port 31007 \
  --dry-run
```

### 3.2 仅起服

```bash
bash /mnt/data2/wbw/MLStest/Qwen3.5-35B-A3B/run_multi_lora_m1_serve.sh \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --kt-weight-path /path/to/verified-qwen35-397b-kt-weights \
  --kt-method AMXBF16 \
  --lora-paths L0=/path/to/L0,L1=/path/to/L1 \
  --devices 0,1,2,3,4,5,6,7 \
  --tp-size 8 \
  --host 127.0.0.1 \
  --port 31007 \
  --served-model-name Qwen3.5-35B-A3B \
  --max-loaded-loras 2 \
  --kt-max-loaded-loras 2 \
  --max-lora-rank 8 \
  --chunked-prefill-size 2048 \
  --max-running-requests 2 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-numa-nodes "0 1"
```

### 3.3 仅 client（服务已在跑）

```bash
bash /mnt/data2/wbw/MLStest/Qwen3.5-35B-A3B/run_multi_lora_m1_client.sh \
  --host 127.0.0.1 \
  --port 31007 \
  --served-model-name Qwen3.5-35B-A3B \
  --adapters L0,L1 \
  --rounds 2 \
  --max-tokens 64
```

### 3.4 端到端

```bash
bash /mnt/data2/wbw/MLStest/Qwen3.5-35B-A3B/run_multi_lora_m1_e2e.sh \
  --kt-weight-path /path/to/verified-qwen35-397b-kt-weights \
  --lora-paths L0=/path/to/L0,L1=/path/to/L1 \
  --devices 0,1,2,3,4,5,6,7 \
  --tp-size 8 \
  --port 31007 \
  --adapters L0,L1 \
  --rounds 2 \
  --ready-timeout-sec 1800
```

## 4. 首次运行建议（分阶段）

对照进度文档 Q0～Q2：

| 阶段 | 目标 | 建议 |
|---|---|---|
| Q0 | base-only 可启动 | 暂不传 `--lora-paths`，单独手写 base launch；记录内存基线 |
| Q1 | 单 adapter | `--lora-paths L0=/path/L0 --max-loaded-loras 1` |
| Q2 | 两 adapter resident，同批 1 | `--lora-paths L0=...,L1=... --max-loras-per-batch 1` + client 交替 |

启动前硬门槛：

1. 至少两块带真实 `adapter_model.safetensors` 的 397B composite adapter。  
2. `--kt-weight-path` 已在本机 base-only 或单 adapter 路径验证。  
3. 397B 体量大，首次建议先 `--dry-run`，再短 prompt / 小 `max_tokens` smoke。  
4. 首轮 `chunked-prefill-size` 用 `2048`，稳定后再试 `4096`。

## 5. 参数

### 5.1 `run_multi_lora_m1_serve.sh`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-path` | `/mnt/data3/models/Qwen3.5-35B-A3B` | HF checkpoint 入口 |
| `--tokenizer-path` | 同 model-path | tokenizer |
| `--kt-weight-path` | **必填** | 已验证 KT CPU expert 权重 |
| `--kt-method` | `AMXBF16` | `AMXBF16` / `AMXINT4` / `AMXINT8` / `BF16` |
| `--lora-paths` | **必填** | `L0=/p0,L1=/p1` 逗号分隔 |
| `--devices` | `0,1,2,3,4,5,6,7` | 物理 GPU；取前 `--tp-size` 张 |
| `--tp-size` | `8` | tensor parallel |
| `--host` / `--port` | `127.0.0.1` / `31007` | 监听地址 |
| `--served-model-name` | `Qwen3.5-35B-A3B` | OpenAI `model` 前缀 |
| `--max-loaded-loras` | `2` | GPU/non-expert 池容量 |
| `--kt-max-loaded-loras` | 同 max-loaded | KT CPU expert 池容量（不含 base slot） |
| `--max-lora-rank` | `8` | 与 adapter `r` 一致 |
| `--chunked-prefill-size` | `2048` | KT staging 相关 |
| `--max-running-requests` | `2` | 调度并发请求上限 |
| `--max-total-tokens` | `8192` | KV / token 预算 |
| `--context-length` | `8192` | 服务侧上下文上限（可按需加大） |
| `--kt-cpuinfer` | `96` | KT CPU 线程 |
| `--kt-threadpool-count` | `2` | KT threadpool |
| `--kt-numa-nodes` | `0 1` | 传参时请加引号：`"0 1"` |
| `--attention-backend` | `flashinfer` | attention 后端 |
| `--mem-fraction-static` | `0.90` | 静态显存比例 |
| `--log-base` | `.../test_log` | 结果根目录 |
| `--conda-env` | `kt-kernel` | conda 环境名 |
| `--dry-run` | 关闭 | 只写配置并打印命令 |

脚本会强制 `--max-loras-per-batch 1`、`--disable-cuda-graph`、`--kt-num-gpu-experts 0`。

### 5.2 `run_multi_lora_m1_client.sh`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` / `--port` | 同 serve | 服务地址 |
| `--served-model-name` | `Qwen3.5-35B-A3B` | base 模型名 |
| `--adapters` | 从 `--lora-paths` 推导 | 如 `L0,L1` |
| `--rounds` | `2` | 交替轮数 |
| `--max-tokens` | `64` | 生成长度 |
| `--temperature` | `0.0` | 采样温度 |
| `--prompt` | 固定中文短句 | 用户内容 |
| `--timeout-sec` | `300` | 单次 curl 超时 |
| `--log-dir` | 自动 | 请求/响应落盘目录 |
| `--dry-run` | 关闭 | 只打印计划请求 |

请求语义：

```text
model=Qwen3.5-35B-A3B
  => base（KT expert slot 0 / zero delta）

model=Qwen3.5-35B-A3B:L0
  => base + composite L0（GPU non-expert + KT expert）

model=Qwen3.5-35B-A3B:L1
  => base + composite L1
```

冒号后名称必须与 `--lora-paths` 左侧注册名一致。

### 5.3 `run_multi_lora_m1_e2e.sh`

在 serve 参数之外额外支持：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ready-timeout-sec` | `1800` | 等待 `/v1/models` 或 `/health` |
| `--adapters` / `--rounds` / ... | 同 client | 转发给 client |
| `--dry-run` | 关闭 | serve+client 均 dry-run |

## 6. 输出与验收

每次 serve 在 `test_log/<run_id>/` 写入：

```text
run_config.json
launch_cmd.txt
server.log
```

client / e2e 额外写入：

```text
01_base.json
02_adapter_L0.json
...
alt_r01_L0.json
summary.md
exit_code.txt          # e2e
```

M1 通过条件（本套脚本覆盖的部分）：

1. 同进程加载 ≥2 块 composite，不再触发 “Only one merged KT composite”。  
2. `model=...:L0` 与 `model=...:L1` 均可返回非空 content。  
3. 交替请求序列不崩溃（验证 batch 边界切换路径可走通）。  
4. base 模型名可请求。

本脚本**不**自动做：logits 与独立起服数值对齐、同批双 adapter admission 拒绝压测、显存/CPU 峰值曲线。这些需手工对照或后续加专用 harness。

## 7. 与 M2 的边界

| | M1（本目录） | M2（未实现） |
|---|---|---|
| 同批 adapter 数 | 1 | ≥2 |
| client 测法 | 顺序交替 | 需同批 mixed-token / 并发构造 |
| 参数 | 无 `--kt-lora-dispatch` | 目标含 `grouped` / `fused` |

不要把 `max_loras_per_batch` 直接改成 2 并期望本脚本验证同批正确性；那是 M2 范围。


## 训练 LoRA（三个 Nemotron 数据集）

脚本与 397B 案例同构，目录：`Qwen3.5-35B-A3B/train/`。

```bash
cd /mnt/data2/wbw/MLStest/Qwen3.5-35B-A3B/train

# 原始数据仍用仓库根目录 dataset/；若 397B 已 prepare，可直接复用 jsonl：
#   SKIP_PREPARE=1 bash run_train_lora.sh cuda
# 否则先转换：
bash run_prepare_data.sh cuda   # 或 swe / cpp / 全部

bash run_train_lora.sh cuda
bash run_train_lora.sh swe
bash run_train_lora.sh cpp
# 或
bash run_train_lora.sh all
```

产出：
- runs：`train/runs/nemotron_{cuda,swe,cpp}/`
- serving composite：`lora-adapter/Qwen3.5-35B-A3B/{cuda,swe,cpp}/`

默认端口 **31007**（避免与 397B 的 31006 冲突）。
