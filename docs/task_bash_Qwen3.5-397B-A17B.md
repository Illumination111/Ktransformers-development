# Qwen3.5-397B-A17B Multi-LoRA Serving（M1）测试

测试目录：

```text
/mnt/data2/wbw/MLStest/Qwen3.5-397B-A17B
```

当前只覆盖 **sglang-kt Multi-LoRA Serving M1**（同进程多块 composite LoRA，同批仅 1 种 adapter）。  
Conda 环境固定为 **`kt-kernel`**。M2（同批 mixed-token）不在本套脚本范围内。

| 脚本 | 作用 |
|---|---|
| `run_multi_lora_m1_serve.sh` | 启动 multi-LoRA server |
| `run_multi_lora_m1_client.sh` | 对已启动服务做 base / 各 adapter / 交替请求 smoke |
| `run_multi_lora_m1_e2e.sh` | 起服 → 就绪探测 → client → 停服 |
| `configs/default_env.sh` | 默认路径与参数 |

设计对照：[`kt-agent/doc/kt-multi-lora-serving/2026-08-06-m1-m2-design.md`](/mnt/data2/wbw/kt-agent/doc/kt-multi-lora-serving/2026-08-06-m1-m2-design.md)。

## 1. 模型与加载契约

默认模型入口：

```text
/mnt/data2/models/Qwen3.5-397B-A17B
```

训练 adapter 的 `base_model_name_or_path` 与 serving `--model-path` 均使用该目录。

关键结构参数：

```text
architecture: Qwen3_5MoeForConditionalGeneration (text MoE)
decoder layers: 60
hidden size: 4096
routed experts / layer: 512
experts / token: 10
moe intermediate size: 1024
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
--lora-backend triton          # 397B dense LoRA 正确性基线
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
bash /mnt/data2/wbw/MLStest/Qwen3.5-397B-A17B/run_multi_lora_m1_serve.sh \
  --kt-weight-path /path/to/verified-qwen35-397b-kt-weights \
  --lora-paths L0=/path/to/L0,L1=/path/to/L1 \
  --devices 0,1,2,3,4,5,6,7 \
  --tp-size 8 \
  --port 31006 \
  --dry-run
```

### 3.2 仅起服

```bash
bash /mnt/data2/wbw/MLStest/Qwen3.5-397B-A17B/run_multi_lora_m1_serve.sh \
  --model-path /mnt/data2/models/Qwen3.5-397B-A17B \
  --kt-weight-path /path/to/verified-qwen35-397b-kt-weights \
  --kt-method AMXBF16 \
  --lora-paths L0=/path/to/L0,L1=/path/to/L1 \
  --devices 0,1,2,3,4,5,6,7 \
  --tp-size 8 \
  --host 127.0.0.1 \
  --port 31006 \
  --served-model-name Qwen3.5-397B-A17B \
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
bash /mnt/data2/wbw/MLStest/Qwen3.5-397B-A17B/run_multi_lora_m1_client.sh \
  --host 127.0.0.1 \
  --port 31006 \
  --served-model-name Qwen3.5-397B-A17B \
  --adapters L0,L1 \
  --rounds 2 \
  --max-tokens 64
```

### 3.4 端到端

```bash
bash /mnt/data2/wbw/MLStest/Qwen3.5-397B-A17B/run_multi_lora_m1_e2e.sh \
  --kt-weight-path /path/to/verified-qwen35-397b-kt-weights \
  --lora-paths L0=/path/to/L0,L1=/path/to/L1 \
  --devices 0,1,2,3,4,5,6,7 \
  --tp-size 8 \
  --port 31006 \
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
| `--model-path` | `/mnt/data2/models/Qwen3.5-397B-A17B` | HF checkpoint 入口 |
| `--tokenizer-path` | 同 model-path | tokenizer |
| `--kt-weight-path` | **必填** | 已验证 KT CPU expert 权重 |
| `--kt-method` | `AMXBF16` | `AMXBF16` / `AMXINT4` / `AMXINT8` / `BF16` |
| `--lora-paths` | **必填** | `L0=/p0,L1=/p1` 逗号分隔 |
| `--devices` | `0,1,2,3,4,5,6,7` | 物理 GPU；取前 `--tp-size` 张 |
| `--tp-size` | `8` | tensor parallel |
| `--host` / `--port` | `127.0.0.1` / `31006` | 监听地址 |
| `--served-model-name` | `Qwen3.5-397B-A17B` | OpenAI `model` 前缀 |
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
| `--served-model-name` | `Qwen3.5-397B-A17B` | base 模型名 |
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
model=Qwen3.5-397B-A17B
  => base（KT expert slot 0 / zero delta）

model=Qwen3.5-397B-A17B:L0
  => base + composite L0（GPU non-expert + KT expert）

model=Qwen3.5-397B-A17B:L1
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
