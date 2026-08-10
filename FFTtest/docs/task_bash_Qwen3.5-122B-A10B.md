# Qwen3.5-122B-A10B Text-Only BF16 全量微调测试

测试目录：

```text
/mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B
```

当前提供三个后端，均只运行 server-side 基准：

| 后端 | 启动脚本 | Conda |
|---|---|---|
| KTransformers | `run_finetune_perf_test_bf16_ktransformers.sh` | `Kllama` |
| DeepSpeed ZeRO-3 + CPU offload | `run_finetune_perf_test_bf16_deepspeed.sh` | `Deepspeed` |
| MegaTrain | `run_finetune_perf_test_bf16_megatrain.sh` | `Megatrain` |

固定测试契约为 8 张 GPU、每卡 batch 1、全局 batch 8、BF16 Full-FT，默认依次测试
`32,64,128,256,512,1024,2048,4096`。每个 sequence length 启动一个独立训练
进程，分别记录 forward、backward、optimizer、完整 step、TPS、CPU 内存和 GPU 显存。

> KTransformers 曾因 singular `shared_expert` 未保留导致参数审计失败；修复说明见
> [`Qwen3.5-KTransformers-shared-expert-fix.md`](Qwen3.5-KTransformers-shared-expert-fix.md)。
> DeepSpeed / MegaTrain 不经过该 KT wrapper，但仍需通过相同的 text-only 参数量契约
> `122,111,526,912`。

## 1. 模型与加载契约

默认 checkpoint：

```text
/mnt/data2/models/Qwen3.5-122B-A10B
```

源 checkpoint 是 `Qwen3_5MoeForConditionalGeneration` 多模态模型。测试入口只提取
`text_config`，并构造 `Qwen3_5MoeForCausalLM`：

```text
logical text parameters: 122,111,526,912
decoder layers: 48
routed experts per layer: 256
activated routed experts per token: 8
```

视觉塔、`model.visual.*`、多模态 processor、`mtp.*` 以及图片、视频、音频字段均不进入
模型、前反向或 optimizer。预检会核对 checkpoint shard、关键结构参数、text-only 参数量、
Transformers 架构映射和数据集 token 长度。训练入口还会在 optimizer 创建前检查所有文本
参数的 Full-FT 覆盖；KTransformers 额外检查 48 个 MoE wrapper 与
`115,964,116,992` 个零存储 routed-expert placeholder。

## 2. 完整测试命令

命令可从任意目录执行。脚本没有 `both`、`server` 或 `consumer` profile 参数；server-side
契约是固定配置。

### KTransformers

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B/run_finetune_perf_test_bf16_ktransformers.sh \
  --devices 0,1,2,3,4,5,6,7 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B/test_log \
  --kt-distributed-checkpoint-reuse on \
  --continue-on-error
```

### DeepSpeed

使用 ZeRO-3、参数与 optimizer CPU offload，并强制 `DS_PROBE_MODE=off`。

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B/run_finetune_perf_test_bf16_deepspeed.sh \
  --devices 0,1,2,3,4,5,6,7 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B/test_log \
  --continue-on-error
```

### MegaTrain

默认 MegaTrain checkout 为 `/mnt/data2/wbw/MegaTrain`。

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B/run_finetune_perf_test_bf16_megatrain.sh \
  --devices 0,1,2,3,4,5,6,7 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B/test_log \
  --megatrain-root /mnt/data2/wbw/MegaTrain \
  --continue-on-error
```

## 3. 首次运行与 dry run

122B Text Full-FT 主机内存峰值可能接近或超过 1 TiB。首次运行建议对三个后端分别做
最短序列、单步 smoke test：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B/run_finetune_perf_test_bf16_ktransformers.sh \
  --seq-length 32 --devices 0,1,2,3,4,5,6,7 --steps 1 --warmup-steps 0

bash /mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B/run_finetune_perf_test_bf16_deepspeed.sh \
  --seq-length 32 --devices 0,1,2,3,4,5,6,7 --steps 1 --warmup-steps 0

bash /mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B/run_finetune_perf_test_bf16_megatrain.sh \
  --seq-length 32 --devices 0,1,2,3,4,5,6,7 --steps 1 --warmup-steps 0
```

只验证配置生成和最终启动命令、不加载模型时，追加 `--skip-dataset-check --dry-run`：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-122B-A10B/run_finetune_perf_test_bf16_megatrain.sh \
  --seq-length 32 \
  --steps 1 \
  --warmup-steps 0 \
  --skip-dataset-check \
  --dry-run
```

确认最短长度可完成一个 optimizer step 后，再依次测试更长序列。

## 4. 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--seq-lengths` | `32,64,128,256,512,1024,2048,4096` | 逗号分隔；保留输入顺序 |
| `--seq-length` | 未设置 | 仅运行一个 canonical 长度；与 `--seq-lengths` 互斥 |
| `--steps` | `15` | 每个长度的 optimizer steps |
| `--warmup-steps` | `5` | TPS 排除的前 N 个 step |
| `--gas` | `1` | Gradient accumulation steps |
| `--learning-rate` | `1.0e-5` | Full-FT 学习率 |
| `--cpu-threads` | 自动 | DeepSpeed/MegaTrain 每个训练 rank 的 CPU 线程数 |
| `--kt-owner-threads` | 自动 | 仅 KTransformers：rank 0 CPU MoE/optimizer 线程数 |
| `--devices` | `0..7` | 使用列表中的前 8 张 GPU |
| `--model-path` | `/mnt/data2/models/Qwen3.5-122B-A10B` | 本地 checkpoint |
| `--dataset-dir` | `FFTtest/dataset` | 数据集注册目录 |
| `--dataset-name` | `fft_real_100` | 注册数据集名称 |
| `--log-base` | 测试目录下 `test_log` | 结果根目录 |
| `--megatrain-root` | `/mnt/data2/wbw/MegaTrain` | 仅 MegaTrain |
| `--kt-distributed-checkpoint-reuse` | `on` | 仅 KTransformers |
| `--continue-on-error` | 关闭 | 一个长度失败后继续后续长度 |
| `--keep-model-output` | 关闭 | 保留最终模型输出 |
| `--skip-dataset-check` | 关闭 | 跳过 checkpoint/tokenizer/dataset 预检 |
| `--dry-run` | 关闭 | 只生成配置并打印命令 |

## 5. CPU 内存统计口径

每个 sequence case 通过 `systemd-run --user --scope` 放入独立 cgroup v2 scope，采样器
每秒读取一次该 scope 的 `memory.current`，并同时记录 `memory.swap.current` 以及
`memory.stat` 中的 anon、file 和 shmem。cgroup 数值是整个训练 scope 当时实际计费的
内存，不是线程 RSS 的叠加，因此不会因为同一地址空间被多个线程重复累计。

进程树 RSS 的采样代码和 `monitor.csv` 数据仍保留，便于诊断和兼容旧结果；但
`memory_summary.json`、`memory_summary.md` 和 sweep `summary.md` 的主 CPU 内存字段只采用
cgroup 峰值。`sweep_results.csv` 同时保留 `cgroup_memory_peak_gb` 和
`process_tree_peak_gb`，后者明确作为诊断数据，不用于三后端的内存结论。

该方式要求主机启用 cgroup v2 且当前用户具有可用的 systemd user manager。脚本会校验
训练进程确实进入预期的独立 scope；无法取得 cgroup 数据时测试会失败，而不是静默退回
RSS。1 秒一次的 `/sys/fs/cgroup` 文本读取和进程采样位于训练计时路径之外，预计对训练
吞吐的影响低于测量噪声。

## 6. 环境、计时与输出

默认环境：

```text
KTransformers: /mnt/data2/wbw/conda/envs/Kllama
DeepSpeed:     /mnt/data2/wbw/conda/envs/Deepspeed
MegaTrain:     /mnt/data2/wbw/conda/envs/Megatrain
LLaMA-Factory: /mnt/data2/wbw/LLaMA-Factory
MegaTrain root:/mnt/data2/wbw/MegaTrain
```

可分别通过 `FFT_CONDA_ENV`、`FFT_LLAMA_FACTORY_DIR`、`FFT_MEGATRAIN_ROOT` 覆盖。

稳定 TPS：

```text
tokens_per_step = GPU数 × 每卡batch × sequence length × GAS
stable TPS = tokens_per_step / 去除warmup后的平均step时间
```

KTransformers / DeepSpeed 记录训练 API 的 host-wall 时间，不主动插入
`torch.cuda.synchronize()`。MegaTrain 保留后端流水线必需的 CUDA 同步，并在
`step_timing.json` 中标记 `timing_mode=megatrain_host_wall_with_backend_cuda_sync`。
系统采样器运行在计时路径之外。

每个 sequence 目录包含：

```text
run_config.json
resource_contract.json
train_config.yaml          # KT / DeepSpeed
megatrain_config.yaml      # MegaTrain
accelerate_config.yaml     # 仅 KT
train.log
step_timing/step_timing.json
monitor.csv
memory_summary.json
memory_summary.md
exit_code.txt
```

Sweep 根目录会生成 `summary.md`、`sweep_results.csv` 和
`dataset_validation.json`。
