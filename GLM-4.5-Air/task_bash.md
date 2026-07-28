# GLM-4.5-Air KTransformers BF16 全量微调脚本调用与参数配置

GLM-4.5-Air 当前只提供 KTransformers 后端。server 与 consumer 共用一套参数解析、
sequence sweep、TPS 计时和资源观测逻辑。训练精度固定为原生 BF16，微调方式固定为
Full-FT，不能通过参数切换成 FP16、FP32 或 LoRA。

测试加载的模型架构必须是：

```text
model_type: glm4_moe
architecture: Glm4MoeForCausalLM
FSDP wrap class: Glm4MoeDecoderLayer
```

## 1. 一步启动完整测试

下面的命令可以在任意目录执行，无需先 `cd`：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data3/models/GLM-4.5-Air \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/GLM-4.5-Air/test_log \
  --kt-distributed-checkpoint-reuse on \
  --continue-on-error
```

该命令会先运行 server，再运行 consumer，不会同时运行两个 profile：

```text
server：
  GPU：8
  global batch：8
  sequence：32、64、128、256、512、1024、2048、4096

consumer：
  GPU：2
  global batch：2
  sequence：16、32、64、128、256、512、1024、2048

公共设置：
  每卡 batch：1
  每个 sequence：15 optimizer steps
  TPS 排除前 5 个 warmup steps
  gradient accumulation：1
  precision：BF16
  finetuning type：full
```

sequence 范围和顺序与当前 Qwen3.5 KTransformers 流程一致。每个 sequence 都使用
一套新的 Accelerate rank/process，当前 sequence 完成并退出后才启动下一个长度；
不会跨 sequence 保留 CUDA caching allocator 或最长 sequence 的 buffer。

`--kt-distributed-checkpoint-reuse on` 会在多卡非重入式 gradient-checkpoint 重算时
复用第一次 forward 生成的 CPU routed-expert 缓存。GPU attention、router 和其他
非 CPU MoE 部分仍按 checkpoint 语义重算。该开关默认就是 `on`，完整测试命令显式
写出是为了保证测试条件可复现。

训练配置固定使用：

```yaml
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
```

## 2. 只运行一个 Profile

### Server

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7 \
  --kt-distributed-checkpoint-reuse on
```

server 固定使用 8 张 GPU、global batch 8。默认 sequence 为：

```text
32,64,128,256,512,1024,2048,4096
```

### Consumer

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile consumer \
  --devices 6,7 \
  --kt-distributed-checkpoint-reuse on
```

consumer 固定使用 2 张 GPU、global batch 2。默认 sequence 为：

```text
16,32,64,128,256,512,1024,2048
```

consumer 不创建 benchmark cgroup、不设置内存上限，也不会在 CPU 峰值超过 1 TiB
时终止训练或自动判定 OOM。它会使用：

```text
numactl --interleave=0,1
```

并在运行结束后生成 CPU/GPU 内存曲线，供人工结合日志审阅。

## 3. Profile 固定配置

| Profile | GPU | 每卡 batch | global micro-batch | 每个 optimizer step 的有效 batch | 内存策略 |
|---|---:|---:|---:|---:|---|
| server | 8 | 1 | 8 | `8 × GAS` | 不设置 benchmark cgroup 上限 |
| consumer | 2 | 1 | 2 | `2 × GAS` | 不设置上限；双 NUMA interleave；人工审阅 1 TiB |

canonical 脚本不提供独立的 `--gpus` 或 `--batch-size` 参数，避免破坏 profile 定义。

## 4. 公共参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--profile` | `server` | `server`、`consumer` 或 `both` |
| `--seq-lengths` | 按 profile 选择 | 用逗号分隔；显式覆盖时保留输入顺序 |
| `--steps` | `15` | 每个 sequence 的 optimizer steps |
| `--warmup-steps` | `5` | 前 N 个 optimizer steps 不计入稳定 TPS |
| `--gas` | `1` | Gradient accumulation steps |
| `--learning-rate` | `1.0e-5` | Full-FT 学习率 |
| `--cpu-threads` | `2` | 每个 KT non-owner rank 的 CPU 线程数 |
| `--kt-owner-threads` | 自动计算 | global rank 0 的 CPU MoE/optimizer 线程数 |
| `--devices` | `0,1,2,3,4,5,6,7` | 物理 GPU 列表；profile 使用前 N 张 |
| `--model-path` | `/mnt/data3/models/GLM-4.5-Air` | GLM checkpoint 目录 |
| `--dataset-dir` | `FFTtest/dataset` | LLaMA-Factory 数据集目录 |
| `--dataset-name` | `fft_real_100` | `dataset_info.json` 中的数据集名称 |
| `--log-base` | GLM 目录下的 `test_log` | Sweep 结果根目录 |
| `--kt-distributed-checkpoint-reuse` | `on` | checkpoint 重算复用 CPU MoE forward |
| `--consumer-numa-nodes` | `0,1` | consumer 等权 interleave 的两个 NUMA 节点 |
| `--continue-on-error` | 关闭 | 某个 sequence 失败后继续剩余项 |
| `--keep-model-output` | 关闭 | 保留最终模型；默认跳过或清理完整权重输出 |
| `--skip-dataset-check` | 关闭 | 跳过模型、tokenizer 和数据长度校验 |
| `--dry-run` | 关闭 | 生成配置并打印命令，不启动训练 |

`--warmup-steps` 只控制 TPS 的统计排除窗口；训练配置中的学习率 warmup 固定为 0。

### 覆盖 Sequence Length

单 profile 临时诊断示例：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile consumer \
  --seq-lengths 128,256,512 \
  --steps 3 \
  --warmup-steps 1
```

覆盖值必须满足：

```text
server：32～4096，不允许 16
consumer：16～2048，不允许 4096
```

`--profile both` 只接受两种 profile 都合法的公共覆盖值，因此不能包含 16 或 4096。
完整对比测试不要传 `--seq-lengths`，让 server 和 consumer 分别使用自己的默认集合。

## 5. GPU 选择

Server 指定 8 张卡：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7
```

Consumer 指定 GPU 6、7：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile consumer \
  --devices 6,7
```

`--profile both` 需要提供至少 8 张卡。server 使用前 8 张，consumer 使用同一列表中的
前 2 张：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7
```

也支持外部 `CUDA_VISIBLE_DEVICES`，但显式 `--devices` 优先。

## 6. Conda 与 CPU 线程

训练默认使用：

```text
/mnt/data2/wbw/conda/envs/Kllama
```

可通过环境变量覆盖环境名称：

```bash
FFT_CONDA_ENV=Kllama \
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server
```

KTransformers 只有 global rank 0 创建 CPU MoE backend。脚本给 non-owner rank 各分配
2 个线程、给系统/NCCL 预留 2 个物理核心，其余可见物理核心交给 rank 0。以 96 个
可见物理核心为例：

```text
server：owner 80 + 7 × 2 non-owner = 94
consumer：owner 92 + 1 × 2 non-owner = 94
```

训练入口会根据 global rank 设置：

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
BLIS_NUM_THREADS
ACCELERATE_KT_OMP_NUM_THREADS
```

生成的 Accelerate 配置会同时把 `kt_num_threads` 设置为 owner 线程数。

显式覆盖示例：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --cpu-threads 2 \
  --kt-owner-threads 80
```

`--cpu-threads` 优先于 `FFT_CPU_THREADS`，`--kt-owner-threads` 优先于
`FFT_KT_OWNER_THREADS`。

资源监控器不参与训练后端。当前它沿用 Qwen3.5 流程，使用已有的 `Deepspeed` Conda
环境运行 `matplotlib`、`psutil` 和 `pynvml` 采样；模型加载、前向、反向和 optimizer
仍全部运行在 `Kllama` 环境中。

## 7. 计时与 TPS 口径

每个 optimizer step 只记录：

- forward host wall time；
- backward host wall time；
- optimizer host wall time；
- 完整 optimizer-step host wall time；
- 根据完整 step 时间计算的 TPS。

计时器不主动调用 `torch.cuda.synchronize()`，不启用 PyTorch profiler，不采集后端
内部细粒度事件，也不在每个 step 写文件。逐 step 数据先缓存在内存，训练结束后统一
写出。

脚本固定关闭：

```text
DS_PROBE_MODE=off
KT_BACKWARD_TIMING=off
KT_SFT_PROFILE=0
FFT_DISABLE_PERF_PROBES=1
```

稳定 TPS 公式：

```text
tokens_per_step = GPU数 × 每卡batch × sequence length × GAS

stable TPS = tokens_per_step
             / 去除 warmup 后的平均 optimizer-step 时间
```

CPU/GPU 采样器运行在计时器外部，因此 `monitor.csv` 不属于 step timing 的内部探针。

## 8. 配置文件和入口

- Canonical 启动脚本：[run_finetune_perf_test_bf16_ktransformers.sh](run_finetune_perf_test_bf16_ktransformers.sh)
- 默认兼容入口：[run_finetune_perf_test_bf16.sh](run_finetune_perf_test_bf16.sh)
- 旧名称兼容入口：[run_full_ft_test_1gpu_bf16.sh](run_full_ft_test_1gpu_bf16.sh)
- Full-FT 模板：[train_full_bf16_glm45_air.yaml](configs/train_full_bf16_glm45_air.yaml)
- KTransformers 8 卡：[accelerate_ktransformers_bf16_8gpu.yaml](configs/accelerate_ktransformers_bf16_8gpu.yaml)
- KTransformers 2 卡：[accelerate_ktransformers_bf16_2gpu.yaml](configs/accelerate_ktransformers_bf16_2gpu.yaml)
- 定时训练入口：[finetune_train_with_timing.py](finetune_train_with_timing.py)
- 模型/数据集校验：[validate_benchmark_dataset.py](validate_benchmark_dataset.py)
- Sweep 汇总器：[aggregate_sweep_results.py](aggregate_sweep_results.py)

每个 sequence 都会从基础模板生成最终的：

```text
train_config.yaml
accelerate_config.yaml
run_config.json
```

这些生成文件才是当次测试实际使用的最终参数。

## 9. Dry-run

正式测试前建议先执行：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --dry-run
```

默认情况下，dry-run 仍会检查模型目录、依赖、NUMA 节点，并运行模型/tokenizer/数据集
长度校验。只想检查命令生成时，可以显式跳过数据校验：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --skip-dataset-check \
  --dry-run
```

`--skip-dataset-check` 不会跳过模型目录存在性和环境依赖检查。

## 10. 输出目录

结果目录结构：

```text
test_log/<timestamp>_KTRANSFORMERS_BF16_FULL_SWEEP/
├── summary.md
├── sweep_results.csv
├── dataset_validation.json
├── server_8gpu_batch8/
│   └── seq_<length>/
│       ├── run_config.json
│       ├── train_config.yaml
│       ├── accelerate_config.yaml
│       ├── resource_contract.json
│       ├── train.log
│       ├── exit_code.txt
│       ├── monitor.csv
│       ├── memory_summary.{md,json}
│       ├── plots/
│       │   ├── 01_gpu_memory.png
│       │   └── 02_cpu_ram.png
│       └── step_timing/
│           ├── step_timing.json
│           ├── step_timing.csv
│           └── step_timing.md
└── consumer_2gpu_batch2/
    └── seq_<length>/（结构同上）
```

`summary.md` 和 `sweep_results.csv` 不要求同时运行两个 profile。脚本注册了 EXIT
finalizer；只要已经生成至少一个 `run_config.json`，正常结束、训练失败或提前退出时
都会尝试汇总当前已有结果。

## 11. 兼容入口

推荐始终使用 canonical 脚本：

```text
run_finetune_perf_test_bf16_ktransformers.sh
```

`run_finetune_perf_test_bf16.sh` 会直接转发全部参数。

旧的 `run_full_ft_test_1gpu_bf16.sh` 只用于兼容历史调用：

- `--phase4-steps N` 转换为 `--steps N`；
- `--gpu-ids LIST` 转换为 `--devices LIST`；
- `--gpus 8` 转换为 `--profile server`；
- `--gpus 2` 转换为 `--profile consumer`；
- `--only-phase4` 被忽略；
- `--skip-phase4` 没有 sequence sweep 等价语义，因此会直接报错。

旧的 `run_full_ft_test_4gpu_int8.sh` 仍是历史 AMXINT8 单长度流程，不属于本文描述的
BF16 server/consumer sweep。
