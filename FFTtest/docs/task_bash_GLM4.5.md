# GLM-4.5-Air BF16 全量微调测试

> Qwen3.5-122B-A10B 的 text-only KTransformers 测试已独立放在
> [`task_bash_Qwen3.5-122B-A10B.md`](task_bash_Qwen3.5-122B-A10B.md)，
> 默认模型路径为 `/mnt/data2/models/Qwen3.5-122B-A10B`。本文其余内容仍只描述
> GLM-4.5-Air。

本目录提供三个后端的 server/consumer sequence sweep：

- KTransformers
- DeepSpeed ZeRO-3 optimizer CPU offload
- MegaTrain CPU master / layer streaming

三套测试固定使用原生 BF16 和 Full-FT，不支持在这些 canonical 脚本中切换
FP16、FP32 或 LoRA。加载契约为：

```text
model_type: glm4_moe
architecture: Glm4MoeForCausalLM
logical trainable parameters: 106,852,245,504
```

## 1. 完整测试启动命令

命令可以在任意目录执行。`--profile both` 会先运行 server，再运行 consumer，
不会同时运行两个 profile。

### KTransformers

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

### DeepSpeed

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_deepspeed.sh \
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
  --continue-on-error
```

DeepSpeed 使用
[`configs/deepspeed_zero3_offload_bf16.json`](../GLM-4.5-Air/configs/deepspeed_zero3_offload_bf16.json)：

```text
ZeRO stage 3
parameter offload: disabled
optimizer offload: CPU
weights / gradients / optimizer states: BF16
```

### MegaTrain

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_megatrain.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data3/models/GLM-4.5-Air \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --megatrain-root /mnt/data2/wbw/MegaTrain \
  --log-base /mnt/data2/wbw/FFTtest/GLM-4.5-Air/test_log \
  --continue-on-error
```

MegaTrain 在构造 optimizer 前检查 `CPUMasterModel.get_parameters()` 是否完整覆盖
GLM checkpoint 的全部 106,852,245,504 个参数。若模型结构发现遗漏了任何参数，
测试会在训练前失败，不能产生有效 TPS。

## 2. Profile 固定配置

```text
server:
  GPU: 8
  global batch: 8
  per-device batch: 1
  sequence: 32,64,128,256,512,1024,2048,4096

consumer:
  GPU: 2
  global batch: 2
  per-device batch: 1
  sequence: 16,32,64,128,256,512,1024,2048

common:
  optimizer steps per sequence: 15
  warmup steps excluded from TPS: 5
  gradient accumulation: 1
  precision: BF16
  finetuning type: full
```

每个 sequence 都启动独立训练进程；前一个进程完全退出后才运行下一个长度。
不会跨 sequence 保留 CUDA caching allocator 或最长 sequence 的 buffer。

consumer 不创建 benchmark cgroup，也不会在 CPU 峰值超过 1 TiB 时自动终止。
它固定使用：

```text
numactl --interleave=0,1
```

内存曲线和峰值会写入结果目录，OOM 状态保留人工审阅。

## 3. 单 Profile、单 Server sequence 与最小诊断

KTransformers 可以通过 `--server-seq-length` 只运行一个 Server sequence。
例如只测试 sequence length 4096：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --server-seq-length 4096 \
  --devices 0,1,2,3,4,5,6,7 \
  --steps 15 \
  --warmup-steps 5
```

`--server-seq-length` 仅适用于 KTransformers 的 `server` profile，可选值为
`32`、`64`、`128`、`256`、`512`、`1024`、`2048`、`4096`。它与
`--seq-lengths` 互斥；启用后只创建并运行对应的一个 `seq_<N>` 目录，其他
Server sequence 不会执行。

下面用 DeepSpeed 演示；把脚本名换为另外两个后端即可。

只运行 server：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7
```

只运行 consumer：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile consumer \
  --devices 6,7
```

建议新后端第一次运行先做单长度、单步 smoke test：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7 \
  --seq-lengths 32 \
  --steps 1 \
  --warmup-steps 0
```

MegaTrain 对应命令：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_megatrain.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7 \
  --seq-lengths 32 \
  --steps 1 \
  --warmup-steps 0
```

覆盖 sequence 时保留输入顺序，并遵守：

```text
server: 32～4096，不允许 16
consumer: 16～2048，不允许 4096
```

`--profile both` 的覆盖值必须同时适用于两种 profile，所以不能包含 16 或 4096。

## 4. 公共参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--profile` | `server` | `server`、`consumer` 或 `both` |
| `--seq-lengths` | 按 profile | 逗号分隔，保留输入顺序 |
| `--steps` | `15` | 每个 sequence 的 optimizer steps |
| `--warmup-steps` | `5` | 不计入稳定 TPS 的前 N 步 |
| `--gas` | `1` | Gradient accumulation steps |
| `--learning-rate` | `1.0e-5` | Full-FT 学习率 |
| `--cpu-threads` | 自动 | DeepSpeed/MegaTrain 默认将物理核均分到 GPU |
| `--devices` | `0..7` | 物理 GPU 列表；profile 使用前 N 张 |
| `--model-path` | `/mnt/data3/models/GLM-4.5-Air` | GLM checkpoint |
| `--dataset-dir` | `FFTtest/dataset` | 数据集注册目录 |
| `--dataset-name` | `fft_real_100` | `dataset_info.json` 中的名称 |
| `--log-base` | 本目录下 `test_log` | Sweep 结果目录 |
| `--consumer-numa-nodes` | `0,1` | consumer interleave 节点 |
| `--continue-on-error` | 关闭 | 失败后继续剩余 sequence |
| `--keep-model-output` | 关闭 | 保留 DeepSpeed 最终模型输出 |
| `--skip-dataset-check` | 关闭 | 跳过模型、tokenizer、长度校验 |
| `--dry-run` | 关闭 | 生成配置并打印命令 |

KTransformers 额外参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--server-seq-length` | 未设置 | 只运行一个 Server sequence；与 `--seq-lengths` 互斥 |
| `--kt-distributed-checkpoint-reuse` | `on` | 复用 checkpoint 重算的 CPU MoE forward |
| `--kt-owner-threads` | 自动 | global rank 0 的 CPU MoE/optimizer 线程 |

MegaTrain 额外参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--megatrain-root` | `/mnt/data2/wbw/MegaTrain` | MegaTrain checkout |

训练配置中的 learning-rate warmup 固定为 0；`--warmup-steps` 只影响 TPS 统计窗口。

## 5. Conda 环境

默认环境：

```text
KTransformers: /mnt/data2/wbw/conda/envs/Kllama
DeepSpeed:     /mnt/data2/wbw/conda/envs/Deepspeed
MegaTrain:     /mnt/data2/wbw/conda/envs/Megatrain
```

可以用 `FFT_CONDA_ENV` 覆盖当前后端环境，例如：

```bash
FFT_CONDA_ENV=Deepspeed \
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile server
```

DeepSpeed 和 MegaTrain 都要求环境中已有预编译的 `DeepSpeedCPUAdam`，测试期间不允许
临时 JIT 编译。MegaTrain checkout 可用 `FFT_MEGATRAIN_ROOT` 或
`--megatrain-root` 覆盖。

## 6. 计时和 TPS

稳定 TPS 公式：

```text
tokens_per_step = GPU数 × 每卡batch × sequence length × GAS

stable TPS = tokens_per_step
             / 去除 warmup 后的平均 optimizer-step 时间
```

DeepSpeed 和 KTransformers 记录 coarse host-wall forward、backward、optimizer 和
完整 step 时间，不主动调用 `torch.cuda.synchronize()`。

MegaTrain 的 forward/backward 内部包含必要 CUDA event 和同步，因此结果使用：

```text
megatrain_host_wall_with_backend_cuda_sync
```

该结果可以用于 MegaTrain 自身 sequence 对比，但不能把阶段时间解释成与
DeepSpeed/KTransformers 完全相同的无同步计时边界。CPU/GPU 资源采样均运行在 step
计时器之外。

## 7. Dry-run

三个后端都支持：

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --skip-dataset-check \
  --dry-run
```

```bash
bash /mnt/data2/wbw/FFTtest/GLM-4.5-Air/run_finetune_perf_test_bf16_megatrain.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --skip-dataset-check \
  --dry-run
```

即使传入 `--dry-run`，脚本仍检查模型目录、后端依赖、CPUAdam 和 NUMA 条件；
`--skip-dataset-check` 只跳过 tokenizer 与数据长度校验。

## 8. 文件与输出

主要入口：

- [`run_finetune_perf_test_bf16_ktransformers.sh`](../GLM-4.5-Air/run_finetune_perf_test_bf16_ktransformers.sh)
- [`run_finetune_perf_test_bf16_deepspeed.sh`](../GLM-4.5-Air/run_finetune_perf_test_bf16_deepspeed.sh)
- [`run_finetune_perf_test_bf16_megatrain.sh`](../GLM-4.5-Air/run_finetune_perf_test_bf16_megatrain.sh)
- [`finetune_train_with_timing.py`](../GLM-4.5-Air/finetune_train_with_timing.py)
- [`megatrain_glm45_air_train.py`](../GLM-4.5-Air/megatrain_glm45_air_train.py)

结果目录分别为：

```text
test_log/<timestamp>_KTRANSFORMERS_BF16_FULL_SWEEP/
test_log/<timestamp>_DEEPSPEED_BF16_FULL_SWEEP/
test_log/<timestamp>_MEGATRAIN_BF16_FULL_SWEEP/
```

每个 sequence 目录包含：

```text
run_config.json
train_config.yaml 或 megatrain_config.yaml
resource_contract.json
train.log
exit_code.txt
monitor.csv
memory_summary.{md,json}
plots/
step_timing/step_timing.{json,csv,md}
```

`summary.md` 和 `sweep_results.csv` 会在正常结束、训练失败或信号退出时尽量根据已经
生成的 case 汇总。由于 GLM-4.5-Air 是 106.85B 模型，正式 full sweep 前应先查看
单步 smoke test 的 CPU 峰值；脚本只观测内存，不会主动把超过 1 TiB 判定为 OOM。
