# Qwen3.5-35B-A3B 四后端文本-only BF16 全量微调脚本调用与参数配置

四个后端共用同一套参数解析、按 profile 独立定义的序列长度 sweep、TPS 计时和
资源观测逻辑。BF16 是硬编码配置，不能切换为 FP16/FP32。

本地 Qwen3.5 checkpoint 是多模态的，但本测试强制只加载
`Qwen3_5MoeForCausalLM` 文本子模型：视觉塔、`Conv3d`、多模态 processor 和视觉
参数均不会进入模型或 optimizer。这里的“全量微调”专指全部文本 CausalLM 参数。

## 1. 一步启动完整测试

下面的命令可在任意目录直接执行，无需先 `cd`。每条命令都会依次完成：

```text
server：8 GPU、全局 batch 8、使用主机约 2T 内存
consumer：2 GPU、全局 batch 2、不设 benchmark 内存上限、完成后人工审阅 1 TiB 峰值
server sequence：32、64、128、256、512、1024、2048、4096
consumer sequence：16、32、64、128、256、512、1024、2048
每个长度：15 steps，排除前 5 个 warmup steps
精度：BF16
模型：text-only Qwen3_5MoeForCausalLM
```

`--profile both` 会先跑 server，再跑 consumer，不会同时运行两个 profile。默认 CPU
线程按可见物理核心数除以训练 rank 数自动计算；在当前 96 物理核心主机上，server
为 12 线程/rank，consumer 为 48 线程/rank。

下面四条完整启动命令均故意不传 `--seq-lengths`。脚本会分别使用 server 和
consumer 自己的默认长度；尤其在
`--profile both` 下，不能在启动命令中设置一份公共 sequence length 列表来替代这两套
profile 默认值。

### KTransformers：一条命令跑完整测试

该命令自动使用 `Kllama` Conda 环境和 AMXBF16 后端：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/test_log \
  --kt-distributed-checkpoint-reuse on \
  --continue-on-error
```

`--kt-distributed-checkpoint-reuse on` 使多卡 gradient-checkpoint 重算阶段复用第一次
forward 已生成的 CPU routed-expert 缓存。GPU attention、router 等非 CPU MoE 部分仍按
checkpoint 语义正常重算。该开关仅作用于 KTransformers，且默认值就是 `on`；命令中显式
写出是为了让性能测试条件清晰可复现。

KTransformers 和 DeepSpeed 生成的 LLaMA-Factory 配置都固定使用非重入式 checkpoint：
`gradient_checkpointing_kwargs: {use_reentrant: false}`。这也避免 Transformers `Trainer`
的二次初始化把 KTransformers 所需的非重入式 checkpoint 静默改回 LLaMA-Factory 的
reentrant 默认值。APTMoE 使用外部 adapter；做跨后端严格对比时，该 adapter 也应采用
等价的非重入式 checkpoint 设置。

### DeepSpeed：一条命令跑完整测试

该命令自动使用 `Deepspeed` Conda 环境、ZeRO-3、参数和 optimizer CPU offload：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/test_log \
  --continue-on-error
```

DeepSpeed 的 CPUAdam/ZeRO 内部探针不可启用。脚本会强制使用
`DS_PROBE_MODE=off`，optimizer 时间只记录完整的 `DeepSpeedEngine.step()`。

### APTMoE：一条命令跑完整测试

APTMoE 使用项目内置的 Qwen3.5 component-isomorphic deployment proxy。下面显式写出
当前主机上已经验证可用的 Aptmoe Python 和 proxy 入口路径：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_aptmoe.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/test_log \
  --aptmoe-python /mnt/data2/wbw/conda/envs/Aptmoe/bin/python3 \
  --aptmoe-entrypoint /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/aptmoe_qwen35_proxy_train.py \
  --continue-on-error
```

这两个参数也可以完全省略，脚本会自动找到同一 Python 环境和内置入口。不要传入
`/path/to/...` 示例占位符：显式 `--aptmoe-python` 会覆盖自动检测，即使当前 shell
已经激活 `(Aptmoe)`，不存在的覆盖路径仍会导致
`Python for backend aptmoe was not found`。

正式测试还要求预先准备精确 route trace、对应 profile 的 lookup table 和
linear-attention fast path；缺少时脚本会在大规模参数分配前终止。仅验证流程时可按
[README_PERF_SWEEP.md](../Qwen3.5-35B-A3B/README_PERF_SWEEP.md) 添加三个显式
fallback 参数，这类结果会固定标记为 `SMOKE_ONLY`。

### MegaTrain：一条命令跑完整测试

该命令使用新建的 `Megatrain` Conda 环境和本地 MegaTrain checkout：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_megatrain.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --learning-rate 1.0e-5 \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --log-base /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/test_log \
  --megatrain-root /mnt/data2/wbw/MegaTrain \
  --continue-on-error
```

脚本默认使用 `/mnt/data2/wbw/conda/envs/Megatrain` 和
`/mnt/data2/wbw/MegaTrain`，因此 `--megatrain-root` 也可以省略。MegaTrain 入口仍然
强制提取 text-only `Qwen3_5MoeForCausalLM`，使用 `CPUMasterModel` 和预编译的
DeepSpeedCPUAdam；不会把视觉塔或 MTP 参数加入训练。由于 MegaTrain 后端流水线自身
包含必要的 CUDA 同步，其结果使用独立 timing mode，并在汇总中标记
`OK_BACKEND_SYNC`。

### 只运行单个 profile

如果只想测试一种机器规格，可直接复制下面的短命令；其余参数使用上述默认值：

```bash
# KTransformers server：8 卡、约 2T 内存
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7 \
  --kt-distributed-checkpoint-reuse on

# KTransformers consumer：2 卡、无硬限制，完成后人工审阅 1 TiB
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile consumer \
  --devices 0,1 \
  --kt-distributed-checkpoint-reuse on

# DeepSpeed server
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7

# DeepSpeed consumer
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile consumer \
  --devices 0,1

# MegaTrain server
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_megatrain.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7

# MegaTrain consumer
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_megatrain.sh \
  --profile consumer \
  --devices 0,1
```

### 四后端统一输出与计时差异

四套脚本只记录每个 optimizer step 的：

- forward host wall time；
- backward host wall time；
- optimizer host wall time；
- 完整 optimizer-step wall time 和据此计算的 TPS。

KTransformers、DeepSpeed、APTMoE 计时器只读取 `time.perf_counter()`，不主动
调用 `torch.cuda.synchronize()`；MegaTrain 保留后端流水线内部必要的 CUDA 同步，
并以独立 timing mode/`OK_BACKEND_SYNC` 状态标注。逐 step 记录先缓存在内存中，
训练结束后统一写出。所有后端都强制设置：

```text
DS_PROBE_MODE=off
KT_BACKWARD_TIMING=off
KT_SFT_PROFILE=0
FFT_DISABLE_PERF_PROBES=1
```

脚本会在 phase timer 外启动进程树 CPU/GPU 内存采样器。采样器不因越过 1 TiB
终止训练；这里的阶段耗时是训练 API 的 host wall time，不应解释为纯 GPU kernel
时间。

## 2. Profile 固定配置

| Profile | GPU | 每卡 batch | 全局 micro-batch | 每个 optimizer step 的有效 batch | 内存 |
|---|---:|---:|---:|---:|---|
| server | 8 | 1 | 8 | `8 × GAS` | 不设置 cgroup 上限 |
| consumer | 2 | 1 | 2 | `2 × GAS` | 不设置 benchmark cgroup 上限；人工审阅 1 TiB |

GPU 数量和 batch 不提供单独的 `--gpus`、`--batch-size` 参数，避免测试时破坏 server/consumer 定义。

consumer 使用：

```text
numactl --interleave=0,1
monitor.csv -> plots/01_gpu_memory.png + plots/02_cpu_ram.png
memory_summary.md/json -> MANUAL_REVIEW_REQUIRED
```

训练启动前只记录外部已有的 cgroup/swap 状态并验证 NUMA policy；不创建
`MemoryMax`、不关闭 swap，也不会在内存峰值超过 1 TiB 时自动按 OOM 归类。

## 3. 公共参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--profile` | `server` | `server`、`consumer` 或 `both` |
| `--seq-lengths` | 按 profile 选择 | 高级单-profile 覆盖项 |
| `--steps` | `15` | 每个 sequence 的 optimizer steps |
| `--warmup-steps` | `5` | 前 N 步不计入稳定 TPS |
| `--gas` | `1` | Gradient accumulation steps |
| `--learning-rate` | `1.0e-5` | 全量微调学习率 |
| `--cpu-threads` | 物理核心数/rank 数 | 每个训练 rank 的统一 CPU 线程数 |
| `--devices` | 自动选择 | 物理 GPU 列表 |
| `--model-path` | `/mnt/data3/models/Qwen3.5-35B-A3B` | 模型目录 |
| `--dataset-dir` | `FFTtest/dataset` | LLaMA-Factory 数据集目录 |
| `--dataset-name` | `fft_real_100` | `dataset_info.json` 中的数据集名称 |
| `--log-base` | 当前目录下 `test_log` | 测试结果根目录 |
| `--kt-distributed-checkpoint-reuse` | `on` | KTransformers 多卡 checkpoint 重算复用第一次 CPU MoE forward；可设为 `off` 做 A/B 对照 |
| `--continue-on-error` | 关闭 | 某个长度失败后继续后续测试 |
| `--keep-model-output` | 关闭 | 保留最终模型；默认跳过完整权重保存 |
| `--skip-dataset-check` | 关闭 | 跳过 tokenizer 长度校验 |
| `--dry-run` | 关闭 | 只生成配置并打印命令 |

默认 sequence length 为：

```text
server: 32,64,128,256,512,1024,2048,4096
consumer: 16,32,64,128,256,512,1024,2048
```

完整对比测试应保留这些 profile 默认值。`--seq-lengths` 只保留给临时的单-profile
诊断；不要在 `--profile both` 的启动命令中传入，否则一份覆盖值会同时作用于 server
和 consumer，破坏两类机器规格各自的长度范围。

修改步数与 warmup：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile consumer \
  --steps 20 \
  --warmup-steps 5
```

这里 `--warmup-steps` 只控制 TPS 统计排除窗口；学习率 warmup 固定为 0。

也可以通过环境变量设置同一开关；显式命令行参数优先：

```bash
FFT_KT_DISTRIBUTED_CHECKPOINT_REUSE=off \
  bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server
```

每个 sequence 的 `run_config.json` 会记录
`kt_distributed_checkpoint_forward_reuse: true/false`。开启后，训练日志应包含：

```text
Checkpoint forward reuse: enabled=True, distributed_opt_in=True, world_size=2
Distributed checkpoint forward reuse active: layer=0, world_size=2
```

server 模式对应的 `world_size` 应为 `8`。第一行表示各 rank 对开关达成一致，第二行
表示 checkpoint 的第二次 forward 已实际进入缓存复用分支，而不仅是配置已开启。

## 4. GPU 选择

Server 指定 8 张卡：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --devices 0,1,2,3,4,5,6,7
```

Consumer 使用 GPU 6、7：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile consumer \
  --devices 6,7
```

`--profile both` 提供 8 张卡时，server 使用全部 8 张，consumer 使用列表中的前两张：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh \
  --profile both \
  --devices 0,1,2,3,4,5,6,7
```

也支持外部 `CUDA_VISIBLE_DEVICES`，但显式 `--devices` 优先。

## 5. Conda 与 CPU 线程配置

```bash
# 覆盖默认 Conda 环境
FFT_CONDA_ENV=Kllama \
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh --profile server

FFT_CONDA_ENV=Deepspeed \
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh --profile server

FFT_CONDA_ENV=Megatrain \
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_megatrain.sh --profile server
```

DeepSpeed、APTMoE 和 MegaTrain 默认使用
`floor(当前进程可见物理核心数 / profile 的 GPU/rank 数)`。KTransformers
只有 global rank0 创建 CPU MoE backend，因此改为 rank-aware 分配：非 owner
rank 各 2 线程，预留 2 个物理核，其余核心交给 rank0。在当前 96 核主机上，
8 卡为 `80 + 7 × 2 = 94`，双卡为 `92 + 1 × 2 = 94`。

训练入口会按 global rank 设置 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、
`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS`、`BLIS_NUM_THREADS` 和
`ACCELERATE_KT_OMP_NUM_THREADS`；生成的 Accelerate 配置同时把
`kt_num_threads` 设置为 owner 线程数。

命令行覆盖方式：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --cpu-threads 2 \
  --kt-owner-threads 80
```

也可以用同一个环境变量控制任意后端：

```bash
FFT_CPU_THREADS=12 \
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_deepspeed.sh --profile server
```

显式 `--cpu-threads` 优先于 `FFT_CPU_THREADS`；`--kt-owner-threads` 优先于
`FFT_KT_OWNER_THREADS`。旧的后端专用
`FFT_OMP_NUM_THREADS`、`FFT_DS_OMP_NUM_THREADS` 和
`FFT_APTMOE_OMP_NUM_THREADS` 不再使用。

## 6. 配置文件位置

- 公共训练模板：[train_full_bf16_qwen35.yaml](../Qwen3.5-35B-A3B/configs/train_full_bf16_qwen35.yaml)
- KTransformers 8 卡：[accelerate_ktransformers_bf16_8gpu.yaml](../Qwen3.5-35B-A3B/configs/accelerate_ktransformers_bf16_8gpu.yaml)
- KTransformers 2 卡：[accelerate_ktransformers_bf16_2gpu.yaml](../Qwen3.5-35B-A3B/configs/accelerate_ktransformers_bf16_2gpu.yaml)
- DeepSpeed ZeRO-3：[deepspeed_zero3_offload_bf16.json](../Qwen3.5-35B-A3B/configs/deepspeed_zero3_offload_bf16.json)
- MegaTrain 基础配置：[megatrain_qwen35_bf16.yaml](../Qwen3.5-35B-A3B/configs/megatrain_qwen35_bf16.yaml)
- MegaTrain text-only 训练入口：[megatrain_qwen35_train.py](../Qwen3.5-35B-A3B/megatrain_qwen35_train.py)
- 公共启动逻辑：[run_finetune_perf_sweep_bf16_common.sh](../Qwen3.5-35B-A3B/run_finetune_perf_sweep_bf16_common.sh)
- 文本-only 加载契约：[qwen35_text_only.py](../Qwen3.5-35B-A3B/qwen35_text_only.py)
- 统一粗粒度计时器：[step_phase_timer.py](../Qwen3.5-35B-A3B/step_phase_timer.py)
- 一次性资源校验/启动器：[resource_scope_exec.py](../Qwen3.5-35B-A3B/resource_scope_exec.py)
- 计时契约校验器：[validate_step_timing.py](../Qwen3.5-35B-A3B/validate_step_timing.py)

KTransformers、DeepSpeed 每个 sequence 会生成 `train_config.yaml`，MegaTrain 会生成
`megatrain_config.yaml`；KTransformers 还会生成写入实际 `kt_num_threads` 的
`accelerate_config.yaml`。这些才是当次测试的最终配置。

建议正式测试前先执行：

```bash
bash /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B/run_finetune_perf_test_bf16_ktransformers.sh \
  --profile both \
  --dry-run
```

结果写入：

```text
test_log/<timestamp>_<backend>_BF16_FULL_SWEEP/
├── summary.md
├── sweep_results.csv
├── dataset_validation.json
├── server_8gpu_batch8/seq_*/
│   ├── resource_contract.json
│   ├── monitor.csv
│   ├── memory_summary.{md,json}
│   ├── plots/{01_gpu_memory.png,02_cpu_ram.png}
│   └── step_timing/step_timing.{json,csv,md}
└── consumer_2gpu_batch2/seq_*/
```

稳定 TPS 公式为：

```text
TPS = GPU数 × 每卡batch × sequence length × GAS
      / 去除warmup后的平均optimizer-step时间
```
