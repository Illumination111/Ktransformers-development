# Qwen3.5-35B-A3B BF16 四后端全量微调与 APTMoE Proxy TPS Sweep

KTransformers、DeepSpeed 和 MegaTrain 测真实文本模型全量微调；APTMoE 测随机
权重的组件同构 full-update proxy。KTransformers/DeepSpeed 还可显式选择 LoRA，
MegaTrain/APTMoE 在本比较中仅支持 full。目标配置固定为本地
`/mnt/data3/models/Qwen3.5-35B-A3B`，训练与后端混合精度均显式设为
BF16。server 默认按 `4096,2048,1024,512,256,128,64,32`，consumer
默认按 `2048,1024,512,256,128,64,32,16` 从最长到最短测试。每种 sequence length 分别运行
15 个 optimizer steps，去除前 5 个 warmup steps 后计算稳定 TPS。这里的 5 步
是性能统计排除窗口；训练配置的学习率 warmup 为 0。

## 文本-only 模型契约

本地 checkpoint 是多模态的 `Qwen3_5MoeForConditionalGeneration`。仅设置
`freeze_vision_tower: true` 仍会构造视觉塔，因此 Torch 2.9 仍能检测到其中的
`Conv3d`；`20260722_181421_KTRANSFORMERS_BF16_FULL_SWEEP` 正是因此在训练开始前
终止。

现在 KTransformers、DeepSpeed 和 MegaTrain 入口会强制执行以下流程：

- 从源配置提取 `text_config`，构造 `Qwen3_5MoeForCausalLM`；
- 由 Transformers 将 checkpoint 的 `model.language_model.*` 映射到文本模型，
  不加载 `model.visual.*` 和 `mtp.*`；
- 只加载 tokenizer，不创建 `AutoProcessor`；
- 使用无多模态 plugin 的 `qwen3` 文本模板；
- optimizer 创建前检查模型类型、`Conv3d` 数量和多模态参数数量；任一不符合即终止。

因此这里的“全量微调”是全部文本 CausalLM 参数的全量微调，不包括视觉塔。数据预检
也禁止样本出现 `image(s)`、`video(s)` 或 `audio(s)` 字段。每个
`run_config.json` 会记录 `modality: text_only` 和实际加载架构。

## 计时与性能干扰约束

四个后端只记录每个 optimizer step 的以下数据：

- forward host wall time；
- backward host wall time；
- optimizer host wall time；
- 完整 optimizer-step wall time 和由其计算的 TPS。

KTransformers、DeepSpeed 和 APTMoE 的计时器只在三个阶段的 API 边界读取
`time.perf_counter()`，不会主动调用 `torch.cuda.synchronize()`。MegaTrain
内部训练实现包含保证其流水线完成和读取 CUDA event 的同步，不能安全伪装成无同步
口径，因此使用独立的 `megatrain_host_wall_with_backend_cuda_sync` timing mode，
汇总状态标为 `OK_BACKEND_SYNC`。逐 step 数据缓存在内存中，训练结束后才统一写入。
脚本强制设置 `DS_PROBE_MODE=off`、`KT_BACKWARD_TIMING=off` 和
`KT_SFT_PROFILE=0`。

每个 profile 的持久训练进程之外会启动一个 2 秒间隔的系统采样器，记录训练进程树 CPU
RSS、整机 RAM、训练进程 GPU 显存、整卡显存和 GPU 利用率。采样器不在 phase timer
内部，不写逐 step 文件，也不会因内存越线向训练进程发送信号。

每个 profile 只启动一次 rank/worker 集合，在同一批持久进程中从最长 sequence
切换到最短 sequence。最长项完成时激活 CUDA cache hold：释放 Python 模型对象时不
调用 `torch.cuda.empty_cache()`，同一进程的 caching allocator/最长 buffer 会继续
占据已经达到的峰值，直至该 profile 的最后一个长度完成，随后进程退出才统一释放。
KTransformers、DeepSpeed 和 APTMoE 的 NCCL process group，以及 MegaTrain 的 GPU
worker，也都跨 sequence 保留到 profile 结束。这不是另起一个显存占坑进程，因此
不会与下一长度的训练 allocator 竞争。

profile 级 `monitor.csv` 按 `seq_<长度>` phase 记录全程曲线；
`gpu_peak_hold.json` 会逐卡检查后续每个长度的最小任务显存没有跌破最长项峰值
（默认允许 512 MiB 采样误差）。不满足时标记
`GPU_PEAK_HOLD_BROKEN_NOT_OOM` 并使该 profile 失败。所选 GPU 启动前已有 compute
process 时标记 `GPU_BUSY_NOT_OOM`；全部长度结束后的统一释放无法确认时标记
`GPU_RELEASE_UNCONFIRMED_NOT_OOM`。这些资源隔离错误都不会当成训练 OOM，也不会
清理同机其他任务。

因此三个阶段是训练 API 的 host wall time，不应解释为纯 GPU kernel 时间。DeepSpeed
的 optimizer 时间对应 `DeepSpeedEngine.step()` 整段，包含 ZeRO/offload 的更新工作，
但不会进一步探测 CPUAdam 或 ZeRO 内部子阶段。

## Profile

| Profile | GPU | 全局 batch | 每卡 batch | 内存与 NUMA |
|---|---:|---:|---:|---|
| server | 8 | 8 | 1 | 不加 cgroup 上限，使用主机现有约 2T 内存 |
| consumer | 2 | 2 | 1 | 不创建 benchmark cgroup 内存上限；NUMA 0/1 等比例 interleave；运行后人工审阅 1 TiB |

consumer 不再设置 `MemoryMax=1T`、禁 swap 或在超过阈值时杀进程。资源记录程序在
模型加载前记录外部环境已有的 cgroup/swap 状态，并检查 NUMA policy，写出
`resource_contract.json` 后用 `exec` 替换自身。训练完成或失败后，
`memory_summary.md/json` 给出观测峰值是否越过 1 TiB，但 `oom_classification`
固定为 `MANUAL_REVIEW_REQUIRED`；是否按 OOM 记账由人工结合曲线和日志决定。

## CPU 线程

DeepSpeed、APTMoE 和 MegaTrain 默认按 GPU 数均分可见物理核心：

```text
floor(当前进程可见物理核心数 / profile 的 GPU/rank 数)
```

KTransformers 使用 rank0 集中式 CPU MoE backend，因此不再把 96 个物理核平均
切给所有 GPU rank。默认给每个非 owner rank 2 线程、给系统和通信辅助线程预留
2 核，其余全部交给 rank0：

```text
server（8 卡）：rank0 80；rank1～7 各 2；计划合计 94
consumer（2 卡）：rank0 92；rank1 2；计划合计 94
```

训练入口会在导入 PyTorch 前按 global rank 分别设置 `OMP_NUM_THREADS`、
`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS`、
`BLIS_NUM_THREADS` 和 `ACCELERATE_KT_OMP_NUM_THREADS`。`kt_num_threads`
单独使用 owner 的线程数。

可分别覆盖普通 rank 和 KT owner：

```bash
bash run_finetune_perf_test_bf16_ktransformers.sh \
  --profile server \
  --cpu-threads 2 \
  --kt-owner-threads 80
```

也可以使用 `FFT_CPU_THREADS=2` 和 `FFT_KT_OWNER_THREADS=80`。显式参数
优先于对应环境变量。

## 启动

```bash
# KTransformers AMX BF16
bash run_finetune_perf_test_bf16_ktransformers.sh --profile server
bash run_finetune_perf_test_bf16_ktransformers.sh --profile consumer

# 原生 LLaMA-Factory + DeepSpeed ZeRO-3 CPU offload BF16
bash run_finetune_perf_test_bf16_deepspeed.sh --profile server
bash run_finetune_perf_test_bf16_deepspeed.sh --profile consumer

# MegaTrain CPU-master full FT（默认环境 /mnt/data2/wbw/conda/envs/Megatrain）
bash run_finetune_perf_test_bf16_megatrain.sh --profile server
bash run_finetune_perf_test_bf16_megatrain.sh --profile consumer

# 仅检查所有生成配置与资源包装命令
bash run_finetune_perf_test_bf16_megatrain.sh \
  --profile both --seq-lengths 32 --dry-run
```

`--profile both` 按 server、consumer 顺序运行。可以通过
`--seq-lengths 32,64` 缩小调试范围；该参数会覆盖所选 profile 的默认值，
与 `--profile both` 一起使用时只能包含两个 profile 共有的长度。无论输入顺序如何，
脚本都会自动从最长到最短执行。正式对比应保留各 profile 的默认八档。

## APTMoE deployment proxy（已实现，非等价后端）

APTMoE 官方 artifact 没有 Qwen3.5 的通用 Hugging Face/LLaMA-Factory 后端。本目录
现在提供一个独立 adapter，它导入 `/mnt/data2/wbw/APTMoE-baseline`，不修改该
checkout：

- 40 层、34,660,610,688 个参数，与 Qwen3.5 text CausalLM 的组件参数量一致；
- GPU 使用 Transformers 的 30 个 Gated DeltaNet 和 10 个 gated full-attention；
- CPU home 部署 40×256 个独立 6 MiB BF16 routed experts；
- 保留 router、shared expert、248,320-way LM head、loss、backward、梯度裁剪和
  AdamW；每 step 是真实的全参数可更新训练路径；
- 权重由固定 seed 随机初始化，不加载 Qwen3.5 checkpoint shard，默认也不保存。

所以它测的是 `APTMoE Qwen3.5 component-isomorphic deployment-proxy TPS`，不产生
有意义的 loss、模型效果或可用 checkpoint。KTransformers、DeepSpeed 仍通过
LLaMA-Factory 测真实 Qwen3.5；APTMoE proxy 明确记录
`llamafactory_backend: false`，汇总器会永久分表。

完整设计、参数/内存推导和误差边界见
[`APTMOE_DEPLOYMENT_PROXY.md`](APTMOE_DEPLOYMENT_PROXY.md)。

### 1. 无 GPU 参数审计

```bash
cd /mnt/data2/wbw/FFTtest/Qwen3.5-35B-A3B

PYTHONPATH="$PWD:/mnt/data2/wbw/APTMoE-baseline" \
  /mnt/data2/wbw/conda/envs/Aptmoe/bin/python \
  aptmoe_qwen35_proxy_train.py \
  --aptmoe-root /mnt/data2/wbw/APTMoE-baseline \
  --deployment-profile server \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --dataset-dir /mnt/data2/wbw/FFTtest/dataset \
  --dataset-name fft_real_100 \
  --output-dir /mnt/data2/wbw/FFTtest/APTMoE-simulate/audit \
  --step-timing-output-dir /mnt/data2/wbw/FFTtest/APTMoE-simulate/audit/timing \
  --sequence-length 32 --num-gpus 8 --global-batch-size 8 \
  --per-device-batch-size 1 --steps 2 --warmup-steps 1 \
  --precision bf16 --text-only --audit-only
```

### 2. 采集真实 Qwen3.5 路由

在任一真实后端 sweep 上加 `--capture-aptmoe-routes`。hook 只在被排除的
warmup forward 把 top-8 expert ID 搬到 CPU（默认 5 个 pattern，GAS>1 时相应
增加），训练退出后自动合并各 rank：

```bash
bash run_finetune_perf_test_bf16_ktransformers.sh \
  --profile both \
  --capture-aptmoe-routes
```

输出为
`APTMoE-simulate/routes/qwen35/{server,consumer}/seq_<长度>.npz`。也可在
DeepSpeed wrapper 上执行同一参数。trace 形状为
`[patterns, 40, global_batch×sequence, 8]`，proxy 按每个 accumulation
microbatch 依次循环重放，覆盖多个 batch 的路由局部性和 optimizer-state
物化。formal run 要求 pattern 数严格等于 `warmup_steps×GAS`，确保所有 route
cache 和稀疏 state 首次触达都被排除。正式 proxy 会校验 trace 的来源、
层数、token 数、top-k、expert 范围和重复 ID；synthetic trace 不能冒充正式结果。

如果 server 的 8 张 GPU 暂时不能同时使用，可在 2 张卡上保持单卡
microbatch=1，并用 GAS=4 得到相同的有效 global batch=8。Qwen3.5 没有
batch-coupled normalization，attention dropout 也为 0；同一 optimizer step
内的 4 个 accumulation microbatch 使用相同权重，因此可按 token 轴精确聚合：

```bash
bash run_finetune_perf_test_bf16_ktransformers.sh \
  --profile consumer --devices 0,1 \
  --seq-lengths 4096,2048,1024,512,256,128,64,32 \
  --steps 6 --warmup-steps 5 --gas 4 \
  --capture-aptmoe-routes \
  --aptmoe-route-root \
    /mnt/data2/wbw/FFTtest/APTMoE-simulate/routes/qwen35/server_source_gas4

for seq in 4096 2048 1024 512 256 128 64 32; do
  /mnt/data2/wbw/conda/envs/Aptmoe/bin/python \
    merge_qwen35_route_traces.py \
    --input-dir \
      "/mnt/data2/wbw/FFTtest/APTMoE-simulate/routes/qwen35/server_source_gas4/consumer/seq_${seq}_ranks" \
    --output \
      "/mnt/data2/wbw/FFTtest/APTMoE-simulate/routes/qwen35/server/seq_${seq}.npz" \
    --expected-ranks 2 --expected-patterns 20 \
    --sequence-length "${seq}" --global-batch-size 8 \
    --source-accumulation-steps 4
done
```

输出仍是 5 个 server pattern。metadata 会保留 source world size、source
microbatch、GAS 和 20 个原始 pattern，不会把等价采集伪装成 8-rank trace。

### 3. 在目标拓扑生成 lookup

当前 `Aptmoe` 环境已验证 PyTorch 2.9.1/CUDA 12.8、
`flash-linear-attention==0.5.1` 和 `causal-conv1d==1.6.2.post1`，
`require_linear_attention_fastpath()` 可用。环境升级后必须重新执行 fast-path
preflight。然后分别在 server/consumer 实际 CPU 线程、NUMA、cgroup 和 PCIe
拓扑下运行：

```bash
export CUDA_CACHE_PATH=/mnt/data2/wbw/FFTtest/APTMoE-simulate/cache/cuda
export TORCH_EXTENSIONS_DIR=/mnt/data2/wbw/FFTtest/APTMoE-simulate/cache/torch_extensions
export TRITON_CACHE_DIR=/mnt/data2/wbw/FFTtest/APTMoE-simulate/cache/triton

/mnt/data2/wbw/conda/envs/Aptmoe/bin/python \
  profile_aptmoe_qwen35_proxy.py \
  --deployment-profile server \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --output /mnt/data2/wbw/FFTtest/APTMoE-simulate/lookups/qwen35/server.json \
  --simulation-root /mnt/data2/wbw/FFTtest/APTMoE-simulate \
  --sequence-length 128 --max-tokens 32768 --cpu-threads 12

/mnt/data2/wbw/conda/envs/Aptmoe/bin/python \
  profile_aptmoe_qwen35_proxy.py \
  --deployment-profile consumer \
  --model-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --output /mnt/data2/wbw/FFTtest/APTMoE-simulate/lookups/qwen35/consumer.json \
  --simulation-root /mnt/data2/wbw/FFTtest/APTMoE-simulate \
  --sequence-length 128 --max-tokens 8192 --cpu-threads 48
```

lookup 覆盖 6 MiB expert H2D/D2H、CPU expert forward/backward 曲线、
256-way router、两种 Qwen3.5 token mixer，以及首尾 stage 的
embedding/final-norm/LM-head 搬运。不能复用 Qwen3-30B 的 9 MiB 表。
`max-tokens` 必须至少覆盖该 profile 最大的 `global_batch×sequence`；不足时正式
runner 会在模型分配前拒绝 lookup，而不会静默 clamp。

### 4. Smoke 与正式运行

仅验证 pipeline/参数路径时，必须显式打开全部 smoke fallback：

```bash
bash run_finetune_perf_test_bf16_aptmoe.sh \
  --profile consumer \
  --seq-lengths 16 \
  --steps 4 --warmup-steps 2 \
  --aptmoe-allow-synthetic-routing \
  --aptmoe-allow-unprofiled-placement \
  --aptmoe-allow-linear-attention-fallback
```

这类结果固定标为 `SMOKE_ONLY`。真实路由、profile lookup 和 linear-attention
fast path 都准备好后，正式命令不带任何 fallback：

```bash
bash run_finetune_perf_test_bf16_aptmoe.sh --profile server
bash run_finetune_perf_test_bf16_aptmoe.sh --profile consumer
```

默认使用 `/mnt/data2/wbw/conda/envs/Aptmoe` 和
`/mnt/data2/wbw/APTMoE-baseline`，均可通过 `--aptmoe-python`、
`--aptmoe-root` 覆盖。正式运行若缺 route、lookup 或 fast path，会在大规模参数
分配前终止。

随机参数只在 RAM 中构造。仅当显式传入 `--keep-model-output` 时，才按 rank 写出
约 64.56 GiB 的 model-only 随机权重；路径固定在
`/mnt/data2/wbw/FFTtest/APTMoE-simulate/random_weights/`。整个
`APTMoE-simulate/` 已被 Git 忽略。不要对 8 个长度和两个 profile 全量保存，
16 份约 1,032.97 GiB，超过当前约 776 GiB 可用空间。

## 结果

真实后端写入 `test_log/<timestamp>_<backend>_BF16_FULL_SWEEP/`；APTMoE 写入
`test_log/<timestamp>_APTMOE_BF16_DEPLOYMENT_PROXY_SWEEP/`。其中：

- 每个 sequence 的训练配置、完整日志和 `step_timing.json/csv/md`；
- `sweep_results.csv`：稳定 TPS，以及 forward、backward、optimizer 平均耗时；
- `summary.md`：按 profile 汇总的对比表；
- `dataset_validation.json`：Qwen3.5 tokenizer 下的数据长度与 BF16 模型校验；
- `run_config.json`：源架构、文本加载架构以及 `text_only` 模态契约；
- 每个 profile 的 `monitor.csv`、`monitor.log`：全量训练过程的 CPU/GPU 内存和
  利用率原始采样，使用 `seq_<长度>` phase 区分各 sequence；
- 每个 sequence 的 `plots/01_gpu_memory.png`、`plots/02_cpu_ram.png`：
  GPU 显存与 CPU 内存曲线；
- 每个 sequence 的 `memory_summary.md/json`：1 TiB 观测结果和人工 OOM 审阅标记；
- 每个 profile 的 `profile_sweep_manifest.json`、`monitor.csv`、`train.log`：
  持久进程内的长度顺序、分段内存采样和完整日志；
- 每个 profile 的 `gpu_peak_hold.json`：最长项峰值是否一直保持到最短项结束；
- 每个 profile 的 `gpu_lifecycle.json`：测试前 GPU 占用、持久训练进程会话、
  最终残留 worker 清理以及全部长度结束后显存回到基线的确认结果；
- 每个 sequence 的 `resource_contract.json`：训练开始前外部已有的 cgroup、swap
  和 NUMA policy；脚本不会据此创建 1 TiB 限制；
- APTMoE 另写 `proxy_manifest.json` 和 `full_update_verification.json`，记录精确
  参数分类、路由/placement/fast-path 来源、optimizer scope、梯度、权重变化以及
  BF16 moment 的 CPU home device。

默认跳过 LLaMA-Factory 在训练结束后的完整模型保存，避免每个 sequence 重复写出
几十 GB 权重；它不属于 optimizer-step TPS 窗口。APTMoE 同样默认不保存随机权重。
需要保留时显式传入 `--keep-model-output`；APTMoE 只允许写入 Git-ignored 的
`APTMoE-simulate/random_weights/`。

TPS 公式为：

```text
TPS = GPU 数 × 每卡 batch × sequence length × GAS
      / 去除 warmup 后的平均 optimizer-step wall time
```
