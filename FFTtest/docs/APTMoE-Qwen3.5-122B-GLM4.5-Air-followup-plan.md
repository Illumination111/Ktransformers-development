# APTMoE Qwen3.5-122B-A10B / GLM-4.5-Air 后续测试计划

更新时间：2026-09-15 UTC

## 当前结论

- Qwen3.5-122B-A10B 已通过 8 GPU、BF16、seq=32、2 step/1 warm-up 的全参数更新冒烟测试，但该次运行使用随机权重、合成路由和未画像 placement，只能标记为 `SMOKE_ONLY`。
- GLM-4.5-Air 原始失败的根因是历史路由误预测专家残留在 CUDA，进而使 Adam 状态同时驻留 CPU 和 CUDA。该生命周期缺陷已经修复；随后完成的 32、64、256、512、1024 五长度回归均退出码为 0，且 `valid_full_update=true`、`after_backward_drop_cpu_only=true`、`co_location_repair_count=0`、Adam 状态仅驻留 CPU。
- GLM 五长度稳定性回归已经完成，下一步执行包含 128 的 Qwen 六长度回归。两轮仍使用随机权重、synthetic routing 和 unprofiled placement，只能作为 `SMOKE_ONLY` 修复回归，不能作为正式 APTMoE 吞吐结论。
- Qwen3.5-122B-A10B 的 server placement lookup 已于 2026-09-13 生成，但正式 route trace 仍缺失；GLM 的正式 route trace 和 placement lookup 均缺失。
- 当前 CPU 主指标仍是进程树 RSS 求和，可能重复计算共享页。独立 per-case cgroup 采集尚未完成。
- 当前 GPU 集合锁只能阻止设备字符串完全一致的 APTMoE sweep；设备顺序不同、部分设备重叠或其他后端占用时仍可能冲突。

## 修复与验证顺序

### 0. 修复 GLM 优化器状态混合驻留

状态：`COMPLETE`

任务：

- 在每个 rank 的 backward stage drop 后、优化器执行前后记录参数、梯度、`exp_avg` 和 `exp_avg_sq` 的设备分布。
- 检查 dense stage、embedding、lm_head、动态加载 hot experts，以及异步 drop event 的完整性。
- 修复根因，避免依靠无条件搬运掩盖错误的参数驻留生命周期。

验收条件：

- seq=32 完整运行退出码为 0。
- `full_update_verification.json` 中 `valid_full_update=true`。
- 所有必需参数类别获得梯度并发生符合 BF16 审计规则的数值更新。
- Adam 状态设备集合严格为 `["cpu"]`。
- 记录修复前后单卡显存峰值与 optimizer 时间。

### 1. GLM 多序列稳定性复测

状态：`COMPLETE`

任务：使用 15 step、5 warm-up，按 `1024,512,256,64,32` 顺序重跑；每个成功 case 采集 10 个稳定 step。

本步骤仍显式启用 synthetic routing 和 unprofiled placement，因此只验证修复后的生命周期、稳定性和资源边界，结果必须标记为 `SMOKE_ONLY`，不能用于正式 APTMoE 性能比较。

运行前检查：

- `nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader` 应无占用本次 8 张 GPU 的计算进程。
- 不要同时启动 Qwen、GLM、route capture、lookup profiler 或其他后端任务。
- 保留整个输出目录，不要只保存终端最后几行。

运行命令：

```bash
cd /mnt/data2/wbw/Ktransformers-development/FFTtest/GLM-4.5-Air

nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid \
  --format=csv,noheader

bash run_finetune_perf_test_bf16_aptmoe.sh \
  --seq-lengths 1024,512,256,64,32 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --cpu-threads 12 \
  --devices 0,1,2,3,4,5,6,7 \
  --allow-synthetic-routing \
  --allow-unprofiled-placement \
  --continue-on-error
```

运行完成后需要提供新生成的目录：

`GLM-4.5-Air/test_log/<时间戳>_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY`

验收条件：

- 每个 case 均有 `exit_code.txt`、`proxy_manifest.json`、`full_update_verification.json`、`step_timing/step_timing.json` 和 `memory_summary.json`。
- 成功 case 有 10 个稳定 step，且 `valid_full_update=true`、`after_backward_drop_cpu_only=true`、`co_location_repair_count=0`。
- Adam `exp_avg`/`exp_avg_sq` 仅驻留 CPU；不存在依靠兼容性搬运得到的伪成功。
- 汇总报告逐 case 给出成功或失败状态，并报告 step/TPS 的均值、P50、P95、标准差和峰值显存。
- 长序列失败可以接受，但必须保留退出码、完整 traceback、GPU/CPU 内存和进程退出证据，并继续执行后续短序列。

验收结果：目录 `20260915_060758_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY` 的 5/5 case 成功，每档均有 10 个稳定 step，完整产物和全更新/驻留审计均通过。稳定 TPS 为 seq=32 `7.66`、64 `12.60`、256 `41.72`、512 `64.81`、1024 `91.00`；详细均值、P50、P95、标准差、阶段时间及资源峰值见 [GLM4.5-100-test.md](GLM4.5-100-test.md)。seq=128 未包含在本步骤的显式长度列表中，因此没有结果。

### 2. Qwen 多序列稳定性复测

状态：`READY_FOR_USER_RUN`

任务：使用 15 step、5 warm-up，按 `1024,512,256,128,64,32` 顺序重跑；每个成功 case 采集 10 个稳定 step。

验收条件：运行成功、全更新审计通过，并报告 step/TPS 的均值、P50、P95、标准差和峰值显存。

### 3. 加固 GPU 互斥与分布式初始化

状态：`PENDING`

任务：

- 将按完整设备字符串生成的单个锁改为逐 GPU 锁；设备 ID 排序后按固定顺序获取全部锁，覆盖顺序不同和部分重叠的设备集合。
- APTMoE、KTransformers、PyTorch、route capture 和 lookup profiler 共用同一套 GPU 锁，并在启动前检查已有计算进程。
- GLM 在 `dist.init_process_group()` 前读取 `LOCAL_RANK` 并执行 `torch.cuda.set_device(local_rank)`；初始化时显式绑定 device，barrier 也使用明确的 device。
- 为 GLM 接入和 Qwen 相同的显式 P2P group；评估使用 `batch_isend_irecv`，消除默认 8-rank group 上的 lazy P2P communicator 警告。
- 将 `PYTORCH_CUDA_ALLOC_CONF` 更新为 `PYTORCH_ALLOC_CONF`。

验收条件：重叠 GPU 的第二个任务在分配模型前退出；日志中不再出现 device 未绑定和默认大 group lazy P2P communicator 警告；信号退出后锁和子进程均被正确清理。

### 4. 修复独立 cgroup 内存采集

状态：`PENDING`

任务：确保每个 case 使用独立 cgroup，并从该 cgroup 的 `memory.current`/`memory.peak` 取得不重复计算共享页的 CPU 内存数据。

验收条件：聚合结果的 CPU 主指标为 dedicated cgroup，`cgroup_memory_peak_gb` 不为空，进程树 RSS 仅作为诊断项。

### 5. 准备正式 route trace 和 placement lookup

状态：`PENDING`

任务：

- Qwen3.5-122B-A10B：保留现有 `APTMoE-simulate/lookups/qwen35_122b/server.json`，先验证 host、GPU、expert size、CPU thread 和 `max_tokens` metadata，再用 exact KTransformers warm-up 采集 `1024,512,256,128,64,32` 的正式 route trace。
- GLM-4.5-Air：实现 exact-model router hook、分 rank trace 和 merge 校验；随后在目标 8-GPU/CPU/NUMA/cgroup 拓扑生成 host-specific lookup。
- route 必须校验模型身份、层数、token 数、top-k、expert 范围、pattern 数和 SHA256；lookup 必须校验 expert BF16 大小、GPU/驱动/软件环境及 CPU curve 覆盖范围。
- 空的 `route_capture_seq32.supervisor.log` 不作为运行证据；后续 supervisor 必须记录命令、PID、开始/结束时间和退出码，并使用无缓冲日志。

验收条件：Qwen 六个、GLM 五个 sequence 均具有可验证的 exact route trace；Qwen 和 GLM 均具有当前主机/拓扑对应的 lookup；不允许用 synthetic/unprofiled fallback 冒充正式结果。

### 6. 正式路由、placement 和全序列扫描

状态：`PENDING`

任务：接入真实模型路由 trace 与正式 placement lookup，关闭 synthetic/unprofiled fallback；Qwen 按 `1024,512,256,128,64,32`、GLM 按 `1024,512,256,64,32` 顺序测试。

验收条件：结果标记为 formal deployment proxy；每个成功 case 均通过全更新审计，失败 case 保留完整错误与资源证据；输出最终对比报告。

### 7. 清理测试与文档可复现性问题

状态：`PENDING`

任务：

- 将文档和配置中的旧路径 `/mnt/data2/wbw/FFTtest` 更新为当前路径 `/mnt/data2/wbw/Ktransformers-development/FFTtest`，或统一改为相对路径/launcher 动态注入。
- 解决 GLM 与 Qwen 合并执行 pytest 时同名 `aggregate_sweep_results` 被 Python 模块缓存错误复用的问题；测试应使用唯一模块名或隔离进程。
- 明确静态合同测试使用的环境；当前 `Aptmoe` 环境未安装 pytest，`Kllama` 环境可分别运行 GLM 和 Qwen 合同测试。
- 更新文档顶部状态，避免同时出现“GLM FAILED”和“修复回归成功”的冲突表述。

验收条件：文档命令可从当前仓库直接执行；两组合同测试既可分别运行，也可由一个顶层测试入口可靠执行；状态表与最新产物一致。

## 执行日志

### 2026-09-13：计划建立

- 已记录现有 Qwen 冒烟成功和 GLM 审计失败结果。
- 开始执行当前计划中的步骤 0。

### 2026-09-13：步骤 0 完成

- 根因：前向阶段按历史路由预加载专家，再按当前路由补载；原 drop 逻辑只重新计算并卸载当前 hot 集合，历史误预测且本轮未命中的专家会残留在 CUDA，随后 Adam 状态也被留在 CUDA。
- 修复：GLM stage drop 改为卸载所有实际标记为 `on_GPU` 的专家，同时继续卸载 dense、attention、norm、router、shared expert、embedding 和 lm_head。
- 增加分 rank、分参数类别的参数/梯度/Adam 状态驻留审计；任何兼容性搬运都会计入 `co_location_repair_count` 并使全更新审计失败。
- 静态验证：`6 passed`。
- 8 GPU 回归目录：`../GLM-4.5-Air/test_log/20260913_070553_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY`。
- 回归结果：退出码 0，`valid_full_update=true`，全部 8 个 rank 在两个 step 的 backward drop 后均为 CPU-only，`co_location_repair_count=0`。
- GPU 任务峰值从修复前 46.13 GiB 降至 26.97 GiB；CPU 进程树 RSS 从 869.30 GB 升至 1002.72 GB，符合 Adam 状态由 GPU 回归 CPU 的预期。
- 单稳定 step 为 135.712 秒、1.886 TPS。该数字比修复前临时值慢，属于正确驻留后的新基线，仍需步骤 1 用 10 个稳定样本定量。
- 为避免详细张量快照进入正式 step 计时，后续仅在每步已有的共置检查中记录轻量设备计数，并在全部计时完成后输出一次完整驻留快照。
- 开始执行当前计划中的步骤 1。

### 2026-09-13：步骤 1 测试范围调整

- 用户将 sequence length 范围改为 `1024,512,256,64,32`，后续 GLM、Qwen 和正式路由/placement 扫描均采用这一集合。
- 已定向终止仅包含 seq=32 的旧稳定性运行；未完成目录当时记录为 `../GLM-4.5-Air/test_log/20260913_072102_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY`，无 `exit_code.txt`，不纳入聚合结果。
- 稳定性复测改为新的五序列扫描；使用 `--continue-on-error` 保证单个长序列失败时仍继续其余 case。

### 2026-09-13：五序列测试交由用户执行

- 已停止代理执行的 seq=1024 运行并释放全部 8 张 GPU；未完成目录 `../GLM-4.5-Air/test_log/20260913_072805_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY` 保留但不纳入结果。
- 当时缺少 Qwen 和 GLM 的正式 route trace 与 placement lookup，因此以下完整五序列命令使用显式 fallback，结果标记为 `SMOKE_ONLY`；Qwen server lookup 随后已经生成，最新状态以前文“当前结论”为准。

GLM-4.5-Air：

```bash
cd /mnt/data2/wbw/Ktransformers-development/FFTtest/GLM-4.5-Air
bash run_finetune_perf_test_bf16_aptmoe.sh \
  --seq-lengths 1024,512,256,64,32 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --cpu-threads 12 \
  --devices 0,1,2,3,4,5,6,7 \
  --allow-synthetic-routing \
  --allow-unprofiled-placement \
  --continue-on-error
```

### 2026-09-13：清理并发任务并加固串行执行

- 已停止两个误并发的 GLM 扫描及其全部 `torchrun`/rank 子进程：旧扫描位于 `20260913_072805.../seq_512`，新扫描位于 `20260913_080057.../seq_1024`；两者均为受并发资源争用影响的未完成结果，不纳入性能汇总。
- 清理后进程列表与 `nvidia-smi` 计算进程列表均为空。
- 已确认两个脚本在单次调用内按用户给定顺序阻塞执行各 sequence；前一个 case 完成并写入结果后，循环才会启动下一个 case。
- 正在为 GLM 与 Qwen 的 APTMoE 脚本增加共用设备锁，防止相同 GPU 集合上的第二个独立扫描与当前扫描并发。

Qwen3.5-122B-A10B：

```bash
cd /mnt/data2/wbw/Ktransformers-development/FFTtest/Qwen3.5-122B-A10B
bash run_finetune_perf_test_bf16_aptmoe.sh \
  --profile server \
  --finetuning-type full \
  --seq-lengths 1024,512,256,128,64,32 \
  --steps 15 \
  --warmup-steps 5 \
  --gas 1 \
  --cpu-threads 12 \
  --devices 0,1,2,3,4,5,6,7 \
  --aptmoe-allow-synthetic-routing \
  --aptmoe-allow-unprofiled-placement \
  --aptmoe-allow-linear-attention-fallback \
  --continue-on-error
```

### 2026-09-15：修复计划复核并开始步骤 1

- 已纠正文档顶部过期状态：GLM 混合驻留缺陷已经修复，当前阻塞项是五序列稳定性验证，而不是再次修改相同 drop 逻辑。
- 已核对 Qwen3.5-122B-A10B 的 server lookup 已存在，但五个正式 route trace 尚未生成；GLM 的正式 route 和 lookup 均未生成。
- 已补充 GPU 锁、分布式初始化/P2P、独立 cgroup、正式输入和路径/测试隔离的后续修复任务。
- 分别运行静态合同测试：GLM `6 passed`，Qwen3.5-122B-A10B `4 passed`；两个目录合并到同一 pytest 进程时存在同名模块导入冲突，已列入步骤 7。
- 步骤 1 的 GLM 五序列命令已交给用户执行，当前状态为 `AWAITING_USER_RESULT`。
- 用户完成后需提供新生成的 `GLM-4.5-Air/test_log/<时间戳>_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY` 目录；收到目录后检查逐 case 退出码、全更新审计、驻留、稳定 step、TPS 和内存证据，再更新本文件的任务状态。

### 2026-09-15：步骤 1 验收完成并准备步骤 2

- 已验收 `20260915_060758_APTMOE_GLM45_AIR_BF16_DEPLOYMENT_PROXY`：5/5 case 成功，每档 10 个稳定 step；全更新、backward 后 CPU-only、最终 optimizer CPU-only 和零兼容性搬运审计全部通过。
- GLM 稳定 TPS 为 seq=32 `7.66`、64 `12.60`、256 `41.72`、512 `64.81`、1024 `91.00`；本轮没有调度 seq=128。
- 步骤 1 状态更新为 `COMPLETE`；步骤 2 更新为 `READY_FOR_USER_RUN`，Qwen 命令已将 seq=128 纳入同一次 sweep。
- 结果仍为 synthetic routing、unprofiled placement、随机 BF16 权重的 `SMOKE_ONLY` deployment proxy；正式 route/placement/cgroup 工作仍为待办。

## PyTorch(oneDNN) 混合全量微调测试支持

状态：`READY_FOR_USER_RUN`

为 Qwen3.5-122B-A10B 和 GLM-4.5-Air 增加了独立的 PyTorch BF16
oneDNN 混合全量微调路径，布局对标 KTransformers 的 8 卡实验：默认 8-rank
pipeline parallel、TP=1、DP=1，GPU 主干加每个 rank 自己的 CPU routed
experts；也支持 `PP=4、DP=2、TP=1`，由两个 4-stage pipeline 组组成，
并在对应 stage 间做梯度 all-reduce。入口分别为：

```bash
cd /mnt/data2/wbw/Ktransformers-development/FFTtest/Qwen3.5-122B-A10B
bash run_finetune_perf_test_pytorch_onednn.sh --dry-run

cd /mnt/data2/wbw/Ktransformers-development/FFTtest/GLM-4.5-Air
bash run_finetune_perf_test_pytorch_onednn.sh --dry-run
```

实现位于 `pytorch_onednn/`：Qwen 使用已有的 text-only 配置提取逻辑，GLM
直接加载 `Glm4MoeForCausalLM`；两者均将 attention/token mixer、router、
embedding、norm、shared expert 和 lm_head 放在对应 GPU，将本 rank 的
routed experts 留在 CPU 并在 CPU BF16 autocast/oneDNN 中计算。虚拟 stage
按 `layer_id % PP` 分配，safetensors loader 只 materialize 当前 rank 的层，
同一 DP 组内显式传递激活和梯度，避免重复加载完整模型。CPU AdamW
状态使用 BF16 CPU moments。默认参数与 KT 日志一致：global batch=8、
per-stage microbatch=1、rank0 CPU threads=80、其他 rank=2；按
`1024,512,256,64,32` 扫描。
每个 case 输出 PP/TP/DP 配置、stage residency、step phase timing、TPS、
最大 RSS、oneDNN 状态和 full-FT 参数数量。

为避免影响正在执行的 APTMoE 任务，launcher 默认检测到 APTMoE 进程时退出
（退出码 3）；只有显式传入 `--allow-concurrent` 才会绕过该保护。该新增
路径没有修改 APTMoE、DeepSpeed 或 MegaTrain 的现有入口和配置。

## PyTorch CUDA 8-GPU 全量微调支持

状态：`READY_FOR_USER_RUN`

在 `pytorch_cuda/` 增加了独立的 `torchrun + FSDP full_shard` 后端，并将
`pytorch_cuda` 加入 Qwen/GLM 计时入口的后端合同。运行时使用独立的
`Onednn` Conda 环境，不设置 `CUDA_VISIBLE_DEVICES` 为空；每个 sequence
使用 8 个 GPU、BF16、`finetuning_type=full`，输出位于
`test_log/*PYTORCH_CUDA_8GPU_FULL`。

入口：

```bash
cd /mnt/data2/wbw/Ktransformers-development/FFTtest
bash pytorch_cuda/setup_onednn_env.sh

cd Qwen3.5-122B-A10B
bash run_finetune_perf_test_pytorch_cuda_8gpu.sh --dry-run

cd ../GLM-4.5-Air
bash run_finetune_perf_test_pytorch_cuda_8gpu.sh --dry-run
```

当前机器的 NVIDIA 驱动不可用时只能执行 dry-run；正式启动前应先确认
`torch.cuda.device_count()==8`。FSDP 路径不会自动切换到 APTMoE 或
DeepSpeed；oneDNN 混合路径则固定使用 GPU 主干、CPU routed experts 和
CPU Adam 状态。
