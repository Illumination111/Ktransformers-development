# Qwen3.5-122B-A10B VLM LoRA 功能测试启动命令

本测试使用完整的 `Qwen3_5MoeForConditionalGeneration`，保留视觉塔和
`patch_embed.proj` Conv3D，并用 KTransformers 包装 48 层 MoE decoder。启动参数
`--lora-scope` 可以选择只训练文本侧 LoRA、只训练视觉侧 LoRA，或同时训练两侧 LoRA。

默认数据集直接使用 LLaMA-Factory 自带的 `mllm_demo`：

```text
注册表：/mnt/data2/wbw/LLaMA-Factory/data/dataset_info.json
标注：  /mnt/data2/wbw/LLaMA-Factory/data/mllm_demo.json
图片：  /mnt/data2/wbw/LLaMA-Factory/data/mllm_demo_data/{1,2,3}.jpg
规模：  6 条中英双语多轮图文样本、8 个图片引用
```

该数据集可以验证图片解码、Qwen3.5 Processor、视觉塔前向、所选模态的 LoRA 梯度、
KT MoE 包装和 optimizer step 的完整功能链路。它不能用于判断微调后模型质量、收敛性、
泛化能力或训练吞吐。

## 1. LoRA 训练范围

smoke 和 formal 两个启动脚本都接受同一个参数：

```text
--lora-scope text    只训练文本侧 LoRA；视觉塔和 projector 冻结（默认）
--lora-scope vision  只训练视觉塔和多模态 projector LoRA；文本侧及 KT 语言专家冻结
--lora-scope all     同时训练文本侧、视觉塔和多模态 projector LoRA
```

脚本会把该值写入生成的 `train.yaml`：

```yaml
finetuning_type: lora
lora_target: all
vlm_lora_scope: text  # 或 vision / all
```

`all` 表示文本与视觉两侧受 PEFT 支持的 Linear/Conv 模块都创建 LoRA adapter，不是
Full-FT，也不会把模型基座权重全部设为可训练。三个 scope 都依赖包含
`vlm_lora_scope` 功能的 LLaMA-Factory 版本；只有 `vision` 模式另外依赖包含
`kt_freeze_experts` 的 KT 版本。

在两个配套 PR 尚未合入各自主分支前，先让测试指向对应工作树：

```bash
export VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LlamaFactory-vlm-pr
export VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel
```

这足以执行预检和三种 scope 的 `--dry-run`。真正启动 `vision` 训练时，当前 Python
环境中的 kt-kernel 也必须包含 `kt_freeze_experts`；仅把 Conv3D helper 注册进旧
`kt-kernel 0.6.3.post1` 不能冻结 KT 持有的语言专家 LoRA。配套 PR 发布或安装后无需
测试脚本额外 monkeypatch KT wrapper。启动脚本会先检查 LLaMA-Factory 的 scope
模块；真正运行 `vision` 时还会检查当前 Python 导入的 `KTConfig`，缺少上述能力会在
加载 122B 权重前明确失败，不会静默退化成包含文本专家的训练。

## 2. 环境准备

测试固定使用 Kllama 环境中的 `ms-swift 4.4.2`。该版本兼容当前
`torch 2.9.1+cu128` 和 `transformers 5.6.0`，导入 `swift.model.utils` 时会自动将
Conv3D 替换成 `unfold + F.linear` 实现，不需要使用 `swift sft` 命令，也不需要降级
PyTorch。导入动作由 LLaMA-Factory 新增的 KT-VLM 兼容模块自动完成，不由测试脚本
主动执行。

首次准备环境时执行：

```bash
HTTPS_PROXY=http://192.168.111.1:7897 \
HTTP_PROXY=http://192.168.111.1:7897 \
/mnt/data2/wbw/conda/envs/Kllama/bin/python -m pip install "ms-swift==4.4.2"
```

测试入口通过 `VLM_KT_CONV3D_COMPAT` 从 KT 源码树加载兼容 helper，同时继续使用
Kllama 中与 `ktransformers 0.6.3.post1` 匹配的正式 `kt-kernel 0.6.3.post1`，避免把
开发树版本 `0.6.4` 强装进环境造成包版本冲突。这个开发态 shim 只注册新增 Conv3D
Python API，不激活补丁，也不替换已安装 KT 二进制。正式 wheel 发布后应改为：

```bash
python -m pip install 'ktransformers[vlm-sft]'
```

顶层 extra 会安装训练栈并转发到 `kt-kernel[vlm-sft]`；`vlm-sft` 最终只是同一个
`kt-kernel` wheel 的 optional dependency extra，不是另行预编译的 VLM 专用 KT
版本。确认开发树源码文件存在：

```bash
test -f /mnt/data2/wbw/ktransformers/kt-kernel/python/sft/conv3d_compat.py
```

确认环境依赖没有破坏：

```bash
/mnt/data2/wbw/conda/envs/Kllama/bin/python -m pip check
/mnt/data2/wbw/conda/envs/Kllama/bin/python -c \
  'import importlib.metadata as m; print(m.version("ms-swift")); print(m.version("kt-kernel"))'
```

预期输出包含：

```text
4.4.2
0.6.3.post1
```

## 3. 不加载 122B 权重的预检

以下命令检查 checkpoint 架构、KT MoE 参数、Processor、全部 demo 图片、训练目标和
ms-swift Conv3D 前向/反向数值等价性，不加载 122B 权重：

```bash
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/run_vlm_lora_smoke.sh \
  --preflight-only \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/LLaMA-Factory/data \
  --dataset-name mllm_demo \
  --lora-scope text \
  --devices 0,1,2,3,4,5,6,7
```

成功输出必须包含：

```text
"status": "ok"
"swift_version": "4.4.2"
"swift_module": "swift.model.utils"
"self_test": "passed"
"processor": "Qwen3VLProcessor"
"rows": 6
"image_references": 8
```

## 4. 渲染配置并检查完整启动命令

```bash
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/run_vlm_lora_smoke.sh \
  --dry-run \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/LLaMA-Factory/data \
  --dataset-name mllm_demo \
  --lora-scope vision \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 1 \
  --cutoff-len 512
```

该命令会在 `test_log/<UTC时间>/train.yaml` 生成本次不可变配置，并打印实际
`accelerate launch` 命令，但不加载权重、不启动训练。

检查生成的 `train.yaml` 时，必须能看到 `vlm_lora_scope: vision`。将参数改为 `text`
或 `all` 即可渲染另外两种模式。

## 5. 一条命令运行 8 卡冒烟测试

```bash
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/run_vlm_lora_smoke.sh \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/LLaMA-Factory/data \
  --dataset-name mllm_demo \
  --lora-scope text \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 1 \
  --cutoff-len 512 \
  --log-base /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/test_log
```

每个 accelerate rank 都会先注册开发树中的新增 API；随后 LLaMA-Factory 在读取
Qwen3.5 VLM config 后、加载权重前自动调用 `enable_swift_conv3d_patch()`，并在模型
构造后验证真实 Conv3D 参数和 patch marker。测试合同本身只检查结果。不能在另一个
`python -c` 进程中提前导入 Swift 来代替这一步，因为 monkeypatch 不会跨进程保留。

要测试视觉侧或联合 LoRA，只需把同一条命令中的参数分别改成
`--lora-scope vision` 或 `--lora-scope all`。

## 6. 一条命令运行 8 卡正式功能/稳定性测试

正式测试与单步冒烟测试使用同一个完整 VLM 合同入口，但改用独立配置和日志目录。
默认执行 20 个 optimizer steps，并把 `mllm_demo` 固定切成 4 条训练数据和 2 条评估
数据：

```bash
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/run_vlm_lora_formal.sh \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/LLaMA-Factory/data \
  --dataset-name mllm_demo \
  --lora-scope all \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 20 \
  --cutoff-len 512 \
  --log-base /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/formal_test_log
```

正式脚本不接受少于 10 个 step。只检查配置与命令时执行：

```bash
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/run_vlm_lora_formal.sh \
  --dry-run \
  --lora-scope all \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 20
```

训练结束后，脚本会自动运行 `validate_formal_run.py`。以下任一条件不满足都会返回
非零状态：

- `trainer_state.json` 的 `global_step` 精确等于请求步数；
- 每个 optimizer step 均有有限、非负的 loss 记录；
- 每一步都观察到所选 LoRA 模态的有限非零梯度和实际参数更新；`all` 要求文本、视觉
  两组均满足；
- 最终 `train_loss`、`eval_loss`、`train_runtime` 有限，且 runtime 大于 0；
- adapter 非空、包含 LoRA tensor，并严格符合请求的 `text`、`vision` 或 `all` 范围；
- 日志中存在完整 VLM 合同和最终功能测试 PASS 标记。

验收结果写入：

```text
formal_test_log/<UTC时间>/train.yaml
formal_test_log/<UTC时间>/train.log
formal_test_log/<UTC时间>/formal_validation.log
formal_test_log/<UTC时间>/formal_summary.json
formal_test_log/<UTC时间>/model_output/{trainer_state,train_results,eval_results}.json
formal_test_log/<UTC时间>/model_output/adapter_model.safetensors
```

这里的“正式”表示多步训练的功能与短时稳定性验收，不表示模型效果评测。6 条 demo
会被快速重复使用，2 条 eval 也过小，因此不能据此声称 loss 收敛、泛化提升或实际
业务质量提升；正式验收只要求 loss 有限，不要求它单调下降。

FSDP2 会把 LoRA 参数和梯度暴露为 DTensor。测试 callback 会先通过 `to_local()` 取得
当前 rank 的本地 shard，再检查梯度和 optimizer 更新；不会对 DTensor 直接调用
`count_nonzero`，也不会通过 `full_tensor()` 聚合完整参数。

## 7. 通过标准

单步冒烟测试的训练与 adapter 验收日志必须同时出现：

```text
[qwen35_vlm_conv3d] required=True active=True ...
[qwen35_vlm_contract] OK scope=<text|vision|all> ... swift_conv3d_patch=active
[qwen35_vlm_functional] GRADIENT_OK scope=<text|vision|all> ...
[qwen35_vlm_functional] OPTIMIZER_OK scope=<text|vision|all> ...
[qwen35_vlm_functional] PASS optimizer_steps=1 global_step=1
[qwen35_vlm_adapter] PASS scope=<text|vision|all> ... text_lora=... visual_lora=...
```

这些标记分别证明：

1. ms-swift 补丁在当前 rank 生效；
2. 完整 VLM、scope 对应的 LoRA 参数和 48 层 KT wrapper 均存在；
3. demo 的真实图片经过视觉 PatchEmbed，请求的 LoRA 模态得到有限非零梯度；
4. optimizer 确实改变 LoRA 权重；
5. 至少完成一个训练 step；
6. 保存出的 adapter 含 LoRA tensor，且模态组成与 `--lora-scope` 严格一致。

正式测试的 `train.log` 必须包含前五类运行时标记；adapter 由正式结果校验器直接读取
并校验。`formal_validation.log` 还必须在末尾出现：

```text
[qwen35_vlm_formal] PASS scope=<text|vision|all> steps=20 losses=... train_loss=... eval_loss=...
```

并生成内容为 `"status": "passed"` 的 `formal_summary.json`。当前执行环境看不到
NVIDIA 驱动，因此这里只能完成两套脚本的 `--preflight-only`、`--dry-run` 以及正式
结果校验器的合成测试；122B 权重加载与上述运行时标记仍需在可见 8 卡的服务器会话中
完成。

`formal_test_log/20260810T091518Z` 是修复前的失败记录：它已完成首个图文
forward/backward，但旧 callback 对 DTensor 调用 `aten.count_nonzero` 后停在
`global_step=0`。该目录没有可恢复 checkpoint，修复后应直接重新运行本页第 6 节命令。
