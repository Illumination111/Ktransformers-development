# Qwen3.5-122B-A10B VLM LoRA 功能测试启动命令

本测试使用完整的 `Qwen3_5MoeForConditionalGeneration`，保留视觉塔和
`patch_embed.proj` Conv3D；冻结视觉塔与多模态 projector，只训练语言侧 LoRA，并用
KTransformers 包装 48 层 MoE decoder。

默认数据集直接使用 LLaMA-Factory 自带的 `mllm_demo`：

```text
注册表：/mnt/data2/wbw/LLaMA-Factory/data/dataset_info.json
标注：  /mnt/data2/wbw/LLaMA-Factory/data/mllm_demo.json
图片：  /mnt/data2/wbw/LLaMA-Factory/data/mllm_demo_data/{1,2,3}.jpg
规模：  6 条中英双语多轮图文样本、8 个图片引用
```

该数据集可以验证图片解码、Qwen3.5 Processor、视觉塔前向、冻结视觉参数、语言侧
LoRA 梯度、KT MoE 包装和 optimizer step 的完整功能链路。它不能用于判断微调后模型
质量、收敛性、泛化能力或训练吞吐。

## 1. 环境准备

测试固定使用 Kllama 环境中的 `ms-swift 4.4.2`。该版本兼容当前
`torch 2.9.1+cu128` 和 `transformers 5.6.0`，导入 `swift.model.utils` 时会自动将
Conv3D 替换成 `unfold + F.linear` 实现，不需要使用 `swift sft` 命令，也不需要降级
PyTorch。

首次准备环境时执行：

```bash
HTTPS_PROXY=http://192.168.111.1:7897 \
HTTP_PROXY=http://192.168.111.1:7897 \
/mnt/data2/wbw/conda/envs/Kllama/bin/python -m pip install "ms-swift==4.4.2"
```

测试入口通过 `VLM_KT_CONV3D_COMPAT` 从 KT 源码树加载兼容 helper，同时继续使用
Kllama 中与 `ktransformers 0.6.3.post1` 匹配的正式 `kt-kernel 0.6.3.post1`，避免把
开发树版本 `0.6.4` 强装进环境造成包版本冲突。确认源码文件存在：

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

## 2. 不加载 122B 权重的预检

以下命令检查 checkpoint 架构、KT MoE 参数、Processor、全部 demo 图片、训练目标和
ms-swift Conv3D 前向/反向数值等价性，不加载 122B 权重：

```bash
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/run_vlm_lora_smoke.sh \
  --preflight-only \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/LLaMA-Factory/data \
  --dataset-name mllm_demo \
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

## 3. 渲染配置并检查完整启动命令

```bash
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/run_vlm_lora_smoke.sh \
  --dry-run \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/LLaMA-Factory/data \
  --dataset-name mllm_demo \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 1 \
  --cutoff-len 512
```

该命令会在 `test_log/<UTC时间>/train.yaml` 生成本次不可变配置，并打印实际
`accelerate launch` 命令，但不加载权重、不启动训练。

## 4. 一条命令运行 8 卡冒烟测试

```bash
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/run_vlm_lora_smoke.sh \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/LLaMA-Factory/data \
  --dataset-name mllm_demo \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 1 \
  --cutoff-len 512 \
  --log-base /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/test_log
```

每个 accelerate rank 都会在导入 LLaMA-Factory 和加载模型前调用
`enable_swift_conv3d_patch()`。不能在另一个 `python -c` 进程中提前导入 Swift 来代替
这一步，因为 monkeypatch 不会跨进程保留。

## 5. 一条命令运行 8 卡正式功能/稳定性测试

正式测试与单步冒烟测试使用同一个完整 VLM 合同入口，但改用独立配置和日志目录。
默认执行 20 个 optimizer steps，并把 `mllm_demo` 固定切成 4 条训练数据和 2 条评估
数据：

```bash
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/run_vlm_lora_formal.sh \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/LLaMA-Factory/data \
  --dataset-name mllm_demo \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 20 \
  --cutoff-len 512 \
  --log-base /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/formal_test_log
```

正式脚本不接受少于 10 个 step。只检查配置与命令时执行：

```bash
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B/run_vlm_lora_formal.sh \
  --dry-run \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 20
```

训练结束后，脚本会自动运行 `validate_formal_run.py`。以下任一条件不满足都会返回
非零状态：

- `trainer_state.json` 的 `global_step` 精确等于请求步数；
- 每个 optimizer step 均有有限、非负的 loss 记录；
- 每一步都观察到语言 LoRA 的有限非零梯度和实际参数更新；
- 最终 `train_loss`、`eval_loss`、`train_runtime` 有限，且 runtime 大于 0；
- adapter 非空、包含 LoRA tensor，且不含视觉塔 LoRA tensor；
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

## 6. 通过标准

单步冒烟测试的训练与 adapter 验收日志必须同时出现：

```text
[qwen35_vlm_conv3d] ... 'active': True ...
[qwen35_vlm_contract] OK ... swift_conv3d_patch=active
[qwen35_vlm_functional] GRADIENT_OK ...
[qwen35_vlm_functional] OPTIMIZER_OK ... max_abs_delta=...
[qwen35_vlm_functional] PASS optimizer_steps=1 global_step=1
[qwen35_vlm_adapter] PASS ... lora_tensors=... visual_lora=0
```

这些标记分别证明：

1. ms-swift 补丁在当前 rank 生效；
2. 完整 VLM、冻结视觉塔、语言 LoRA 和 48 层 KT wrapper 均存在；
3. demo 的真实图片经过视觉 PatchEmbed，语言 LoRA 得到有限非零梯度；
4. optimizer 确实改变 LoRA 权重；
5. 至少完成一个训练 step；
6. 保存出的 adapter 含 LoRA tensor，且没有冻结视觉塔的 LoRA tensor。

正式测试的 `train.log` 必须包含前五类运行时标记；adapter 由正式结果校验器直接读取
并校验。`formal_validation.log` 还必须在末尾出现：

```text
[qwen35_vlm_formal] PASS steps=20 losses=... train_loss=... eval_loss=...
```

并生成内容为 `"status": "passed"` 的 `formal_summary.json`。当前执行环境看不到
NVIDIA 驱动，因此这里只能完成两套脚本的 `--preflight-only`、`--dry-run` 以及正式
结果校验器的合成测试；122B 权重加载与上述运行时标记仍需在可见 8 卡的服务器会话中
完成。

`formal_test_log/20260810T091518Z` 是修复前的失败记录：它已完成首个图文
forward/backward，但旧 callback 对 DTensor 调用 `aten.count_nonzero` 后停在
`global_step=0`。该目录没有可恢复 checkpoint，修复后应直接重新运行本页第 5 节命令。
