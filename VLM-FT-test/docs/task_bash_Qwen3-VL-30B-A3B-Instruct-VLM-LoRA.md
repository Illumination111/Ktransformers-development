# Qwen3-VL-30B-A3B-Instruct VLM LoRA 功能测试启动命令

本测试使用完整的 `Qwen3VLMoeForConditionalGeneration`，保留视觉塔、
`model.visual.patch_embed.proj` Conv3D 和语言模型，并用 KTransformers 包装 48 层
Qwen3-VL-MoE decoder。`--lora-scope` 可以选择文本侧、视觉侧或联合 LoRA。

测试目录：

```text
/mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3-VL-30B-A3B-Instruct
```

模型权重：

```text
/mnt/data3/models/Qwen3-VL-30B-A3B-Instruct
```

默认数据集使用 LLaMA-Factory 开发工作树中的 `mllm_demo`：

```text
注册表：/mnt/data2/wbw/LlamaFactory-vlm-pr/data/dataset_info.json
标注：  /mnt/data2/wbw/LlamaFactory-vlm-pr/data/mllm_demo.json
图片：  /mnt/data2/wbw/LlamaFactory-vlm-pr/data/mllm_demo_data/
规模：  6 条中英双语多轮图文样本、8 个图片引用
```

该数据集用于验证图片解码、Qwen3-VL Processor、视觉塔前向、LoRA 梯度、KT MoE
包装、optimizer step 和 adapter 保存链路。它不能用于判断微调质量、收敛性、泛化能力
或实际训练吞吐。

## 1. 当前适配状态

当前环境属于开发态测试，不是已发布组件的完整适配：

- LLaMA-Factory 开发工作树已包含 `qwen3_vl` 模板、`qwen3_vl_moe` 复合模型注册、
  `vlm_lora_scope` 和 KT Conv3D 处理；
- 当前安装的 `kt-kernel` 会拒绝 `Qwen3VLMoeForConditionalGeneration`；
- `/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel/python/sft/arch.py` 已包含对应架构解析，
  但该修改尚未安装；
- Kllama 环境中的 Transformers 5.6.0 被当前 LLaMA-Factory 开发分支显式排除，测试
  runner 会为诊断设置 `DISABLE_VERSION_CHECK=1`；
- 必须完成真实 8 卡 smoke，才能认定训练链路已适配。

runner 会在每个 accelerate rank 内显式启用 KT 开发源码中的 Python 架构 dispatch，
同时继续使用当前环境已经安装的 KT 二进制。这个 shim 仅供验证待合入改动，不能替代
正式安装包含 Qwen3-VL-MoE 支持的 kt-kernel。

完整检测记录见：

```text
/mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3-VL-30B-A3B-Instruct/ADAPTATION_STATUS.md
```

## 2. LoRA 训练范围

smoke 和 formal 脚本接受同一组选项：

```text
--lora-scope text    只训练语言模型 LoRA；视觉塔和 projector 冻结（默认）
--lora-scope vision  只训练视觉塔和多模态 projector LoRA；语言侧冻结
--lora-scope all     同时训练语言模型、视觉塔和 projector LoRA
```

生成的 `train.yaml` 包含：

```yaml
finetuning_type: lora
lora_target: all
vlm_lora_scope: all  # text / vision / all
template: qwen3_vl
use_kt: true
```

这里的 `all` 只表示为文本和视觉两侧受 PEFT 支持的 Linear/Conv 模块创建 LoRA，
不是 Full-FT，视觉与语言基座权重仍被冻结。

真正运行 `vision` scope 时，已安装的 KTConfig 还必须包含 `kt_freeze_experts`。启动脚本
会在加载 30B 权重前检查这一能力，缺失时直接失败。`text` 和 `all` 不执行这一项专用
检查。

## 3. 环境和源码路径

默认环境及工作树：

```text
Python：       /mnt/data2/wbw/conda/envs/Kllama/bin/python
LLaMA-Factory：/mnt/data2/wbw/LlamaFactory-vlm-pr
KT source：    /mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel
```

可以先在当前 shell 设置：

```bash
export VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LlamaFactory-vlm-pr
export VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel
export VLM_PYTHON=/mnt/data2/wbw/conda/envs/Kllama/bin/python
```

后续完整命令仍以内联方式指定工作树，因此无需依赖这些 `export`。

确认关键开发源码存在：

```bash
test -f /mnt/data2/wbw/LlamaFactory-vlm-pr/src/llamafactory/model/model_utils/vlm_lora.py
test -f /mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel/python/sft/arch.py
test -f /mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel/python/sft/conv3d_compat.py
```

确认 Python 依赖：

```bash
/mnt/data2/wbw/conda/envs/Kllama/bin/python -m pip check
/mnt/data2/wbw/conda/envs/Kllama/bin/python -c \
  'import importlib.metadata as m, torch, transformers; print(torch.__version__); print(transformers.__version__); print(m.version("ms-swift")); print(m.version("kt-kernel"))'
```

当前预检使用 torch 2.9.1、Transformers 5.6.0 和 ms-swift 4.4.2。若缺少 Swift：

```bash
/mnt/data2/wbw/conda/envs/Kllama/bin/python -m pip install "ms-swift==4.4.2"
```

## 4. 不加载 30B 权重的适配预检

以下命令读取 config、权重索引和 Processor，但不会实例化或加载 30B 模型权重：

```bash
VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LlamaFactory-vlm-pr \
VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel \
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3-VL-30B-A3B-Instruct/run_vlm_lora_smoke.sh \
  --preflight-only \
  --model-path /mnt/data3/models/Qwen3-VL-30B-A3B-Instruct \
  --dataset-dir /mnt/data2/wbw/LlamaFactory-vlm-pr/data \
  --dataset-name mllm_demo \
  --lora-scope text \
  --devices 0,1,2,3,4,5,6,7
```

成功输出应包含：

```text
"status": "development_only_requires_distributed_smoke"
"architecture": "Qwen3VLMoeForConditionalGeneration"
"shards": 13
"rows": 6
"image_references": 8
"processor": "Qwen3VLProcessor"
"installed_supported": false
"source_supported": true
"source_prefix": "model.language_model.layers"
"source_experts": 128
"source_moe_intermediate_size": 768
"source_top_k": 8
"self_test": "passed"
```

其中 `installed_supported: false` 是当前已知适配缺口，不应忽略；真实训练由 runner 的
开发 shim 继续验证。

## 5. 渲染配置并检查启动命令

```bash
VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LlamaFactory-vlm-pr \
VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel \
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3-VL-30B-A3B-Instruct/run_vlm_lora_smoke.sh \
  --dry-run \
  --model-path /mnt/data3/models/Qwen3-VL-30B-A3B-Instruct \
  --dataset-dir /mnt/data2/wbw/LlamaFactory-vlm-pr/data \
  --dataset-name mllm_demo \
  --lora-scope all \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 1 \
  --cutoff-len 512
```

该命令会生成：

```text
VLM-FT-test/Qwen3-VL-30B-A3B-Instruct/test_log/<UTC时间>/train.yaml
```

并打印实际 `accelerate launch` 命令，但不加载模型权重、不启动训练。检查 YAML 时应
确认：

```yaml
model_name_or_path: /mnt/data3/models/Qwen3-VL-30B-A3B-Instruct
template: qwen3_vl
vlm_lora_scope: all
max_steps: 1
use_kt: true
```

## 6. 一条命令运行 8 卡 smoke

建议先从文本侧 LoRA 开始：

```bash
VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LlamaFactory-vlm-pr \
VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel \
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3-VL-30B-A3B-Instruct/run_vlm_lora_smoke.sh \
  --model-path /mnt/data3/models/Qwen3-VL-30B-A3B-Instruct \
  --dataset-dir /mnt/data2/wbw/LlamaFactory-vlm-pr/data \
  --dataset-name mllm_demo \
  --lora-scope text \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 1 \
  --cutoff-len 512 \
  --log-base /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3-VL-30B-A3B-Instruct/test_log
```

联合文本与视觉 LoRA：

```bash
VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LlamaFactory-vlm-pr \
VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel \
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3-VL-30B-A3B-Instruct/run_vlm_lora_smoke.sh \
  --model-path /mnt/data3/models/Qwen3-VL-30B-A3B-Instruct \
  --dataset-dir /mnt/data2/wbw/LlamaFactory-vlm-pr/data \
  --dataset-name mllm_demo \
  --lora-scope all \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 1 \
  --cutoff-len 512
```

视觉侧专用测试将 `--lora-scope` 改成 `vision`。如果当前安装的 KTConfig 缺少
`kt_freeze_experts`，脚本会在加载权重前终止；此时应先安装包含该字段的 KT 开发版本，
不能删除检查或把失败解释为训练结果。

## 7. 运行 20 步功能/短时稳定性测试

```bash
VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LlamaFactory-vlm-pr \
VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel \
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3-VL-30B-A3B-Instruct/run_vlm_lora_formal.sh \
  --model-path /mnt/data3/models/Qwen3-VL-30B-A3B-Instruct \
  --dataset-dir /mnt/data2/wbw/LlamaFactory-vlm-pr/data \
  --dataset-name mllm_demo \
  --lora-scope all \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 20 \
  --cutoff-len 512 \
  --log-base /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3-VL-30B-A3B-Instruct/formal_test_log
```

只检查 formal 配置与命令：

```bash
VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LlamaFactory-vlm-pr \
VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel \
bash /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3-VL-30B-A3B-Instruct/run_vlm_lora_formal.sh \
  --dry-run \
  --lora-scope all \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 20
```

当前 Qwen3-VL formal runner 是 20 步功能/短时稳定性测试：每个 optimizer step 都检查
真实视觉前向、所选模态的有限非零梯度及 LoRA 参数更新，结束后检查保存的 adapter
范围。它不会像 Qwen3.5 的增强 formal runner 一样自动切分 eval 数据，也不会生成
`formal_summary.json`；因此不能用它声明 eval loss、收敛性或模型效果。

## 8. 日志和资源记录

每次运行写入独立 UTC 时间目录：

```text
test_log/<UTC时间>/train.yaml
test_log/<UTC时间>/train.log
test_log/<UTC时间>/resource_samples.jsonl
test_log/<UTC时间>/resource_summary.json
test_log/<UTC时间>/adapter_validation.log
test_log/<UTC时间>/model_output/adapter_config.json
test_log/<UTC时间>/model_output/adapter_model.safetensors
```

20 步测试使用 `formal_test_log/<UTC时间>/`，文件布局相同。

资源监控中：

- 主机内存记录系统物理内存基线与峰值；
- GPU 显存由 `nvidia-smi` 采样；
- `resource_summary.json` 保存本次训练的资源汇总。

## 9. 通过标准

训练日志必须出现：

```text
[qwen3vl_kt_arch_shim] source=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel/python/sft/arch.py
[qwen3vl_contract] OK class=Qwen3VLMoeForConditionalGeneration scope=<text|vision|all> ... kt_wrappers=48
[qwen3vl_functional] GRADIENT_OK scope=<text|vision|all> ...
[qwen3vl_functional] OPTIMIZER_OK updates=...
[qwen3vl_functional] PASS optimizer_steps=...
[qwen3vl_adapter] PASS ...
```

这些标记分别证明：

1. 每个 rank 使用了指定的 KT Qwen3-VL-MoE 架构适配源码；
2. 完整 VLM、视觉塔、语言模型、请求范围的 LoRA 和 48 层 KT wrapper 均存在；
3. demo 中的真实图片到达 `model.visual.patch_embed.proj`；
4. 所选模态至少有一个有限非零 LoRA 梯度；
5. optimizer 实际改变了每个请求模态的 LoRA 参数；
6. 保存的 adapter 非空，且严格符合 `text`、`vision` 或 `all` 范围。

任一检查失败，runner 都应返回非零状态。不能只根据进程启动成功、模型加载成功或出现
loss 日志判断测试通过。

## 10. 推荐执行顺序

```text
1. --preflight-only
2. --dry-run --lora-scope text
3. smoke --lora-scope text --max-steps 1
4. --dry-run --lora-scope all
5. smoke --lora-scope all --max-steps 1
6. formal --lora-scope all --max-steps 20
7. KTConfig 支持 kt_freeze_experts 后再测试 vision scope
```

只有第 3、5、6 步在真实 8 卡环境通过后，才能认为 Qwen3-VL 的 KT +
LLaMA-Factory VLM LoRA 开发链路通过集成测试。只有把对应 KT 适配打包安装、移除开发
shim 后再次通过，才能认为安装态适配完成。
