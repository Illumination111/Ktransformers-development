# VLM LoRA：训练模块与配置说明

更新日期：2026-09-17。本文以已合并的 KTransformers 和 LLaMA-Factory 上游实现为准，适用于 Qwen3-VL MoE、Qwen3.5 MoE VLM 的 LoRA SFT。开始训练前，先确认所用模型在当前 LLaMA-Factory 的多模态模型注册表中有对应条目，并使用匹配的模板与图文数据。

## 先决定训练什么

VLM 可分为三个需要分别考虑的部分：

| 部分 | 作用 | 什么时候训练 |
| --- | --- | --- |
| 语言模型，包括 Attention、普通 MLP 和 MoE 专家 | 生成回答、遵循指令、推理与表达 | 新的回答格式、术语、任务规则或跨模态推理能力需要调整时。通常作为第一轮基线。 |
| 视觉塔 | 提取图像或视频特征 | 图像域与预训练数据差异较大、细节识别能力不足，并且有足量标注数据时。 |
| 多模态连接层（如 Qwen 的 `visual.merger`） | 把视觉特征送入语言模型 | 视觉与文本的对齐不足时单独评估。它不会因 `lora_target: all` 自动获得 LoRA。 |

建议先用 **语言侧 LoRA** 建立可复现基线，再依据错误分析决定是否加入视觉塔。图像数据本身不要求解冻视觉塔：冻结的视觉塔仍参与前向计算，语言侧仍可学习使用其输出。如果主要错误来自领域图像特征，可比较“语言侧”与“语言 + 视觉侧”两组实验。只需要视觉侧参数更新时，使用纯 GPU LLaMA-Factory 路径；当前 KT 路径不能仅凭冻结标志保证 CPU MoE 专家 LoRA 不更新。

### MoE VLM LoRA 的使用场景

下表的“先训练”是实验起点，不保证某项任务一定需要解冻视觉塔。每个场景都应准备带真实图像或视频的训练集和独立验证集，先观察错误来自视觉识别还是语言侧的理解、推理与输出。

| 场景与样本 | 先训练的模块 | 何时扩大范围 | 主要验收指标 |
| --- | --- | --- | --- |
| **票据、合同截图结构化**：输入扫描件和“提取发票号、金额并按 JSON 输出”；目标是指定字段和值。 | 语言侧 LoRA，视觉塔与连接层冻结。模型已能读出字段却经常漏字段、混淆业务规则或破坏 JSON 格式时尤其合适。 | 若小字、印章或特殊版式持续识别错误，先检查图像分辨率和标注，再比较语言 + 视觉侧 LoRA。 | 字段级准确率、数值一致性、JSON 合法率。 |
| **工业质检图像问答**：输入产线照片，输出缺陷种类、位置和处理建议。 | 语言 + 视觉侧 LoRA，连接层先冻结。工业纹理、光照和缺陷形态与通用图像差异较大时，可让视觉侧适应领域图像。 | 若模型能找对缺陷，但分类术语或处理建议错误，可退回语言侧基线；若视觉特征已改善而跨模态对齐仍差，再单独评估连接层。 | 缺陷召回率、类别准确率、位置匹配及误报率。 |
| **图表与仪表盘分析**：输入业务图表截图，回答“哪一条曲线在 8 月后增长最快，并引用数值”。 | 先做语言侧 LoRA，训练读取后的比较、归纳和带依据回答。 | 若频繁读错曲线、图例或坐标，再比较语言 + 视觉侧 LoRA。 | 数值读取误差、比较题准确率、证据引用正确率。 |
| **设备操作视频问答**：输入操作片段，判断步骤顺序并解释违规动作。 | 在确认所用数据模板与视频预处理链路可运行后，比较语言侧与语言 + 视觉侧 LoRA。 | 只有在错误确实来自画面细节或动作识别时才加入视觉侧；调整帧采样、分辨率和 `video_max_pixels` 后重新比较。 | 步骤顺序准确率、事件定位、误报率。 |

例如票据场景可以用“图片 + `请只返回发票号、开票日期和含税金额的 JSON` → 标准 JSON”构造样本，并在验证集中保留未见过的版式。工业质检则应让同一种缺陷覆盖不同设备、角度与光照，避免模型只记住固定背景。视频场景属于需要单独验收的扩展；上游 #2156 报告的 VLM 端到端检查不能替代该场景的视频训练验证。

对大型 MoE VLM，KT 的意义是让大量路由专家在 CPU 后端运行，使 LoRA 实验不必把全部专家基座权重放入 GPU。专家按样本稀疏路由；如果业务数据只覆盖少数任务类型，即使配置了专家 LoRA，也不能假定每个专家都获得充分更新。训练集应覆盖实际任务分布，并用独立测试集比较各场景的效果。

## 配置如何对应到模块

以下表格中的 `lora_target: all` 指 **自动发现可用的 Linear LoRA 目标**，不是全参数微调，也不是所有模块都获得 LoRA。三个 `freeze_*` 字段由 LLaMA-Factory 解释；它们只控制 PEFT 的目标筛选，不控制 KT 独立管理的融合专家 LoRA。

| 训练目标 | `use_kt` | `freeze_vision_tower` | `freeze_multi_modal_projector` | `freeze_language_model` | 实际结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 语言侧基线 | `true` | `true` | `true` | `false` | 语言侧 PEFT LoRA；KT MoE 专家 LoRA。视觉塔、连接层不训练。 |
| 语言 + 视觉侧 | `true` | `false` | `true` | `false` | 上一行加上受支持的视觉侧 Linear LoRA；连接层仍不训练。 |
| 严格视觉侧 | `false` | `false` | `true` | `true` | 视觉侧 Linear LoRA；语言模型与连接层不训练。需要让完整模型适配纯 GPU 路径。 |

使用 KT 时，**不要把 `freeze_language_model: true` 当作严格视觉侧配置**。LLaMA-Factory 会从 PEFT 目标中移除语言模块，但 KT 会按 LoRA rank 为融合 MoE 专家创建独立参数，并将其交给优化器。#2154 曾尝试增加 `kt_freeze_experts` 和 `vlm_lora_scope`，该 PR 已关闭，相关改动没有进入 #2156 的合并实现。当前正式配置使用上表中的原生 `freeze_*` 字段，**不要填写 `vlm_lora_scope` 或 `kt_freeze_experts`**。

`lora_target: all` 仅发现 Linear。当前 LLaMA-Factory 的 Qwen3-VL 注册表还把 `patch_embed` 列为 LoRA 冲突目标。因此，`freeze_vision_tower: false` 不等于训练 Conv3D patch embedding、位置嵌入或所有视觉参数；应在训练前核对最终目标列表。Qwen3-VL 使用 `template: qwen3_vl`，Qwen3.5 VLM 使用 `template: qwen3_5`，两者不能互换。

### 可运行的语言侧基线

从官方 `examples/ktransformers/train_lora/qwen3vlmoe_lora_sft_kt.yaml` 复制训练 YAML，再设置下列字段。模型路径、数据集和输出目录须替换为实际值；`mllm_demo` 只适合冒烟测试。

```yaml
model_name_or_path: Qwen/Qwen3-VL-30B-A3B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_target: all
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false

dataset: mllm_demo
template: qwen3_vl
cutoff_len: 512
packing: false

use_kt: true
kt_config:
  kt_expert_weight_format: bf16
  kt_backend: auto
```

只有在 CPU/权重格式满足要求时才保留所选 KT 后端。原生 BF16 权重可从 `model_name_or_path` 加载；量化专家权重需要相匹配的转换产物与路径。`lora_rank`、`lora_alpha`、`lora_dropout` 写在训练 YAML 顶层，LLaMA-Factory 会派生 KT 专家 LoRA 参数；不要再在 `kt_config` 或 Accelerate YAML 中写 `lora_rank`、`kt_lora_rank` 等派生值。`kt_config` 的其他 KT 设置也写在**训练 YAML**；Accelerate YAML 只写进程数、混合精度、FSDP 等分布式设置。

要同时训练视觉侧受支持的 Linear，只改这一项：

```yaml
freeze_vision_tower: false
```

严格视觉侧实验应在纯 GPU LLaMA-Factory 环境中使用 `use_kt: false`、`freeze_vision_tower: false`、`freeze_language_model: true`，并保持 `finetuning_type: lora` 与 `lora_target: all`。大模型是否能放入 GPU，需要另行评估。

### 连接层需要单独配置

`lora_target: all` 的自动发现会排除当前模型注册表中的 projector key。因此仅设置 `freeze_multi_modal_projector: false` **不会**让连接层获得 LoRA。若实验目标包含连接层，先查看所用模型修订版中的 `named_modules()`，确认连接层下哪些模块是 PEFT 支持的 Linear，再把那些模块的**完整名称**显式加入 `lora_target`，并设置 `freeze_multi_modal_projector: false`。显式列表要同时包含其他计划训练的语言或视觉目标；它会替代 `all`，不会与 `all` 自动求并集。不要凭 `visual.merger` 这个容器名推断所有子层都适合 LoRA。

## 启动前后检查

1. 在训练日志中核对模型类型、模板、`Found linear modules`、`Set ... not trainable`、PEFT trainable 参数，以及 KT MoE wrapper 数量。仅凭 `lora_target: all` 字面值不能证明实际训练范围。
2. 对含真实图像的样本做至少一个优化器 step，检查 loss 和梯度范数有限，且预期模态的 LoRA-B 参数在 step 后发生变化。视觉侧运行应明确检查视觉 LoRA；KT 运行应检查专家 LoRA。MoE 稀疏路由意味着一个 step 不一定更新每个专家。
3. KT LoRA 的常规 PEFT adapter 文件与 `fused_expert_lora.safetensors` 应一起保存、复制和恢复。只检查常规 adapter 文件会漏掉 KT 专家部分。

## 对上游 VLM 指南的修订建议

已核对 [KTransformers VLM LoRA 指南](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/SFT/KTransformers-VLM-LoRA-Fine-Tuning-Guide.md) 的 2026-09-17 `main` 内容：

| 位置 | 目前表述 | 建议更新 |
| --- | --- | --- |
| `Models and LoRA scope` 表格 | 把 `use_kt: true` 下的 `freeze_language_model: true` 标为 “Vision only” | 加上 KT 融合专家仍会创建 LoRA 的限制；严格视觉侧示例改用 `use_kt: false`，或先实现并验证 KT 专家冻结。 |
| 同一节 | 只列冻结标志与 projector 排除规则 | 明确默认三个冻结值为 `true / true / false`，`lora_target: all` 自动选择 Linear，排除 projector 与 Qwen `patch_embed`；给出语言侧和语言 + 视觉侧的实际模块范围。 |
| `Data and resource settings` | 要求在 FSDP 配置中保持 `kt_config.lora_rank` 与训练 YAML 一致 | 删除此要求。现行 LLaMA-Factory 要求 `kt_config` 放在训练 YAML，LoRA rank 由顶层 `lora_rank` 派生；Accelerate 配置不应填写 KT 参数。 |
| 开头及 `Models and LoRA scope` | 称 KT 只支持 `finetuning_type: lora`，全参数 VLM 必须 `use_kt: false` | #10760 已允许 KT `full` 配置。可说明 BF16 全参数路径已开放，但该 VLM 指南的端到端数据只验证了 LoRA，不把未验证组合称为已验证。 |
| `Quick Start` 的示例 | 只有 `lora_target: all`，未写冻结标志 | 显式写出默认语言侧范围，避免读者误以为 `all` 同时训练视觉塔与连接层。 |

指南中的**官方 LLaMA-Factory 克隆地址**已经由 #2165 修正；**实例级 Conv3D 兼容实现**和“不需要 ms-swift”说明与 #2156 一致，这两点无需回退到 #2154 的旧草案。

## 依据与版本边界

- [KT #2154](https://github.com/kvcache-ai/ktransformers/pull/2154)：已关闭、未合并。其早期 `vlm_lora_scope` / `kt_freeze_experts` 方案不能作为上游配置依据。
- [KT #2156](https://github.com/kvcache-ai/ktransformers/pull/2156)：已合并，加入 Qwen VLM MoE 映射及实例级 Conv3D 兼容，并增加上游 VLM 指南。
- [KT #2165](https://github.com/kvcache-ai/ktransformers/pull/2165)：已合并，仅把指南中的个人 fork 克隆地址改为官方仓库。
- [LLaMA-Factory #10760](https://github.com/hiyouga/LLaMA-Factory/pull/10760)：已合并，加入 Qwen3-VL KT LoRA 示例、KT `full` 配置与 Conv3D 兼容检查。
- 实现核对：[LoRA 目标发现](https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/model/model_utils/misc.py)、[多模态冻结与目标过滤](https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/model/model_utils/visual.py)、[KT 专家 LoRA 创建与参数收集](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/sft/lora.py)、[KT 配置派生](https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/hparams/model_args.py)。

本文关于严格视觉侧 KT 限制的结论，是根据以上代码路径作出的推断；上游 #2156 的端到端验证覆盖默认 LoRA 路径，并未报告独立的视觉侧冻结验收。
