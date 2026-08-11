# 使用 KTransformers + LlamaFactory 微调 Qwen3.5 VLM LoRA

本文面向需要从零搭建训练环境的用户，说明如何同步 GitHub `main`、安装
KTransformers/LlamaFactory、准备 Qwen3.5 VLM 权重和图文/视频数据集、配置 LoRA
范围、启动训练并验收输出。

本文以 **Qwen3.5-122B-A10B 原生 BF16 权重、8 张 GPU、Intel AMX BF16 CPU 后端**
作为已经实际验证的基线。其他 Qwen3.5 MoE VLM 可以参考相同流程，但必须根据模型规模、
decoder 层类型、CPU 内存、GPU 数量和权重精度调整配置，不能直接把 122B 的资源结论套用
到所有型号。

> 状态日期：2026-08-11。两个功能 PR 已合入 `Illumination111` fork 的 `main`，相关临时
> feature 分支已经删除；普通用户应直接 clone `main`。这些改动尚未合入原始上游仓库或
> 发布为新的 PyPI wheel。

## 1. 功能边界

这组 PR 增加的主要能力如下：

- 使用完整的 `Qwen3_5MoeForConditionalGeneration`，不把 VLM 降级为纯文本模型；
- KTransformers 接管 Qwen3.5 MoE 语言专家的 CPU/GPU 异构训练；
- 在 PyTorch 2.9.x 下兼容视觉塔 `patch_embed.proj` 的 Conv3D；
- 通过 `vlm_lora_scope` 分别选择文本侧、视觉侧或两侧 LoRA；
- `vision` 模式可冻结 KT 持有的语言专家，只计算输入梯度；
- 自动 LoRA 目标发现排除 grouped/depthwise Conv，避免 PEFT rank/groups 约束问题。

三种 LoRA 范围：

| `vlm_lora_scope` | 实际含义 | KT 语言专家 |
| --- | --- | --- |
| `text` | 语言模型侧 Linear/受支持 Conv LoRA | 开启 LoRA |
| `vision` | 视觉塔、PatchEmbed 和多模态 merger/projector LoRA | 冻结 |
| `all` | `text` 与 `vision` 的并集 | 开启 LoRA |

`all` 表示“文本 + 视觉两类模态都创建 LoRA”，**不是 Full-FT**。基座权重仍冻结；MoE
又是稀疏路由，因此一次有限数据训练也不保证每个专家都被路由和更新。

当前随测试目录提供的 preflight/smoke/formal 脚本只对
**Qwen3.5-122B-A10B 图像 SFT**做严格合同检查。LlamaFactory 本身支持视频字段，但视频训练
应使用本文的标准 YAML 启动方式并单独验收，不能把图像 formal 脚本的 PASS 当成视频链路
已经通过。

## 2. 同步 GitHub `main` 代码

### 2.1 已合并 PR 与已验证版本

| 项目 | 功能 PR | 功能 head | merge commit | 当前已验证 `main` |
| --- | --- | --- | --- | --- |
| KTransformers | [Illumination111/ktransformers#3](https://github.com/Illumination111/ktransformers/pull/3) | `76f40b5d5006eb4199374a05d3bbc5f392a7e61e` | `62e6bfdbf177d2d38443d6b8fab993eed4b11d6c` | `3e7b77ba94ea7d1cdd4ecdd72604fd2387698e52` |
| LlamaFactory | [Illumination111/LlamaFactory#1](https://github.com/Illumination111/LlamaFactory/pull/1) | `a4db213141060285cc538db52c12442d7562189a` | `f524cba60ea2f8bb18e42958b855efa534d25ca2` | `f524cba60ea2f8bb18e42958b855efa534d25ca2` |

这两个 PR 当前开在 `Illumination111` 的 fork 内，不是上游
`kvcache-ai/ktransformers`、`hiyouga/LLaMA-Factory` 仓库中的 PR。

### 2.2 第一次克隆

先选择一个有足够空间的工作目录。以下示例中的路径均可替换：

```bash
mkdir -p /data/qwen35-vlm
cd /data/qwen35-vlm

git clone --recursive https://github.com/Illumination111/ktransformers.git
git clone https://github.com/Illumination111/LlamaFactory.git
```

设置后续命令使用的路径：

```bash
export KT_DIR=/data/qwen35-vlm/ktransformers
export LF_DIR=/data/qwen35-vlm/LlamaFactory
```

为以后同步上游主分支，可添加只读 upstream remote：

```bash
git -C "$KT_DIR" remote add upstream https://github.com/kvcache-ai/ktransformers.git
git -C "$LF_DIR" remote add upstream https://github.com/hiyouga/LLaMA-Factory.git
```

检查版本：

```bash
git -C "$KT_DIR" rev-parse HEAD
git -C "$LF_DIR" rev-parse HEAD
git -C "$KT_DIR" status --short --branch
git -C "$LF_DIR" status --short --branch
```

如果要求完全复现本文状态，两个 SHA 应与上表“当前已验证 `main`”一致；以后 `main`
继续前进时，应先查看变更并重新执行本文的 preflight/smoke。

### 2.3 后续同步 `main` 更新

同步前先确认工作区干净；不要用会覆盖本地修改的 `reset --hard`：

```bash
git -C "$KT_DIR" status --short
git -C "$LF_DIR" status --short
```

然后只做 fast-forward 更新：

```bash
git -C "$KT_DIR" fetch --prune origin
git -C "$KT_DIR" switch main
git -C "$KT_DIR" merge --ff-only origin/main
git -C "$KT_DIR" submodule update --init --recursive

git -C "$LF_DIR" fetch --prune origin
git -C "$LF_DIR" switch main
git -C "$LF_DIR" merge --ff-only origin/main
```

使用代理时建议只对当前命令设置，不要把含凭据的代理永久写进仓库配置：

```bash
git -c http.proxy=http://PROXY_HOST:PROXY_PORT -C "$KT_DIR" fetch --prune origin
git -c http.proxy=http://PROXY_HOST:PROXY_PORT -C "$LF_DIR" fetch --prune origin
```

`upstream` remote 只用于观察原始上游进度。当前 VLM LoRA 改动位于
`Illumination111` fork 的 `main`，在原始上游合入前不要把本地 fork `main` 直接替换为
`upstream/main`。可以这样比较差异：

```bash
git -C "$KT_DIR" fetch upstream main
git -C "$LF_DIR" fetch upstream main
git -C "$KT_DIR" log --oneline --left-right upstream/main...main
git -C "$LF_DIR" log --oneline --left-right upstream/main...main
```

## 3. 硬件、系统和软件前提

已验证基线：

- Linux x86-64；
- Python 3.12（KTransformers 当前源码要求 Python 3.11+）；
- PyTorch `2.9.1+cu128`；
- `transformers-kt==5.6.0.post1`；
- `accelerate-kt==1.14.0.post1`；
- `ms-swift>=4.4.2,<4.5`，实际验证版本为 4.4.2；
- 8 个 CUDA process，FSDP2；
- Intel AMX BF16 后端，双 thread pool。

122B 原生 BF16 权重本身约需数百 GB 存储和 CPU 内存。建议从 **512 GB CPU RAM** 级别
开始规划，并为数据缓存、LoRA、optimizer、保存时 state dict 和操作系统预留空间。GPU
显存需求取决于 GPU 数量、序列长度、图片分辨率、batch size 和 FSDP 策略；本文已验证的
脚本固定为 8 卡，不代表任意 8 卡组合都一定足够。

检查 CPU、NUMA、内存和 GPU：

```bash
lscpu
numactl --hardware
free -h
nvidia-smi
```

使用本文 `AMXBF16` 配置前，`lscpu` 的 flags 应包含 AMX/BF16 相关能力。没有 AMX 的
机器必须改用 KTransformers 支持的其他后端及对应配置，不能继续照抄 `AMXBF16`。

## 4. 创建 Python 环境并安装 `main` 代码

### 4.1 创建环境

```bash
conda create -n qwen35-vlm-kt python=3.12 -y
conda activate qwen35-vlm-kt
python -m pip install --upgrade pip setuptools wheel cmake ninja
```

按本机驱动选择 PyTorch CUDA wheel。已验证环境使用 2.9.1 + CUDA 12.8：

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
```

CUDA 13.0 环境可选择 PyTorch 对应的 cu130 index。不要在一个已经稳定工作的环境里盲目
混装多个 CUDA wheel。

### 4.2 安装 LlamaFactory 与 KTransformers VLM 依赖

先安装 LlamaFactory，再用 KT fork 包替换其 Transformers/Accelerate 依赖：

```bash
python -m pip install -e "$LF_DIR"

git -C "$KT_DIR" submodule update --init --recursive
cd "$KT_DIR"
python -m pip install -r requirements-vlm-lora.txt
```

专用 requirements 会安装本地 editable `kt-kernel[vlm-sft]` 和
`ktransformers[vlm-sft]`，并固定已验证的 PyTorch、Transformers-KT、Accelerate-KT 与
ms-swift 组合。不要只执行 PyPI 上的 `pip install ktransformers[vlm-sft]` 后就假设已经
获得 fork `main` 的未发布代码。

源码构建若提示缺少 CMake、hwloc、NUMA 或编译器，请根据发行版安装对应开发包。编译
前应确保 `nvcc`/CUDA toolkit 与所选 PyTorch、GPU 架构兼容。

### 4.3 安装后检查

```bash
python -m pip check

LLAMAFACTORY_ALLOW_TRANSFORMERS_KT=1 python - <<'PY'
import importlib.metadata as md
import accelerate
import torch
import transformers
from accelerate.utils.dataclasses import KTransformersPlugin
from kt_kernel.sft.config import KTConfig

print("torch             =", torch.__version__)
print("transformers      =", transformers.__version__)
print("accelerate        =", accelerate.__version__)
print("kt-kernel dist    =", md.version("kt-kernel"))
print("transformers-kt   =", md.version("transformers-kt"))
print("accelerate-kt     =", md.version("accelerate-kt"))
print("KTransformersPlugin =", KTransformersPlugin.__name__)
print("kt_freeze_experts =", "kt_freeze_experts" in KTConfig.__dataclass_fields__)
PY
```

还应确认两个 `main` 的关键文件存在：

```bash
test -f "$KT_DIR/kt-kernel/python/sft/conv3d_compat.py"
test -f "$LF_DIR/src/llamafactory/model/model_utils/kt_vlm.py"
test -f "$LF_DIR/src/llamafactory/model/model_utils/vlm_lora.py"
```

## 5. 准备 Qwen3.5 VLM 模型权重

### 5.1 从 Hugging Face 下载

安装并登录 Hugging Face CLI：

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
```

下载到固定本地目录：

```bash
export MODEL_DIR=/data/models/Qwen3.5-122B-A10B
hf download Qwen/Qwen3.5-122B-A10B --local-dir "$MODEL_DIR"
```

`hf download --local-dir` 会保留仓库目录结构，并利用本地 metadata 避免重复下载。生产
训练最好额外记录所下载的模型 revision/commit；需要固定版本时使用 `--revision`。

如果模型仓库需要授权，先在网页接受许可，再运行 `hf auth login`。也可以使用企业内部
对象存储或 ModelScope，但最终目录必须是 Transformers 可直接加载的完整 checkpoint。

### 5.2 权重目录最低检查

```bash
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
test -f "$MODEL_DIR/tokenizer.json"
test -f "$MODEL_DIR/preprocessor_config.json"
test -f "$MODEL_DIR/video_preprocessor_config.json"
```

检查 VLM 架构，而不是误用纯文本 checkpoint：

```bash
MODEL_DIR="$MODEL_DIR" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_DIR"])
cfg = json.loads((root / "config.json").read_text())
print("architectures =", cfg.get("architectures"))
print("model_type    =", cfg.get("model_type"))
print("text type     =", (cfg.get("text_config") or {}).get("model_type"))
print("vision type   =", (cfg.get("vision_config") or {}).get("model_type"))
assert cfg.get("architectures") == ["Qwen3_5MoeForConditionalGeneration"]
assert cfg.get("model_type") == "qwen3_5_moe"
assert cfg.get("vision_config")
PY
```

本文的 `AMXBF16` 配置直接读取原生 BF16 checkpoint，不需要把同一个未转换目录再填入
`kt_weight_path`。如果使用 INT8/INT4/其他转换权重，应改用相应 KT backend 和转换流程，
不要只修改 backend 名称。

## 6. 准备并注册多模态数据集

### 6.1 推荐目录结构

```text
/data/qwen35-vlm/datasets/
├── dataset_info.json
├── my_qwen35_vlm.json
├── images/
│   ├── 000001.jpg
│   └── 000002.png
└── videos/
    └── 000001.mp4
```

训练 YAML 中使用：

```yaml
dataset: my_qwen35_vlm
dataset_dir: /data/qwen35-vlm/datasets
```

### 6.2 图像 SFT JSON

`my_qwen35_vlm.json` 是 JSON 数组；也可使用 LlamaFactory 支持的 JSONL。ShareGPT 风格
示例：

```json
[
  {
    "messages": [
      {"role": "user", "content": "<image>请描述图片中的主要内容。"},
      {"role": "assistant", "content": "图片中有两个人正在足球场上庆祝。"},
      {"role": "user", "content": "他们的动作有什么特点？"},
      {"role": "assistant", "content": "两人面向彼此，正在击掌。"}
    ],
    "images": ["images/000001.jpg"]
  },
  {
    "messages": [
      {"role": "user", "content": "比较这两张图：<image><image>"},
      {"role": "assistant", "content": "第一张是白天，第二张是夜晚。"}
    ],
    "images": ["images/000001.jpg", "images/000002.png"]
  }
]
```

关键约束：

- `<image>` 数量必须与 `images` 列表长度一致；
- 多图顺序按 placeholder 在对话中出现的顺序对应；
- 相对路径以 `dataset_dir` 为基准，也可使用绝对路径；
- 媒体 placeholder 应出现在用户输入中；
- 至少有一个非空 assistant 回复作为监督目标；
- 训练前先排除损坏、超大、EXIF 方向异常或不可解码的图片。

### 6.3 `dataset_info.json` 注册

```json
{
  "my_qwen35_vlm": {
    "file_name": "my_qwen35_vlm.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

如果在 LlamaFactory 自带 `data/dataset_info.json` 中追加条目，应保持整个文件仍是合法
JSON，不能写入第二个顶层对象。

### 6.4 视频数据

视频数据使用 `<video>` 与 `videos` 列：

```json
[
  {
    "messages": [
      {"role": "user", "content": "<video>总结这个视频。"},
      {"role": "assistant", "content": "视频展示了一个人在厨房做饭。"}
    ],
    "videos": ["videos/000001.mp4"]
  }
]
```

注册项把 `images` 换成 `videos`：

```json
{
  "my_qwen35_video": {
    "file_name": "my_qwen35_video.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

视频会显著增加预处理和视觉 token 开销。先用很短的视频、小 `video_max_pixels`、单步
训练验证解码器和 Processor，再扩大数据。本文附带的 `validate_vlm_setup.py` 强制检查
`images`，因此不能用于视频数据验收。

## 7. 编写训练配置

### 7.1 训练 YAML

复制 PR 中最接近目标的示例：

```bash
cp "$LF_DIR/examples/ktransformers/train_lora/qwen3_5moe_vlm_all_lora_sft_kt.yaml" \
  /data/qwen35-vlm/train_qwen35_vlm_all.yaml
```

以下是使用完整数据集的参考配置；首次调试时可临时增加 `max_samples: 4` 和
`max_steps: 1`。正式超参数应根据业务数据重新设计：

```yaml
### model
model_name_or_path: /data/models/Qwen3.5-122B-A10B
trust_remote_code: true
image_max_pixels: 262144
video_max_pixels: 16384

### method
stage: sft
do_train: true
do_eval: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all
vlm_lora_scope: all

### dataset
dataset: my_qwen35_vlm
dataset_dir: /data/qwen35-vlm/datasets
template: qwen3_5
cutoff_len: 512
packing: false
overwrite_cache: true
preprocessing_num_workers: 4
dataloader_num_workers: 1
val_size: 0.02

### output
output_dir: /data/qwen35-vlm/outputs/qwen35-122b-vlm-all
logging_steps: 1
save_strategy: steps
save_steps: 100
eval_strategy: steps
eval_steps: 100
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
fp16: false
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
ddp_timeout: 360000000
max_grad_norm: 1.0
resume_from_checkpoint: null

### KTransformers
use_kt: true
```

注意：

- `vlm_lora_scope` 非默认值时，`finetuning_type` 必须为 `lora`；
- `vlm_lora_scope` 非默认值时，`lora_target` 必须为 `all`；
- 图像问答通常保持 `packing: false`，除非已经验证多模态 packing；
- `cutoff_len`、`image_max_pixels`、`video_max_pixels` 会直接影响显存、token 数和速度；
- 首次运行建议临时加 `max_samples: 4` 和 `max_steps: 1`；
- `save_only_model: false` 才适合保留完整 Trainer checkpoint 并继续训练；只需最终 adapter
  时可改为 `true`。

若只训练文本或视觉，把范围分别改为：

```yaml
vlm_lora_scope: text
```

或：

```yaml
vlm_lora_scope: vision
```

### 7.2 Accelerate/FSDP2 + KT 配置

122B 八卡已验证参考：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: Qwen3_5MoeDecoderLayer
  fsdp_version: 2
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
use_cpu: false

kt_config:
  enabled: true
  kt_backend: AMXBF16
  kt_num_threads: 80
  kt_tp_enabled: true
  kt_threadpool_count: 2
  kt_max_cache_depth: 2
  kt_share_backward_bb: true
```

保存为：

```text
/data/qwen35-vlm/accelerate_qwen35_vlm_8gpu.yaml
```

调整原则：

- `num_processes` 必须等于实际参与训练的 GPU 数；
- `CUDA_VISIBLE_DEVICES` 的数量也必须匹配；
- `kt_num_threads` 根据物理核心和其他进程占用调整，不应超过可用 CPU 线程；
- `kt_threadpool_count`、TP 和 NUMA 拓扑相关，修改后必须重新做 smoke；
- 原生 BF16 权重使用 `AMXBF16`；INT8/INT4 需要对应权重流程和配置；
- 不要删除 `Qwen3_5MoeDecoderLayer` 的 FSDP wrap 约束，除非已经验证其他模型结构。

## 8. 启动训练

### 8.1 标准 LlamaFactory 启动方式

这是用于真实完整数据训练的推荐入口：

```bash
cd "$LF_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
USE_KT=1 \
ACCELERATE_USE_KT=true \
LLAMAFACTORY_ALLOW_TRANSFORMERS_KT=1 \
accelerate launch \
  --config_file /data/qwen35-vlm/accelerate_qwen35_vlm_8gpu.yaml \
  src/train.py \
  /data/qwen35-vlm/train_qwen35_vlm_all.yaml
```

建议先把 YAML 临时设置为：

```yaml
max_samples: 4
max_steps: 1
save_strategy: "no"
eval_strategy: "no"
```

单步成功后，删除 `max_samples`/`max_steps`，恢复正式 epoch、保存和评估配置。

LlamaFactory `main` 还提供了三个直接示例：

```text
examples/ktransformers/train_lora/qwen3_5moe_vlm_text_lora_sft_kt.yaml
examples/ktransformers/train_lora/qwen3_5moe_vlm_vision_lora_sft_kt.yaml
examples/ktransformers/train_lora/qwen3_5moe_vlm_all_lora_sft_kt.yaml
```

### 8.2 使用本目录的 122B 图像预检和 smoke

如果用户同时获得了本测试目录，可使用额外合同检查。设定路径：

```bash
export TEST_DIR=/path/to/Ktransformers-development/VLM-FT-test/Qwen3.5-122B-A10B
export VLM_LLAMA_FACTORY_DIR="$LF_DIR"
export VLM_KT_SOURCE_DIR="$KT_DIR/kt-kernel"
export VLM_PYTHON="$(command -v python)"
```

不加载 122B 权重的预检：

```bash
bash "$TEST_DIR/run_vlm_lora_smoke.sh" \
  --preflight-only \
  --model-path "$MODEL_DIR" \
  --dataset-dir /data/qwen35-vlm/datasets \
  --dataset-name my_qwen35_vlm \
  --lora-scope all \
  --devices 0,1,2,3,4,5,6,7
```

预检会检查：

- checkpoint 是否是 48 层、256 expert、top-k 8 的 122B-A10B VLM；
- 视觉塔及 Processor 文件是否完整；
- 图片路径、placeholder 数量和 assistant 目标；
- KT 能否识别 Qwen3.5 MoE 层前缀；
- ms-swift Conv3D 替换的前向/反向自测。

渲染并查看实际启动命令但不训练：

```bash
bash "$TEST_DIR/run_vlm_lora_smoke.sh" \
  --dry-run \
  --model-path "$MODEL_DIR" \
  --dataset-dir /data/qwen35-vlm/datasets \
  --dataset-name my_qwen35_vlm \
  --lora-scope all \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 1 \
  --cutoff-len 512
```

一步真实 smoke：

```bash
bash "$TEST_DIR/run_vlm_lora_smoke.sh" \
  --model-path "$MODEL_DIR" \
  --dataset-dir /data/qwen35-vlm/datasets \
  --dataset-name my_qwen35_vlm \
  --lora-scope all \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 1 \
  --cutoff-len 512 \
  --log-base "$TEST_DIR/test_log"
```

> 该 smoke 模板包含 `max_samples: 6`，用于功能验证，不要把它当作完整业务训练入口。

### 8.3 20-step formal 功能/稳定性测试

```bash
bash "$TEST_DIR/run_vlm_lora_formal.sh" \
  --model-path "$MODEL_DIR" \
  --dataset-dir "$LF_DIR/data" \
  --dataset-name mllm_demo \
  --lora-scope all \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 20 \
  --cutoff-len 512 \
  --log-base "$TEST_DIR/formal_test_log"
```

formal 脚本固定把 6 条 demo 切为 4 条训练、2 条评估，只用于验证多步训练链路，不用于
评估模型收敛、泛化或业务质量。自定义数据若不足 3 条或不适合该固定切分，应继续使用标准
LlamaFactory 入口。

已验证运行 `formal_test_log/20260811T052157Z` 在 8 个 rank 上完成 `all` scope 20 个
optimizer/global steps，first/last loss 为 `10.9609375/1.0723876953125`，train/eval loss
为 `4.882965087890625/3.1796875`。validator 确认 552 个文本和 222 个视觉 LoRA tensors，
共 774 个 LoRA tensors，并保存 `adapter_model.safetensors` 与
`fused_expert_lora.safetensors`。该结果只证明图片 demo 的功能和短时稳定性。

## 9. 训练期间和输出验收

### 9.1 关键日志

正常启动应看到：

```text
Fine-tuning method: LoRA
Found ... VLM LoRA modules for scope `all`
Enabled KT VLM Conv3D compatibility ...
Injected ... fused expert LoRA params into optimizer
```

合同测试还应看到：

```text
[qwen35_vlm_conv3d] required=True active=True ...
[qwen35_vlm_contract] OK scope=all ...
[qwen35_vlm_functional] GRADIENT_OK scope=all ...
[qwen35_vlm_functional] OPTIMIZER_OK scope=all ...
[qwen35_vlm_functional] PASS optimizer_steps=... global_step=...
```

不要只依据 loss 下降认定视觉 LoRA 已工作；至少还要确认视觉 forward、视觉 LoRA 梯度、
optimizer delta 和保存 adapter 中的视觉键。

### 9.2 典型输出

根据范围和保存策略，输出目录通常包含：

```text
adapter_config.json
adapter_model.safetensors
fused_expert_lora.safetensors   # text/all 且使用 KT fused experts 时
trainer_state.json
train_results.json
eval_results.json
tokenizer_config.json
processor_config.json
```

确认训练完成和 adapter 非空：

```bash
test -s /data/qwen35-vlm/outputs/qwen35-122b-vlm-all/adapter_model.safetensors
python -m json.tool \
  /data/qwen35-vlm/outputs/qwen35-122b-vlm-all/adapter_config.json
```

使用 formal 脚本时，最终还应有：

```text
formal_summary.json
formal_validation.log
```

且 `formal_summary.json` 中为：

```json
{"status": "passed"}
```

短 demo 上的 PASS 只证明功能和短时稳定性。MoE 是稀疏路由，未被选中的专家切片保持初始
值属于可能出现的正常现象；若验收标准要求每个目标参数都更新，必须另外实现逐参数/逐专家
覆盖测试。当前 6-row formal 基线还观察到 48 层 `mlp.gate` 的标准 PEFT LoRA-B 保持
为零，而文本主体、视觉侧和被路由的 fused experts 有更新；因此依赖 router LoRA 的任务
必须额外检查 router 梯度，不能只看 formal PASS。

## 10. 常见问题

### 10.1 `unknown keys (['kt_config'])`

环境仍在使用上游 `accelerate`，没有加载 `accelerate-kt`。重新按第 4 节顺序安装，并确认：

```bash
python -c 'from accelerate.utils.dataclasses import KTransformersPlugin; print(KTransformersPlugin)'
```

### 10.2 LlamaFactory 拒绝 `transformers==5.6.0`

普通 Transformers 5.6.0 被排除；当前 KT 组合要求发行包
`transformers-kt==5.6.0.post1`，其 import 版本仍显示 5.6.0。确认安装的是 KT fork，并在
直接 CLI/pytest 运行时设置：

```bash
export LLAMAFACTORY_ALLOW_TRANSFORMERS_KT=1
```

Accelerate 配置中启用 `kt_config` 时也会设置 KT opt-in 标记。

### 10.3 `KT VLM Conv3D compatibility API is unavailable`

通常是导入了旧 `kt-kernel`。检查：

```bash
python -c 'from kt_kernel.sft import conv3d_compat; print(conv3d_compat.__file__)'
python -m pip show kt-kernel ms-swift
```

路径应指向本次 PR 安装，`ms-swift` 应处于 `>=4.4.2,<4.5`。

### 10.4 `vision` 模式提示缺少 `kt_freeze_experts`

这表示 Python 实际导入的 KTConfig 不含 PR #3 的冻结字段：

```bash
python -c 'from kt_kernel.sft.config import KTConfig; print(KTConfig.__dataclass_fields__.keys())'
```

重新安装 `${KT_DIR}/kt-kernel`，不要只设置源码路径或只安装旧 wheel。

### 10.5 找不到 `vlm_lora_scope` 或 `vlm_lora.py`

导入了错误的 LlamaFactory checkout：

```bash
python -c 'import llamafactory; print(llamafactory.__file__)'
git -C "$LF_DIR" rev-parse HEAD
```

重新执行 `python -m pip install -e "$LF_DIR"`。

### 10.6 图片 placeholder/reference 不匹配

逐行统计 `<image>` 和 `images` 数量。多轮对话中的全部 `<image>` 数量对应同一行的完整
`images` 列表，不是只对应当前 user message。

### 10.7 CPU/GPU OOM

依次尝试：

- 降低 `cutoff_len`；
- 降低 `image_max_pixels`/`video_max_pixels`；
- 保持 batch size 1，调整 gradient accumulation；
- 减少数据预处理 worker；
- 确认没有其他任务占用 CPU RAM、shared memory 和 GPU；
- 根据 CPU/权重格式改用正确的 INT8/INT4 KT 流程；
- 增加 GPU 数或 CPU RAM，并重新规划 FSDP/TP，而不是只改一个数字重试。

### 10.8 训练停在保存阶段

FSDP `FULL_STATE_DICT` 保存可能需要额外 CPU RAM 和时间。确保输出盘空间充足，正式长训
应先用小 checkpoint 验证保存与恢复。`save_only_model: true` 会减少保存内容，但不能完整
恢复 optimizer/scheduler 状态。

## 11. 推荐执行顺序

1. clone 两个 fork 的 `main`，并查看功能 PR/目标 SHA；
2. 创建独立 Python 环境；
3. editable 安装 LlamaFactory、KT kernel 和顶层 KT extra；
4. 执行版本、插件、Conv3D、`kt_freeze_experts` 检查；
5. 下载完整 VLM BF16 checkpoint 并核对架构；
6. 用 1～4 条真实图像检查 dataset 注册和 Processor；
7. 运行 `--preflight-only`；
8. 运行 `--dry-run` 并人工检查生成 YAML；
9. 执行 1-step `all` smoke；
10. 分别按需要验证 `text`/`vision`；
11. 执行 20-step formal 功能测试；
12. 最后再用标准 LlamaFactory YAML 启动完整业务数据训练。

## 12. 相关链接

- [KTransformers VLM PR #3](https://github.com/Illumination111/ktransformers/pull/3)
- [LlamaFactory VLM PR #1](https://github.com/Illumination111/LlamaFactory/pull/1)
- [KTransformers VLM Quick Start](https://github.com/Illumination111/ktransformers/blob/main/doc/en/SFT/KTransformers-Fine-Tuning_Quick-Start.md#qwen35-vlm-lora-quick-start)
- [KTransformers VLM Full Documentation](https://github.com/Illumination111/ktransformers/blob/main/doc/en/SFT/KTransformers-Fine-Tuning_User-Guide.md#qwen35-vlm-lora-full-guide)
- [LlamaFactory VLM LoRA 示例说明](https://github.com/Illumination111/LlamaFactory/blob/main/examples/ktransformers/train_lora/README_VLM.md)
- [KTransformers 上游仓库](https://github.com/kvcache-ai/ktransformers)
- [LlamaFactory 上游仓库](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen3.5-122B-A10B 模型页面](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
- [Hugging Face 下载说明](https://huggingface.co/docs/huggingface_hub/en/guides/download)
