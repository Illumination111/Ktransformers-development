# Multi-LoRA Serving test for KT

面向 **sglang-kt / KTransformers** 的 Multi-LoRA Serving（MLS）测试工程：在 **Qwen3.5-397B-A17B** 上覆盖同进程多 adapter 服务（M1 / M2），以及基于 Nemotron SFT 的 LoRA 训练与 KT composite 转换。

仓库：<https://github.com/Illumination111/Multi-LoRA-Serving-test-for-KT>

## 工程说明

| 能力 | 说明 |
|---|---|
| **M1 serving** | 同批 1 种 LoRA；切换 composite 不重载 base |
| **M2 serving** | `--kt-lora-dispatch grouped`，同批最多 N 种 adapter，支撑 N 路并发 |
| **LoRA 训练** | LLaMA-Factory + KT FSDP2；三套 Nemotron 数据 → `cuda` / `swe` / `cpp` adapter |
| **数据转换** | HF parquet/jsonl → openai 格式 jsonl + `dataset_info.json` |

更细的协议与踩坑见：

- [`docs/task_bash_Qwen3.5-397B-A17B.md`](docs/task_bash_Qwen3.5-397B-A17B.md) — 起服 / client / e2e
- [`docs/progress-of-MLS.md`](docs/progress-of-MLS.md) — MLS 进度与结论

### 目录布局

```text
MLStest/
  README.md
  docs/                          # 任务说明与进度
  dataset/                       # HF 原始数据目录（仅路径/脚本入库，权重不上传）
    download.sh
    Nemotron-SFT-CUDA-v1/
    Nemotron-SFT-SWE-v3/
    Nemotron-SFT-Competitive-Programming-v2/
  lora-adapter/                  # 转换后的 serving composite（权重不上传）
  Qwen3.5-397B-A17B/
    run_multi_lora_m1_serve.sh
    run_multi_lora_m1_client.sh
    run_multi_lora_m2_client_concurrent.sh
    run_multi_lora_m1_e2e.sh
    configs/default_env.sh
    train/
      run_prepare_data.sh
      run_train_lora.sh
      configs/                    # accelerate + train yaml
      data/dataset_info.json     # LLaMA-Factory 注册（jsonl 本地生成）
      scripts/                   # prepare / text-only train / convert
```

### 环境依赖（默认路径可覆盖）

| 角色 | 默认 |
|---|---|
| Serving conda | `kt-kernel`（`MLS_CONDA_ENV`） |
| Train conda | `Kllama`（含 LLaMA-Factory） |
| Base 模型 | `/mnt/data2/models/Qwen3.5-397B-A17B`（`MODEL_PATH`） |
| KT 权重包 | `KT_WEIGHT_PATH`（必须显式指定已验证包） |
| KTransformers | `/mnt/data2/wbw/ktransformers`（`KTRANSFORMERS_ROOT`） |

所有脚本均可通过环境变量覆盖本机路径，见各 `configs/default_env.sh`。

---

## 数据集下载

仓库**跟踪** `dataset/` 目录结构与下载脚本，**不上传** parquet / jsonl 权重。

```bash
# 可选代理
export HTTP_PROXY=http://host:port
export HTTPS_PROXY="$HTTP_PROXY"

bash dataset/download.sh          # cuda + swe + cpp
bash dataset/download.sh cuda     # 单个任务
```

| 任务 | 本地目录 | HF repo |
|---|---|---|
| `cuda` | `dataset/Nemotron-SFT-CUDA-v1/` | `nvidia/Nemotron-SFT-CUDA-v1` |
| `swe` | `dataset/Nemotron-SFT-SWE-v3/` | `nvidia/Nemotron-SFT-SWE-v3` |
| `cpp` | `dataset/Nemotron-SFT-Competitive-Programming-v2/` | C++ jsonl 子集 |

详见 [`dataset/README.md`](dataset/README.md)。断点续传：`dataset/resume_*.sh`。

---

## 数据集转换

将 HF 原始数据转为 LLaMA-Factory **openai** 格式 jsonl，并写入 `train/data/dataset_info.json`。

```bash
cd Qwen3.5-397B-A17B/train

# 三个任务全部转换
bash run_prepare_data.sh

# 单个 / 多个
bash run_prepare_data.sh cuda
bash run_prepare_data.sh swe cpp

# 限制条数（调试）
MAX_SAMPLES=1000 bash run_prepare_data.sh cuda
```

输出（默认，可被 `DATASET_DIR` 覆盖）：

```text
Qwen3.5-397B-A17B/train/data/
  dataset_info.json          # 已入库
  nemotron_cuda.jsonl        # 本地生成，gitignore
  nemotron_swe.jsonl
  nemotron_cpp.jsonl
```

要点：

- 脚本：`scripts/prepare_nemotron_datasets.py`
- tool `arguments` 在 jsonl 中保持 JSON **字符串**（避免 Arrow schema 冲突；训练侧 formatter 不再二次 `json.dumps`）
- 原始根目录：`DATASET_ROOT`（默认 `MLStest/dataset`）

---

## 训练说明

训练入口会：

1. （可选）调用 `run_prepare_data.sh` 生成 jsonl  
2. 以 **text-only** 方式加载 `Qwen3_5MoeForCausalLM`（`text_config`，不建 visual/Conv3d）  
3. 8 卡 FSDP2 + KT LoRA SFT  
4. （可选）将 run 目录转为 sglang-kt composite → `lora-adapter/Qwen3.5-397B-A17B/<task>/`

```bash
cd Qwen3.5-397B-A17B/train

# 准备数据后训练某一任务
bash run_prepare_data.sh cuda
bash run_train_lora.sh cuda

# 数据已就绪时跳过 prepare
SKIP_PREPARE=1 bash run_train_lora.sh swe

# 顺序训练全部（缺 jsonl 的任务会失败并跳过逻辑以脚本为准）
bash run_train_lora.sh all

# 冒烟：少样本 + 少步
bash run_train_lora.sh cuda max_samples=64 max_steps=20

# 跳过转换 / 强制重做 prepare
SKIP_CONVERT=1 bash run_train_lora.sh cuda
FORCE_PREPARE=1 bash run_train_lora.sh cuda
```

| 任务 | 数据 | run 目录 | serving adapter 名 |
|---|---|---|---|
| `cuda` | `nemotron_cuda.jsonl` | `train/runs/nemotron_cuda/` | `cuda` |
| `swe` | `nemotron_swe.jsonl` | `train/runs/nemotron_swe/` | `swe` |
| `cpp` | `nemotron_cpp.jsonl` | `train/runs/nemotron_cpp/` | `cpp` |

常用环境变量：

| 变量 | 含义 |
|---|---|
| `DEVICES` / `NUM_GPUS` | 可见 GPU / 进程数（默认 8） |
| `MODEL_PATH` | base 模型目录 |
| `KT_WEIGHT_PATH` | 若设置，倾向 INT8 accelerate 配置 |
| `MLS_CONDA_ENV` | 默认 `Kllama` |
| `MLS_TEXT_ONLY` | 默认 `1`（text-only 加载） |
| `MLS_CACHE_ROOT` / `TRITON_CACHE_DIR` / `MLS_TMPDIR` | 把 JIT 缓存放到数据盘，避免根分区满 |

单独转换已有 run：

```bash
bash scripts/convert_kt_adapter.sh \
  /path/to/train/runs/nemotron_cuda \
  cuda
```

权重与 checkpoint（`train/runs/`、`*.safetensors`）默认 gitignore，请勿提交。

---

## Serving 快速上手

```bash
cd Qwen3.5-397B-A17B

# Dry-run（不加载模型）
bash run_multi_lora_m1_serve.sh \
  --kt-weight-path /path/to/verified-kt-weights \
  --lora-paths L0=/path/to/cuda,L1=/path/to/swe \
  --devices 0,1,2,3,4,5,6,7 --tp-size 8 --port 31006 --dry-run

# E2E：起服 → smoke → 停服
bash run_multi_lora_m1_e2e.sh \
  --kt-weight-path /path/to/verified-kt-weights \
  --lora-paths cuda=/path/to/cuda,swe=/path/to/swe \
  --devices 0,1,2,3,4,5,6,7 --tp-size 8 --port 31006
```

M2 并发 client：

```bash
bash run_multi_lora_m2_client_concurrent.sh --base-url http://127.0.0.1:31006
```

完整参数与硬约束见 [`docs/task_bash_Qwen3.5-397B-A17B.md`](docs/task_bash_Qwen3.5-397B-A17B.md)。

---

## License

[MIT](LICENSE)
