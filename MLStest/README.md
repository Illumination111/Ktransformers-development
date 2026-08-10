# Multi-LoRA Serving test for KT

面向 **sglang-kt / KTransformers** 的 Multi-LoRA Serving（MLS）测试工程：在 **Qwen3.5-397B-A17B** 与 **Qwen3.5-35B-A3B** 上覆盖同进程多 adapter 服务（M1 / M2），以及基于 Nemotron SFT 的 LoRA 训练与 KT composite 转换。两套 case 目录同构，仅模型路径 / 端口 / adapter 输出目录不同。

仓库：<https://github.com/Illumination111/Multi-LoRA-Serving-test-for-KT>

## 工程说明

| 能力 | 说明 |
|---|---|
| **M1 serving** | 同批 1 种 LoRA；切换 composite 不重载 base |
| **M2 serving** | `--kt-lora-dispatch grouped`，同批最多 N 种 adapter，支撑 N 路并发 |
| **LoRA 训练** | LLaMA-Factory + KT FSDP2；三套 Nemotron 数据 → `cuda` / `swe` / `cpp` adapter |
| **数据转换** | HF parquet/jsonl → openai 格式 jsonl + `dataset_info.json` |

更细的协议与踩坑见：

- [`docs/task_bash_Qwen3.5-397B-A17B.md`](docs/task_bash_Qwen3.5-397B-A17B.md) — 397B 起服 / client / e2e
- [`docs/task_bash_Qwen3.5-35B-A3B.md`](docs/task_bash_Qwen3.5-35B-A3B.md) — 35B 起服 / client / e2e / 训练
- [`docs/progress-of-MLS.md`](docs/progress-of-MLS.md) — MLS 进度与结论

### 目录布局

```text
MLStest/
  README.md
  docs/
  dataset/                       # 两套模型共用 HF 原始数据
  lora-adapter/
    Qwen3.5-397B-A17B/{cuda,swe,cpp}/
    Qwen3.5-35B-A3B/{cuda,swe,cpp}/
  Qwen3.5-397B-A17B/             # 默认端口 31006
  Qwen3.5-35B-A3B/               # 默认端口 31007；脚本与 397B 同构
    run_multi_lora_m1_serve.sh
    run_multi_lora_m1_client.sh
    run_multi_lora_m2_client_concurrent.sh
    run_multi_lora_m1_e2e.sh
    configs/default_env.sh
    train/
      run_prepare_data.sh
      run_train_lora.sh
      configs/
      data/dataset_info.json
      scripts/
```

### 环境依赖（默认路径可覆盖）

| 角色 | 397B 默认 | 35B 默认 |
|---|---|---|
| Serving conda | `kt-kernel` | 同左 |
| Train conda | `Kllama` | 同左 |
| Base 模型 | `/mnt/data2/models/Qwen3.5-397B-A17B` | `/mnt/data3/models/Qwen3.5-35B-A3B` |
| 默认端口 | `31006` | `31007` |
| Adapter 输出 | `lora-adapter/Qwen3.5-397B-A17B/` | `lora-adapter/Qwen3.5-35B-A3B/` |
| KT 权重包 | `KT_WEIGHT_PATH`（必须显式指定） | 同左（**各自**已验证的 35B/397B 包） |
| KTransformers | `/mnt/data2/wbw/ktransformers` | 同左 |

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

将 HF 原始数据转为 LLaMA-Factory **openai** 格式 jsonl，并写入各 case 的 `train/data/dataset_info.json`。原始数据目录 `dataset/` **两套模型共用**；prepared jsonl 按 case 分目录（35B 可对已有 397B jsonl 做软链复用）。

```bash
# 任选一个 case（命令相同）
cd Qwen3.5-397B-A17B/train
# 或
cd Qwen3.5-35B-A3B/train

bash run_prepare_data.sh              # cuda + swe + cpp
bash run_prepare_data.sh cuda
bash run_prepare_data.sh swe cpp
MAX_SAMPLES=1000 bash run_prepare_data.sh cuda
```

输出（默认，可被 `DATASET_DIR` 覆盖）：

```text
<case>/train/data/
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
4. （可选）将 run 目录转为 sglang-kt composite → `lora-adapter/<model>/<task>/`

**35B 必须重新训练**（不可复用 397B 的 adapter 权重）；YAML / 三数据集配置与 397B 一致，仅 `model_name_or_path` 与输出路径不同。

```bash
# 397B
cd Qwen3.5-397B-A17B/train
# 35B（模型在 /mnt/data3/models/Qwen3.5-35B-A3B）
cd Qwen3.5-35B-A3B/train

bash run_prepare_data.sh cuda
bash run_train_lora.sh cuda

SKIP_PREPARE=1 bash run_train_lora.sh swe
bash run_train_lora.sh all
bash run_train_lora.sh cuda max_samples=64 max_steps=20
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
# 397B（默认端口 31006）
cd Qwen3.5-397B-A17B
# 35B（默认端口 31007）
cd Qwen3.5-35B-A3B

bash run_multi_lora_m1_serve.sh \
  --kt-weight-path /path/to/verified-kt-weights-for-this-base \
  --lora-paths L0=/path/to/cuda,L1=/path/to/swe \
  --devices 0,1,2,3,4,5,6,7 --tp-size 8 --dry-run

bash run_multi_lora_m1_e2e.sh \
  --kt-weight-path /path/to/verified-kt-weights-for-this-base \
  --lora-paths cuda=/path/to/cuda,swe=/path/to/swe \
  --devices 0,1,2,3,4,5,6,7 --tp-size 8
```

M2 并发 client（对已起服实例）：

```bash
bash run_multi_lora_m2_client_concurrent.sh --host 127.0.0.1 --port 31007 --adapters cuda,swe,cpp
```

完整参数见对应 `docs/task_bash_Qwen3.5-*.md`。

---

## License

[MIT](LICENSE)
