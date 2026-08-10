# Datasets (Nemotron SFT)

本目录保存 **Hugging Face 原始数据集** 的下载位置。仓库只跟踪目录结构与脚本，**不上传** parquet / jsonl 权重本身。

| 本地目录 | Hugging Face repo | 用途 |
|---|---|---|
| `Nemotron-SFT-CUDA-v1/` | [nvidia/Nemotron-SFT-CUDA-v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-CUDA-v1) | CUDA / GPU 编程 LoRA |
| `Nemotron-SFT-SWE-v3/` | [nvidia/Nemotron-SFT-SWE-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3) | Software engineering LoRA |
| `Nemotron-SFT-Competitive-Programming-v2/` | [nvidia/Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2) | Competitive C++ 子集 LoRA |

下载后期望布局：

```text
dataset/
  Nemotron-SFT-CUDA-v1/data/*.parquet (或官方结构)
  Nemotron-SFT-SWE-v3/data/train-*-of-*.parquet
  Nemotron-SFT-Competitive-Programming-v2/data/competitive_programming_cpp_*.jsonl
```

## 下载

需要本机已安装 Hugging Face CLI（`hf` / `huggingface-cli`），并视网络设置代理：

```bash
export HTTP_PROXY=http://host:port   # 可选
export HTTPS_PROXY="$HTTP_PROXY"

# 三个数据集全部下载（Competitive 仅拉 C++ jsonl）
bash dataset/download.sh

# 或单独任务：cuda | swe | cpp | all
bash dataset/download.sh cuda
bash dataset/download.sh swe
bash dataset/download.sh cpp
```

断点续传可使用同目录下的 `resume_*.sh`（同样尊重 `HTTP_PROXY` / `HF` / `DATASET_ROOT`）。

下载完成后，用训练侧脚本转成 LLaMA-Factory openai jsonl，见仓库根目录 [README.md](../README.md#数据集转换)。
