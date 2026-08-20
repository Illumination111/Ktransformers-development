# RLFT probability consistency test

`compare_probability.py` compares teacher-forced next-token log-probabilities
between the local KTransformers SGLang runtime and the current
`transformers-kt`/`kt-kernel` HuggingFace integration. It compares identical
token IDs, so sampling, chat-template differences, and generated-text
divergence do not obscure a backend numerical mismatch.

The default model is `/mnt/qjh007/models/Qwen3-30B-A3B`. The 397B option is
available for a later run, but its checkpoint must first exist at the supplied
`--model-path`.

## Runtime and dependencies

The test uses only local source roots under `/home/wubowen`; the KT and
SGLang trees below are the RL-adapted worktree:

```text
veRL:    /home/wubowen/ktransformers-RL/verl
overlay: /home/wubowen/ktransformers-RL
KT:      /home/wubowen/ktransformers-RL/ktransformers
SGLang:  /home/wubowen/ktransformers-RL/ktransformers/third_party/sglang
```

Install `requirements-qjh007.txt` in the target Python 3.11 environment,
then install the local source trees in editable mode:

```bash
python -m pip install -r requirements-qjh007.txt
python -m pip install -e /home/wubowen/ktransformers-RL/ktransformers/kt-kernel --no-deps
python -m pip install -e /home/wubowen/ktransformers-RL/ktransformers/third_party/sglang/python --no-deps
python -m pip install -e /home/wubowen/ktransformers-RL/verl --no-deps
```

The test no longer imports `ktransformers/archive` or `KTransformersOps`. Build
the RL kernel with AMX support before running HF+KT:

```bash
CPUINFER_ENABLE_AMX=ON CPUINFER_ENABLE_AVX512=ON \
  python -m pip install -e /home/wubowen/ktransformers-RL/ktransformers/kt-kernel --no-deps
```

For the base-model probability check, HF+KT selects
`AMXBF16_SkipLoRA`; no PEFT/LoRA adapter is loaded.
`--engine both` runs SGLang first, stops its process group, and only then
loads HF+KT so the two backends do not compete for GPU memory.

## 30B run

The original model directory is the GPU/HF weight directory. `--kt-weight-path`
must point to the corresponding KT CPU weight directory (for example a BF16,
FP8, AMX, or GGUF conversion). The ordinary model directory must not be passed
as an AMX/GGUF directory unless that is the intended format.

```bash
conda activate kt-rlft
cd /home/wubowen/Ktransformers-development/RLFTtest
CUDA_VISIBLE_DEVICES=0,1 python compare_probability.py \
  --model-size 30b \
  --model-path /mnt/qjh007/models/Qwen3-30B-A3B \
  --kt-method BF16 \
  --kt-weight-path /mnt/qjh007/models/Qwen3-30B-A3B \
  --kt-src /home/wubowen/ktransformers-RL/ktransformers \
  --sglang-src /home/wubowen/ktransformers-RL/ktransformers/third_party/sglang \
  --verl-src /home/wubowen/ktransformers-RL/verl \
  --overlay /home/wubowen/ktransformers-RL \
  --kt-python /home/wubowen/miniconda3/envs/kt-rlft/bin/python \
  --output-dir results
```

每次实际运行都会在 `results/` 下创建独立的时间戳目录，例如
`results/30b_bf16_both_20260820_173000_12345/`，其中保存 `config.json`、
`result.json` 和 `sglang.log`。也可以用 `--output-dir` 指定其他结果根目录。
如果使用 `--output`，它仍表示一个明确的 JSON 文件路径。

Use `--engine sglang` or `--engine hf` to isolate one backend. Use
`--dry-run` to validate the model and runtime paths and print the selected case
configuration without loading weights. `--prompt` can be repeated to add
custom prompts.

## 397B later

```bash
python compare_probability.py \
  --model-size 397b \
  --model-path /mnt/qjh007/models/Qwen3.5-397B-A17B \
  --kt-weight-path /path/to/Qwen3.5-397B-A17B-kt
```

The script records per-token differences plus maximum absolute error, mean
absolute error, RMSE, signed bias, and length mismatch in the JSON report.
The 30B BF16 run on the two visible GPUs completed both backends successfully;
the sample run reported maximum absolute log-probability differences of
`1.3248` and `0.1891` for its two prompts, so it is not bitwise probability
identical and needs further numerical investigation.

The test requires a node where `nvidia-smi` sees the requested GPUs. For BF16,
the original HF model directory can also be passed as `--kt-weight-path`; for
AMXINT8/AMXINT4 use the corresponding converted KT weight directory. Do not
install the legacy `KTransformersOps` extension: the current runtime uses
`transformers.integrations.kt`, `accelerate-kt`, and `kt_kernel`.
