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
cd /mnt/qjh001/wubowen/Ktransformers-development/RLFTtest
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

## BF16 storage-rounding probe

`probe_bf16_storage.py` isolates the error caused by storing FP32 intermediate
results in BF16 buffers. It is a small native `AMXBF16_MOE` test, so attention,
layernorm, routing differences, and the vocabulary head are excluded.

For identical BF16 inputs/weights and FP32 routing weights, it computes:

```text
fp32_ideal:  FP32 calculation without intermediate BF16 stores
bf16_store:  same calculation with explicit FP32 -> BF16 -> FP32 stores
amx:         actual AMXBF16_MOE output
```

The report separates:

```text
storage_rounding = fp32_ideal vs bf16_store
amx_extra        = bf16_store vs amx
total            = fp32_ideal vs amx
```

Run it on an idle node. The probe uses CPU AMX; `CUDA_VISIBLE_DEVICES` is set
for resource isolation and does not change the micro-kernel calculation:

```bash
conda activate kt-rlft
CUDA_VISIBLE_DEVICES=1 \
  python probe_bf16_storage.py \
  --threads 48 \
  --qlens 1 8 32 \
  --output results/bf16_storage_probe.json
```

The run on 2026-08-21 used GPU 1 and produced:

| qlen | storage relative L2 | AMX-extra relative L2 | total relative L2 |
|---:|---:|---:|---:|
| 1 | 0.003966 | 0.003486 | 0.003680 |
| 8 | 0.003956 | 0.003250 | 0.003723 |
| 32 | 0.003800 | 0.003173 | 0.003818 |

Interpretation: FP32-to-BF16 storage is a material part of the error, about
`0.38%` relative L2 in this micro-test. The AMX implementation adds a similar
but slightly smaller error, about `0.32%-0.35%`. Therefore the current result
does not support attributing the model-level logprob drift to storage rounding
alone; AMX kernel/packing and operation-order differences must also be retained
as independent contributors.

## Rollout vs score consistency

`rollout_score_consistency.py` tests the actual RLFT comparison contract. It
generates a response once, saves its `response_ids` and rollout
`output_token_logprobs`, and then scores the exact same prompt+response without
sampling again. It reports token-level error, sequence ratio, and PPO clipping
fraction. Use `--skip-hf` for the SGLang self-consistency control and
`--replay-report` to run HF score on an already saved response:

```bash
CUDA_VISIBLE_DEVICES=5,6 \
  python rollout_score_consistency.py \
  --model-path /mnt/qjh007/models/Qwen3-30B-A3B \
  --kt-weight-path /mnt/qjh007/models/Qwen3-30B-A3B \
  --kt-method BF16 --tp-size 2 --kt-num-threads 48 \
  --max-new-tokens 8 --temperature 1.0 --top-p 1.0 \
  --skip-hf --output-dir results

CUDA_VISIBLE_DEVICES=5,6 \
  python rollout_score_consistency.py \
  --model-path /mnt/qjh007/models/Qwen3-30B-A3B \
  --kt-weight-path /mnt/qjh007/models/Qwen3-30B-A3B \
  --kt-method BF16 --hf-device-map balanced --hf-max-memory-gib 44 \
  --score-backend kt \
  --replay-report results/<rollout-report>/rollout_score_result.json
```

The 2026-08-21 30B run used one prompt and eight response tokens. SGLang
rollout and the same SGLang+KT server's fixed-response score matched to
`2.38e-7` maximum absolute logprob error, with zero PPO clipping. The cross
runtime comparisons were:

| comparison | mean absolute error | max absolute error | sequence ratio | clip fraction |
|---|---:|---:|---:|---:|
| SGLang+KT rollout vs SGLang+KT score | `4.24e-8` | `2.38e-7` | `1.00000014` | `0%` |
| SGLang+KT rollout vs HF+KT score | `0.40505` | `1.84347` | `0.0426` | `37.5%` |
| SGLang+KT rollout vs plain HF score | `0.37467` | `1.76345` | `0.0619` | `25%` |
| HF+KT score vs plain HF score | `0.09556` | `0.42927` | n/a | n/a |

The SGLang log reports `AVX2_BF16_MOE_TP`, while the HF+KT log reports
`AMX_MOE_TP`. Therefore the large cross-runtime difference cannot be assigned
to BF16 storage rounding alone. The controls prove that SGLang's own rollout
and score API are aligned; HF/SGLang runtime and kernel differences dominate
this particular cross-path result, with an additional KT effect visible in the
HF+KT versus plain-HF control.
