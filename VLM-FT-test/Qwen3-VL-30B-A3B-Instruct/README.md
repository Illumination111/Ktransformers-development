# Qwen3-VL-30B-A3B-Instruct VLM LoRA tests

This directory is an independent Qwen3-VL-MoE counterpart of
`../Qwen3.5-122B-A10B`. It does not modify or import that test suite.

The test targets the local checkpoint at
`/mnt/data3/models/Qwen3-VL-30B-A3B-Instruct` and the six-row image dataset in
the selected LLaMA-Factory worktree.

## Adaptation status

Run the weight-free audit first:

```bash
bash run_vlm_lora_smoke.sh --preflight-only
```

The audit distinguishes the installed runtime from development sources. At
the time this test was added:

- LLaMA-Factory's VLM development worktree recognizes `qwen3_vl_moe`, the
  `qwen3_vl` template, scoped text/vision/all LoRA, and KT Conv3D handling;
- the installed `kt-kernel` rejects `Qwen3VLMoeForConditionalGeneration`;
- `/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel/python/sft/arch.py` contains
  the required Qwen3-VL-MoE architecture and checkpoint-prefix support, but
  that change is not installed and is currently uncommitted.

Consequently the stack is **not yet adapted in the installed environment**.
The runner can explicitly activate the development-only Python architecture
shim so the integration can be tested before the kt-kernel change is packaged.
It never claims that the released package supports the model.

The Kllama environment currently has Transformers 5.6.0, which the selected
LLaMA-Factory revision explicitly excludes. The staged test bypasses that
version gate for diagnostics and unit tests, but a real smoke run remains
mandatory. See [ADAPTATION_STATUS.md](ADAPTATION_STATUS.md) for the evidence.

## Commands

Render the immutable training config without loading model weights:

```bash
bash run_vlm_lora_smoke.sh --dry-run
```

Run one optimizer step on eight GPUs:

```bash
bash run_vlm_lora_smoke.sh \
  --lora-scope text \
  --max-steps 1 \
  --devices 0,1,2,3,4,5,6,7
```

Use `--lora-scope vision` for the vision tower/projector only or `all` for
both modalities. A real run asserts that the complete
`Qwen3VLMoeForConditionalGeneration` is retained, all 48 decoder MoE layers
are KT-wrapped, a real image reaches `model.visual.patch_embed.proj`, only the
requested LoRA scope is trainable, gradients are finite/non-zero, and an
optimizer step changes a LoRA tensor in every requested modality.

Default development locations can be overridden with:

```bash
export VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LlamaFactory-vlm-pr
export VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel
```
