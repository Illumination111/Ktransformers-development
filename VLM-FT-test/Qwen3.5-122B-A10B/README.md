# Qwen3.5-122B-A10B VLM LoRA tests

This directory mirrors the staged structure of `FFTtest/Qwen3.5-122B-A10B`,
but it deliberately keeps the full `Qwen3_5MoeForConditionalGeneration`
instead of installing the FFT text-only loader.

The default data is the six-row local image dataset:

- registry: `/mnt/data2/wbw/LLaMA-Factory/data/dataset_info.json`
- annotation: `/mnt/data2/wbw/LLaMA-Factory/data/mllm_demo.json`
- images: `/mnt/data2/wbw/LLaMA-Factory/data/mllm_demo_data/{1,2,3}.jpg`

The directory provides two levels of testing:

- `run_vlm_lora_smoke.sh`: one-step integration smoke test;
- `run_vlm_lora_formal.sh`: 20-step functional/stability test with a final
  two-row evaluation and machine-readable result validation.

Until the companion branches are merged, select their development worktrees:

```bash
export VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LlamaFactory-vlm-pr
export VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel
```

Run the weight-free preflight first:

```bash
bash run_vlm_lora_smoke.sh --preflight-only
```

The Kllama environment uses torch 2.9.1. LLaMA-Factory's additive KT-VLM
integration automatically imports the verified `ms-swift==4.4.2` package in
every rank before model loading. The test entrypoint does not activate the
patch; it checks that the framework did so, while the preflight runs a
Qwen3.5-shaped forward/backward self-test. For a diagnostic-only config render:

```bash
bash run_vlm_lora_smoke.sh --dry-run
```

Install the test dependency without changing torch or Transformers:

```bash
/mnt/data2/wbw/conda/envs/Kllama/bin/python -m pip install "ms-swift==4.4.2"
```

For a released KT package, use `pip install 'ktransformers[vlm-sft]'` for the
complete training stack, or `pip install 'kt-kernel[vlm-sft]'` at the kernel
layer. Both install the same kt-kernel wheel plus an optional Python dependency;
neither is a separate VLM-specific precompiled kernel build. The source-tree
test uses a small module registration shim only because Kllama currently
retains the older matched `ktransformers/kt-kernel 0.6.3.post1` pair.

For a real one-step run:

```bash
bash run_vlm_lora_smoke.sh --lora-scope text --max-steps 1 --devices 0,1,2,3,4,5,6,7
```

Use `--lora-scope vision` for the vision tower and multimodal projector only,
or `--lora-scope all` for text and vision LoRA together. `text` remains the
default for backward compatibility. The runner writes the selection as
`vlm_lora_scope` in the immutable per-run YAML and validates gradients,
optimizer updates, and saved adapter tensors against the requested scope.

A real vision-only run also requires the active kt-kernel Python package to
include `kt_freeze_experts`; the development Conv3D-module shim alone does not
change KT expert training.
The runners fail before loading 122B weights if scoped LLaMA-Factory support is
missing, or if a real vision-only launch imports a kt-kernel without that flag.

For the formal 20-step run:

```bash
bash run_vlm_lora_formal.sh \
  --lora-scope all \
  --model-path /mnt/data2/models/Qwen3.5-122B-A10B \
  --dataset-dir /mnt/data2/wbw/LLaMA-Factory/data \
  --dataset-name mllm_demo \
  --devices 0,1,2,3,4,5,6,7 \
  --max-steps 20 \
  --cutoff-len 512
```

The formal runner deterministically splits the six demo rows into four
training rows and two evaluation rows. It rejects runs shorter than 10
optimizer steps and, after training, checks every logged step for finite loss,
checks that every requested LoRA modality received a non-zero gradient and a
real weight update at every optimizer step, verifies the final eval loss and
saved adapter scope, and writes `formal_summary.json` below the run directory.
Under FSDP2, the callback performs these checks on each rank's local DTensor
shard; it never gathers a complete parameter.

The custom entrypoint fails before training unless all of these are true:

- the full conditional-generation VLM and its `patch_embed.proj` Conv3D remain;
- the ms-swift Conv3D patch is active in the same rank before model loading;
- no visual base parameter is trainable;
- LoRA parameters exist only in the requested text/vision scope;
- all 48 Qwen3.5-MoE decoder layers are wrapped by KT;
- a real image reaches the visual PatchEmbed;
- every requested LoRA modality receives a finite non-zero gradient;
- an optimizer step changes the sampled LoRA parameter in every requested modality;
- the saved adapter LoRA tensors match the requested scope.

`mllm_demo` is sufficient for these functional integration and short stability
tests because every row has real image references, user image placeholders and
non-empty assistant targets. Six rows are not sufficient for convergence,
quality or throughput claims. Smoke results are written below
`test_log/<UTC timestamp>/`; formal results are written below
`formal_test_log/<UTC timestamp>/`.

See `../docs/task_bash_Qwen3.5-122B-A10B-VLM-LoRA.md` for copy-paste launch
commands and acceptance criteria.
