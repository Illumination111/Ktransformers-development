# Qwen3.5-122B-A10B VLM LoRA tests

The one-step full-modality resource-instrumented test passed on 2026-08-11.
See [RESULTS.md](RESULTS.md) for the parameter-update proof and measured
RAM/VRAM peaks.

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
export VLM_LLAMA_FACTORY_DIR=/mnt/data2/wbw/LLaMA-Factory
export VLM_KT_SOURCE_DIR=/mnt/data2/wbw/ktransformers/kt-kernel
```

Run the weight-free preflight first:

```bash
bash run_vlm_lora_smoke.sh --preflight-only
```

The Kllama environment uses torch 2.9.1. KTransformers patches only the
supported Conv3D instances on the loaded VLM and marks them before returning
the model. The test entrypoint checks those markers, while the preflight runs a
Qwen3.5-shaped forward/backward self-test. For a diagnostic-only config render:

```bash
bash run_vlm_lora_smoke.sh --dry-run
```

No VLM-only Python dependency is required. For a released KT package, use the
same `pip install 'ktransformers[sft]'` command as text SFT. The source-tree test
uses a small wrapper hook only because Kllama currently
retains the older matched `ktransformers/kt-kernel 0.6.3.post1` pair.

For a real one-step run:

```bash
bash run_vlm_lora_smoke.sh --lora-scope text --max-steps 1 --devices 0,1,2,3,4,5,6,7
```

The current runners intentionally cover only LlamaFactory's ordinary text-side
VLM LoRA path. They explicitly freeze the vision tower and multimodal projector;
the former `vision`/`all` scope experiment is outside the current PR.

For the formal 20-step run:

```bash
bash run_vlm_lora_formal.sh \
  --lora-scope text \
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
- every Conv3D instance carries the KT compatibility marker in the same rank;
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

New smoke and formal runs also write `resource_samples.jsonl` and
`resource_summary.json`. Host memory uses the `htop`/`free` top-panel physical
memory value and never sums process RSS; the training increment is peak minus
the pre-launch baseline. GPU memory comes from `nvidia-smi`. The existing
`formal_test_log/20260811T052157Z` run predates this monitor and therefore has
no auditable RAM or VRAM samples/peaks.

See `../docs/task_bash_Qwen3.5-122B-A10B-VLM-LoRA.md` for copy-paste launch
commands and acceptance criteria.
