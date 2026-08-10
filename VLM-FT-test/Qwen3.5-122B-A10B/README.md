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

Run the weight-free preflight first:

```bash
bash run_vlm_lora_smoke.sh --preflight-only
```

The Kllama environment uses torch 2.9.1. The runner imports the verified
`ms-swift==4.4.2` Python package in every rank before model loading and checks
its `swift.model.utils` Conv3D replacement with a Qwen3.5-shaped forward and
backward self-test. For a diagnostic-only config render:

```bash
bash run_vlm_lora_smoke.sh --dry-run
```

Install the test dependency without changing torch or Transformers:

```bash
/mnt/data2/wbw/conda/envs/Kllama/bin/python -m pip install "ms-swift==4.4.2"
```

For a real one-step run:

```bash
bash run_vlm_lora_smoke.sh --max-steps 1 --devices 0,1,2,3,4,5,6,7
```

For the formal 20-step run:

```bash
bash run_vlm_lora_formal.sh \
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
checks that the callback observed a non-zero language-LoRA gradient and a real
LoRA weight update at every optimizer step, verifies the final eval loss and
saved adapter, and writes `formal_summary.json` below the run directory. Under
FSDP2, the callback performs these checks on each rank's local DTensor shard;
it never gathers a complete parameter.

The custom entrypoint fails before training unless all of these are true:

- the full conditional-generation VLM and its `patch_embed.proj` Conv3D remain;
- the ms-swift Conv3D patch is active in the same rank before model loading;
- the complete visual tower is frozen and has no LoRA parameters;
- LoRA parameters exist in the language model;
- all 48 Qwen3.5-MoE decoder layers are wrapped by KT.
- a real image reaches the visual PatchEmbed;
- at least one language LoRA parameter receives a finite non-zero gradient;
- an optimizer step changes the sampled LoRA parameter.
- the saved adapter contains LoRA tensors and no frozen-vision LoRA tensors.

`mllm_demo` is sufficient for these functional integration and short stability
tests because every row has real image references, user image placeholders and non-empty assistant
targets. Six rows are not sufficient for convergence, quality or throughput
claims. Smoke results are written below `test_log/<UTC timestamp>/`; formal
results are written below `formal_test_log/<UTC timestamp>/`.

See `../docs/task_bash_Qwen3.5-122B-A10B-VLM-LoRA.md` for copy-paste launch
commands and acceptance criteria.
