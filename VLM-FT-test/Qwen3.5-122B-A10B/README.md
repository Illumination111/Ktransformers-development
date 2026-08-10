# Qwen3.5-122B-A10B VLM LoRA smoke test

This directory mirrors the staged structure of `FFTtest/Qwen3.5-122B-A10B`,
but it deliberately keeps the full `Qwen3_5MoeForConditionalGeneration`
instead of installing the FFT text-only loader.

The default data is the six-row local image dataset:

- registry: `/mnt/data2/wbw/LLaMA-Factory/data/dataset_info.json`
- annotation: `/mnt/data2/wbw/LLaMA-Factory/data/mllm_demo.json`
- images: `/mnt/data2/wbw/LLaMA-Factory/data/mllm_demo_data/{1,2,3}.jpg`

Run the weight-free preflight first:

```bash
bash run_vlm_lora_smoke.sh --preflight-only
```

The current Kllama environment uses torch 2.9.1, so the command above is
expected to stop at the Conv3D safety gate. For a diagnostic-only config render:

```bash
bash run_vlm_lora_smoke.sh --dry-run --allow-torch29-conv3d
```

For a real one-step run, use an environment with torch `<2.9` or `>=2.10`, then:

```bash
bash run_vlm_lora_smoke.sh --max-steps 1 --devices 0,1,2,3,4,5,6,7
```

The custom entrypoint fails before training unless all of these are true:

- the full conditional-generation VLM and its `patch_embed.proj` Conv3D remain;
- the complete visual tower is frozen and has no LoRA parameters;
- LoRA parameters exist in the language model;
- all 48 Qwen3.5-MoE decoder layers are wrapped by KT.

This is an integration probe, not a throughput benchmark or a declaration of
family-wide support. Results are written below `test_log/<UTC timestamp>/`.
