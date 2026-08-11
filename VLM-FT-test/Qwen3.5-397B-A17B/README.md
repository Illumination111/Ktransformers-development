# Qwen3.5-397B-A17B VLM LoRA test

The one-step full-modality test passed on 2026-08-11. See [RESULTS.md](RESULTS.md)
for the parameter-update proof and measured RAM/VRAM peaks.

This test item reuses the validated Qwen3.5 VLM/KT harness and supplies the
397B-A17B checkpoint contract: 60 language layers, 512 experts, top-10 routing,
and 60 KT-wrapped MoE layers. The smoke test defaults to `all`, so both language
and vision LoRA adapters must receive non-zero gradients and change after a real
optimizer step.

## One-step functional test

```bash
cd /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/Qwen3.5-397B-A17B
bash run_vlm_lora_smoke.sh --lora-scope all --max-steps 1 --cutoff-len 512
```

Use `--preflight-only` to validate the local checkpoint, dataset, processor,
Conv3D compatibility and KT architecture without loading model weights. Use
`--dry-run` to additionally render the YAML and print the eight-GPU launch
command.

The default inputs are:

- checkpoint: `/mnt/data2/models/Qwen3.5-397B-A17B`
- dataset: `mllm_demo` under `/mnt/data2/wbw/LlamaFactory-vlm-pr/data`
- LLaMA-Factory: `/mnt/data2/wbw/LlamaFactory-vlm-pr`
- KT source: `/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel`

Each real run creates a UTC timestamped directory under `test_log/` containing
the rendered `train.yaml`, `train.log`, adapter artifacts and validation log.
The callback makes the run fail unless a real image batch reaches the vision
encoder, both modality groups have non-zero LoRA gradients, and sampled LoRA
parameters differ after the optimizer step.

Resource files are also generated:

- `resource_samples.jsonl`: one-second system RAM and per-GPU samples.
- `resource_summary.json`: baseline, peak and delta RAM/VRAM values.
- `resource_summary.log`: human-readable copy of the summary.

Host RAM uses the physical-memory `used` value in the `htop`/`free` top panel,
derived from `/proc/meminfo`; process RSS values are never added together. GPU
memory uses the driver values shown by `nvidia-smi` for the selected physical
GPU indices. The reported peak minus the pre-launch baseline is the training
framework's actual incremental usage when the server is otherwise idle.

## Multi-step stability test

```bash
bash run_vlm_lora_formal.sh --lora-scope all --max-steps 20 --cutoff-len 512
```

This is optional after the one-step functional test and is much more expensive.
It uses the same resource sampling and writes under `formal_test_log/`.
