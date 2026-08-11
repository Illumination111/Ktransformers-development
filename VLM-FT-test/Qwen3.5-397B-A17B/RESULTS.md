# Qwen3.5-397B-A17B VLM LoRA smoke result

## 2026-08-11 one-step full-modality run

- Run id: `20260811T074953Z`
- Result: **PASS**
- Hardware: 8 x NVIDIA GeForce RTX 4090, 49,140 MiB per GPU
- Scope: `all` (language and vision LoRA)
- Model contract: 60 KT-wrapped MoE layers, 512 experts, top-10 routing
- Parameters: 396,076,145,904 total; 29,005,568 trainable (0.0073%)
- Adapter tensors: 912 total; 690 language and 222 vision
- Gradient assertion: 792 LoRA parameters with gradients per rank
- Optimizer assertion: sampled language and vision LoRA parameters both changed
  by approximately `1.0e-4`
- Optimizer steps: 1
- Train runtime: 51.46 seconds
- Train loss: 10.6328
- Saved output: 111 MiB PEFT adapter plus 7.1 GiB fused-expert LoRA state

### Physical memory and VRAM

The 458.44-second launch window produced 400 samples. Host memory follows the
`htop`/`free` top-panel physical-memory formula from `/proc/meminfo`; no process
RSS values are added together.

- Host baseline panel-used: 17,168.01 MiB (16.77 GiB)
- Host peak panel-used: 907,379.69 MiB (886.11 GiB)
- Training peak increase over baseline: 890,211.68 MiB (869.35 GiB)
- Minimum host memory still available: 1,135,659.49 MiB (1,109.04 GiB)
- Eight-GPU aggregate baseline: 11 MiB
- Eight-GPU aggregate peak: 113,870 MiB (111.20 GiB)
- Per-GPU peak: 14,220-14,300 MiB (maximum 13.96 GiB)
- Peak GPU utilization: 100% on every GPU

The raw evidence is retained locally under
`test_log/20260811T074953Z/`: `train.log`, `adapter_validation.log`,
`resource_samples.jsonl`, `resource_summary.json`, the rendered `train.yaml`,
and the saved adapter artifacts.
