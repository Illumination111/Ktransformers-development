# Qwen3.5-122B-A10B VLM LoRA smoke result

## 2026-08-11 one-step full-modality run

- Run id: `20260811T080922Z`
- Result: **PASS**
- Hardware: 8 x NVIDIA GeForce RTX 4090, 49,140 MiB per GPU
- Scope: `all` (language and vision LoRA)
- Model contract: 48 KT-wrapped MoE layers, 256 experts, top-8 routing
- Parameters: 122,131,325,424 total; 21,639,936 trainable (0.0177%)
- Adapter tensors: 774 total; 552 language and 222 vision
- Gradient assertion: 678 LoRA parameters with gradients per rank
- Optimizer assertion: sampled language and vision LoRA parameters both changed
  by approximately `1.0e-4`
- Optimizer steps: 1
- Train runtime: 22.90 seconds
- Train loss: 10.7734
- Saved output: 83 MiB PEFT adapter plus 2.3 GiB fused-expert LoRA state

### Physical memory and VRAM

The 185.03-second launch window produced 163 samples. Host memory follows the
`htop`/`free` top-panel physical-memory formula from `/proc/meminfo`; no process
RSS values are added together.

- Host baseline panel-used: 16,795.46 MiB (16.40 GiB)
- Host peak panel-used: 335,894.71 MiB (328.02 GiB)
- Training peak increase over baseline: 319,099.25 MiB (311.62 GiB)
- Minimum host memory still available: 1,707,082.62 MiB (1,667.07 GiB)
- Eight-GPU aggregate baseline: 11 MiB
- Eight-GPU aggregate peak: 87,234 MiB (85.19 GiB)
- Per-GPU peak: 10,878-10,946 MiB (maximum 10.69 GiB)
- Peak GPU utilization: 100% on every GPU

The raw evidence is retained locally under
`test_log/20260811T080922Z/`: `train.log`, `adapter_validation.log`,
`resource_samples.jsonl`, `resource_summary.json`, the rendered `train.yaml`,
and the saved adapter artifacts.
