# PyTorch CUDA 8-GPU Full-FT

This backend is separate from the CPU oneDNN and APTMoE paths. It uses the
`Onednn` runtime environment, launches one process per GPU with `torchrun`, and
uses Hugging Face Trainer FSDP full-shard wrapping around the native model.
Both Qwen3.5-122B-A10B and GLM-4.5-Air keep `finetuning_type: full`; Qwen also
keeps the existing text-only model contract.

Create the isolated environment once:

```bash
cd /mnt/data2/wbw/Ktransformers-development/FFTtest
bash pytorch_cuda/setup_onednn_env.sh
```

The setup first attempts an offline Conda clone. If package archives are not
cached, it falls back to a same-filesystem copy of `Aptmoe` under the distinct
`Onednn` prefix; the source environment is not modified.

Run a configuration check:

```bash
cd Qwen3.5-122B-A10B
bash run_finetune_perf_test_pytorch_cuda_8gpu.sh --dry-run

cd ../GLM-4.5-Air
bash run_finetune_perf_test_pytorch_cuda_8gpu.sh --dry-run
```

Run the 8-GPU sweep after confirming the driver and memory budget:

```bash
bash run_finetune_perf_test_pytorch_cuda_8gpu.sh \
  --seq-lengths 1024,512,256,64,32 --steps 15 --warmup-steps 5 \
  --threads 12 --devices 0,1,2,3,4,5,6,7 --conda-env Onednn
```

The launcher refuses to run if another APTMoE, oneDNN, or CUDA benchmark is
active. It writes one independent case under
`test_log/*PYTORCH_CUDA_8GPU_FULL/seq_*`, including the generated YAML,
`run_config.json`, `train.log`, step timing, and exit code.

Full-FT Adam states are substantially larger than BF16 weights. FSDP shards
parameters, gradients, and optimizer state across eight ranks, but this path
does not silently offload to CPU or switch to DeepSpeed; an out-of-memory case
is a valid capacity result and is preserved in its case log.
