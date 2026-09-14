# PyTorch oneDNN Hybrid Full-FT

This directory contains an isolated PyTorch BF16 full-parameter fine-tuning
path for Qwen3.5-122B-A10B and GLM-4.5-Air. It mirrors the KTransformers
placement and 8-rank distributed contract without importing KTransformers. Each
pipeline rank owns its virtual decoder stages, the dense trunk (attention,
router, embedding, norm, shared expert and LM head) runs on its GPU, and routed
expert weights and expert matmuls stay on CPU through oneDNN. Every parameter
remains trainable; AdamW moments are explicitly kept on CPU.

The default layout is `PP=8`, `TP=1`, `DP=1`, global batch 8, per-stage
microbatch 1, BF16. To model KT's FSDP-style replicated ranks on one host, use
`PP=4`, `DP=2`, `TP=1` with `--num-gpus 8`; each DP group has its own 4-stage
pipeline and gradients are reduced across corresponding stages. Qwen's 48
layers and GLM's 46 layers are assigned as `layer_id % PP`; only assigned
layers are materialized from the safetensors index on each rank.

The launcher refuses to start while an APTMoE or CUDA benchmark process is
visible, because CPU expert execution can contend for system RAM and CPU time.
Use
`--allow-concurrent` only when that contention is intentional. `--dry-run` is
always safe and only validates the command/data paths.

Examples:

```bash
cd /mnt/data2/wbw/Ktransformers-development/FFTtest/Qwen3.5-122B-A10B
bash run_finetune_perf_test_pytorch_onednn.sh --dry-run

cd /mnt/data2/wbw/Ktransformers-development/FFTtest/GLM-4.5-Air
bash run_finetune_perf_test_pytorch_onednn.sh \
  --seq-lengths 1024,512,256,64,32 --steps 15 --warmup-steps 5 \
  --conda-env Onednn --num-gpus 8 --pipeline-parallel-size 8 \
  --data-parallel-size 1 --tensor-parallel-size 1
```

Set `--dnnl-verbose 1` to ask oneDNN to print dispatched CPU kernels. Each
sequence is a separate 8-rank `torchrun` case and writes `run_config.json`,
`step_timing.{json,csv}`, `summary.md`, and `exit_code.txt` under the model's
`test_log/*PYTORCH_ONEDNN_FULL/seq_*` directory. Use `--devices 0,1,2,3,4,5,6,7`
to select the eight GPUs, or override `--owner-threads` and
`--non-owner-threads` for a controlled comparison. The launcher still protects
active APTMoE jobs from accidental concurrent resource contention.
