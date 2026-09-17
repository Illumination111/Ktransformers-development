# Qwen3-30B-A3B local rollout/score consistency runner

This machine-specific runner follows the contract in
`2026-08-18-397b-sglang-mismatch-to-p2-ar-stage-report.md`:

* **rollout forward** is the SGLang + KT forward used while sampling. The
  test records `meta_info.output_token_logprobs`, including the logprob of
  each token that was actually sampled.
* **score forward** is the HuggingFace + KT teacher-forced forward on the
  *same* `prompt_ids + response_ids`. For response token `y[t]`, it reads
  `logits[prompt_len + t - 1]` and gathers `y[t]`.
* Only the active response mask is compared. Prompt-token logprobs, output
  text, and independently generated responses are not used as consistency
  criteria.

The project-level `../README.md` is the canonical RLFT test documentation.
This file documents the local `/mnt/data2/wbw` runner and environment.

## Run

Use the `kt-RLFT` environment created for the three source repositories:

```bash
cd /mnt/data2/wbw/Ktransformers-development/RLFTtest/Qwen3-30B-A3B
bash run_rollout_score_consistency.sh
```

The default model is `/mnt/data3/models/Qwen3-30B-A3B`; override it with
`MODEL_PATH=/path/to/model`. The script starts an SGLang server, writes its
stdout/stderr and the comparison JSON below `test_log/<UTC timestamp>/`, then
stops the owned server before loading HF+KT, so the two 30B runtimes do not
need to coexist on one GPU. Existing servers can be used with
`SGLANG_BASE_URL=http://127.0.0.1:30000`.

Useful overrides include `MAX_NEW_TOKENS`, `SGLANG_PORT`, `SGLANG_TP_SIZE`,
`KT_NUM_THREADS`, `KT_NUM_GPU_EXPERTS`, `ABS_TOL`, `RATIO_TOL`, and `PROMPT`.

`CUDA_VISIBLE_DEVICES` must be exported by the caller. To manually choose
visible GPUs and use tensor parallelism:

```bash
export CUDA_VISIBLE_DEVICES=<gpu-id-1>,<gpu-id-2>
export SGLANG_TP_SIZE=2
bash run_rollout_score_consistency.sh
```

Alternatively, use a one-command prefix:
`CUDA_VISIBLE_DEVICES=<gpu-id-1>,<gpu-id-2> SGLANG_TP_SIZE=2 bash run_rollout_score_consistency.sh`.
Running an assignment on a line by itself does not export the variable to the
subsequently launched shell. The runner infers the TP size from exported
visible devices when `SGLANG_TP_SIZE` is unset.

The runner redirects HuggingFace, Torch, Triton, pip, CUDA, XDG, and temporary
caches to `/mnt/data2/wbw/.cache/kt-RLFT` by default. Change the root with
`KT_RLFT_CACHE_ROOT=/mnt/data2/wbw/<cache-dir>`; this keeps model/runtime
artifacts off the root filesystem.

`--dry-run` checks paths and prints the exact launch command without loading a
model. A real run requires a CUDA device.

## Pass criteria

The default threshold is `max_abs_logprob_diff <= 1e-3` and
`max_abs(exp(score-rollout)-1) <= 1e-3`. The JSON records all per-token values,
the active mask, token ids, and aggregate statistics. A missing token-level
rollout logprob or a response/token alignment mismatch is an error rather than
a pass with an incomplete comparison.
