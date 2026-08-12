# Qwen3-VL adaptation status

Checked on 2026-08-12 against:

- checkpoint: `/mnt/data3/models/Qwen3-VL-30B-A3B-Instruct`;
- LLaMA-Factory: `/mnt/data2/wbw/LlamaFactory-vlm-pr`;
- KTransformers source: `/mnt/data2/wbw/ktransformers-vlm-pr/kt-kernel`;
- Python: `/mnt/data2/wbw/conda/envs/Kllama/bin/python`.

## Result

The installed stack has **not completed Qwen3-VL-MoE adaptation**.

| Layer | Result | Evidence |
| --- | --- | --- |
| Checkpoint | Pass | `Qwen3VLMoeForConditionalGeneration`, 48 layers, 128 experts, top-8, 13 non-empty referenced shards |
| Processor/data | Pass | Local `Qwen3VLProcessor` emitted `pixel_values` and `image_grid_thw`; all six demo rows and eight image references validated |
| LLaMA-Factory source | Pass (development source) | `qwen3_vl` template, `qwen3_vl_moe` composite registration, scoped VLM LoRA and KT Conv3D path are present |
| KT Conv3D | Pass | torch 2.9.1 + ms-swift 4.4.2 self-test passed with zero numerical difference |
| Installed kt-kernel | Fail | Rejects `Qwen3VLMoeForConditionalGeneration` as unsupported |
| KT development source | Pass (not installed) | Resolves `model.language_model.layers`, 128 experts, MoE width 768 and top-8 |
| Transformers/LLaMA-Factory version gate | Warning | Installed Transformers 5.6.0 is explicitly excluded by this LLaMA-Factory revision; tests require `DISABLE_VERSION_CHECK=1` |
| Real distributed training | Not run | CUDA is not visible in the current execution context; an 8-GPU smoke run remains the final integration gate |

The runner explicitly activates only the development KT Python architecture
dispatch before importing LLaMA-Factory. This is suitable for validating the
pending KT change; it is not a substitute for installing a packaged kt-kernel
that contains the adaptation.

## Verified commands

```bash
bash run_vlm_lora_smoke.sh --preflight-only
bash run_vlm_lora_smoke.sh --dry-run --lora-scope all
```

The local Qwen3-VL test module passed four tests, including construction of a
tiny real `Qwen3VLMoeForConditionalGeneration` and discovery of disjoint text
and vision LoRA targets. The selected LLaMA-Factory development worktree also
passed its 30 KT-VLM/scoped-LoRA unit tests.

## Remaining acceptance command

```bash
bash run_vlm_lora_smoke.sh \
  --lora-scope all \
  --max-steps 1 \
  --devices 0,1,2,3,4,5,6,7
```

A successful run must print `qwen3vl_contract OK`, `GRADIENT_OK`,
`OPTIMIZER_OK`, and `PASS`, then save an adapter whose tensors cover exactly
the requested modality scope.
