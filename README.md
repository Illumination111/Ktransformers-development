# Multi-LoRA-Serving test for KT

Public test harness and scripts for **sglang-kt Multi-LoRA Serving (M1)** on Qwen3.5-397B-A17B.

## Scope

- Same-process multi-LoRA serving (composite adapters)
- One adapter type per batch (M1); M2 mixed-token is out of scope
- Conda env: `kt-kernel`

## Layout

| Path | Description |
|---|---|
| `Qwen3.5-397B-A17B/` | Serve / client / e2e scripts, configs, train helpers |
| `docs/` | Task notes and runbooks |
| `lora-adapter/` | Adapter output directory (local) |
| `dataset/` | Hugging Face datasets (gitignored; download separately) |

## License

[MIT](LICENSE)
