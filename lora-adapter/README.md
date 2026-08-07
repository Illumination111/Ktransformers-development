# LoRA serving composites

训练完成后，`convert_kt_adapter.sh` 会把 KT LoRA run 目录转成 sglang-kt 可用的 merged composite，默认输出到：

```text
lora-adapter/Qwen3.5-397B-A17B/<cuda|swe|cpp>/
  adapter_config.json
  adapter_model.safetensors
```

仓库只保留目录占位，**不上传** `.safetensors` 权重。
