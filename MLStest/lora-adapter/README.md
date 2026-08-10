# LoRA serving composites

训练完成后，`convert_kt_adapter.sh` 会把 KT LoRA run 目录转成 sglang-kt 可用的 merged composite，默认输出到：

```text
lora-adapter/Qwen3.5-397B-A17B/<cuda|swe|cpp>/
lora-adapter/Qwen3.5-35B-A3B/<cuda|swe|cpp>/
  adapter_config.json
  adapter_model.safetensors
```

各 base 必须用**自己**训练出的 composite；仓库只保留目录占位，**不上传** `.safetensors` 权重。
