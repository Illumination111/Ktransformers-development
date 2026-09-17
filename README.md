# Ktransformers Development

Development and validation workspace for KTransformers fine-tuning, multi-LoRA
serving, and vision-language model training.

## Test suites

- `FFTtest/`: full fine-tuning and performance test harnesses.
- `MLStest/`: multi-LoRA serving test harnesses.
- `VLM-FT-test/`: vision-language model fine-tuning tests.

For VLM LoRA module selection, configuration examples, and upstream guide
corrections, see [VLM LoRA configuration notes](VLM-FT-test/docs/VLM-LoRA-%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9D%97%E4%B8%8E%E9%85%8D%E7%BD%AE%E8%AF%B4%E6%98%8E.md).

The original Git histories of FFTtest and MLStest are retained as merge
parents. Their original branch tips are also available under
`history/FFTtest/*` and `history/MLStest/*`. VLM-FT-test was not previously
a Git repository and enters this repository as an import commit.

Large local datasets, model adapters, caches, and generated test logs remain
ignored and are not distributed through GitHub.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
