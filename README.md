# Ktransformers Development

Development and validation workspace for KTransformers fine-tuning, multi-LoRA
serving, and vision-language model training.

## Test suites

- `FFTtest/`: full fine-tuning and performance test harnesses.
- `MLStest/`: multi-LoRA serving test harnesses.
- `VLM-FT-test/`: vision-language model fine-tuning tests.

The original Git histories of FFTtest and MLStest are retained as merge
parents. Their original branch tips are also available under
`history/FFTtest/*` and `history/MLStest/*`. VLM-FT-test was not previously
a Git repository and enters this repository as an import commit.

Large local datasets, model adapters, caches, and generated test logs remain
ignored and are not distributed through GitHub.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
