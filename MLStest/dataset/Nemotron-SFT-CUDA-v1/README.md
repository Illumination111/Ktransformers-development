---
language:
- en
license:
- cc-by-4.0
size_categories:
- 1K<n<10K
task_categories:
- text-generation
tags:
- text
- code
- software-engineering
- agentic
- supervised-fine-tuning
- post-training
- synthetic
- Nemotron_3_Ultra
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

## Dataset Description:

Nemotron-SFT-CUDA-v1 is a training dataset for CUDA code. It helps language models write CUDA kernels and solve CUDA programming problems.

We start from CUDA code in [Nemotron Pretraining Code v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2), which has a permissive license. An OpenCode agent powered by [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) reads that code and writes new CUDA programming problems. Each problem comes with a hidden answer and tests. A second OpenCode + GLM-4.7 agent then tries to solve the problems using only the prompt and a few helper files. We record the full trace and keep only the runs that pass the tests with no Compute-Sanitizer errors.

This dataset is ready for commercial use.

## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
Created on: 2026-03-04  
Last Modified on: 2026-03-16

## Version:
Nemotron-SFT-CUDA-v1  
Previous Version(s): N/A


## License/Terms of Use: 
This dataset is governed by the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Intended Usage:
This dataset is intended for post-training (supervised fine-tuning) of language models on CUDA tasks. It targets:

* writing CUDA kernels and library calls from a spec;
* reasoning about shapes, memory, indexing, and the CUDA API;
* using compiler errors and test output to fix code.

## Dataset Characterization
**Data Collection Method**  
* [Synthetic]  

**Labeling Method**
* [Synthetic]
* [Automated]

## Dataset Format
Modality: Text  
Format: JSONL  
Structure: Text + Metadata

## Dataset Quantification
Samples: 2,276  
Total Data Storage: ~84 MB.


## Reference(s):
- [nvidia/Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2)
- [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7)

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)