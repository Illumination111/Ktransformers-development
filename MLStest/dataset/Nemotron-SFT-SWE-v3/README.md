---
language:
- en
license:
- cc-by-4.0
- apache-2.0
- mit
- bsd-3-clause
- bsd-2-clause
task_categories:
- text-generation
tags:
- tool-use
- supervised-fine-tuning
- blend
- code
- SWE
- Nemotron_3_Ultra
- text
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.parquet
size_categories:
- 100K<n<1M
---

## Dataset Description:

Nemotron-SFT-SWE-v3 is a software engineering instruction tuning dataset designed to advance the capabilities of LLMs on SWE-Bench style tasks. 
It includes agentic trajectories collected using a variety of agent harnesses, including the OpenHands, SWE-agent, and mini-SWE-agent frameworks.

This dataset is ready for commercial use.

## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
Created on: 2026-06-04 <br>
Last Modified on: 2026-06-04

## License/Terms of Use: 
This dataset is licensed under Creative Commons Attribution 4.0 International (CC-BY 4.0).  
Additional Information: Apache 2.0 License; MIT License; BSD-3 License; BSD-2 License.

## Intended Usage:

This dataset is intended for LLM engineers and research teams building autonomous software engineering agents and code-focused assistants. It is suitable for 
supervised fine-tuning and distillation of models that must interpret real-world issue statements, plan multi-step tool use, navigate codebases, and implement fixes in 
a SWE-Bench–style setting. The trajectories can also be used to benchmark and debug agent policies, improve repository-aware reasoning, and study robust, regression-free 
code editing behaviors in both academic and production environments.

## Dataset Characterization
**Data Collection Method** <br>
Hybrid: Automated, Synthetic  

**Labeling Method** <br>
Hybrid: Automated, Synthetic 

## Dataset Format  
Modality: Text  
Format: JSONL  
Structure: Text + Metadata

## Dataset Quantification

| Subset | Samples |
|--------|---------|
| Total            | 237,970 |

Total Data Storage: ~11.7GB

## Reference(s):

* [Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139)
* [R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents](https://arxiv.org/abs/2504.07164)
* [SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](https://arxiv.org/abs/2505.20411)
* [The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents](https://arxiv.org/abs/2511.03690)
* [Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models](https://arxiv.org/pdf/2512.13607)

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)