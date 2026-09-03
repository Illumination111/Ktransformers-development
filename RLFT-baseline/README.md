# Qwen3-30B-A3B veRL GRPO B0

本目录是原生 veRL 基线的唯一实验根目录。固定 veRL worktree 位于 `worktree/`，自定义代码只放在 `scripts/`、`configs/` 等相邻目录，不修改 worktree。

当前边界：使用原生 veRL，不启用 KTransformers；actor/ref 为 FSDP2，rollout 为 SGLang。当前执行配置已收缩为 2 张 H20 的 NekoQA 纯 GPU 计算方案，CPU offload 只用于停放非活跃参数和优化器状态。此前的 8 卡调试结果只作为历史记录保留，最终两卡 persona smoke 已完成，正式 60 步尚未启动。

## 固定实验

- veRL：`68e9571a9eaa20ee44954567f507a92c986e9db0`
- 模型：`/mnt/qjh007/models/Qwen3-30B-A3B-Instruct-2507`
- 算法/训练/rollout：GRPO + FSDP2 + SGLang 0.5.8
- 数据：NekoQA-10K 的确定性 2048 train + 256 validation 子集
- reward：`scripts/nekoqa_reward.py`（字符重合度 + 猫娘风格标记）
- 模板：固定猫娘 system prompt + Qwen3 `enable_thinking=False`，将生成预算用于简洁角色回答
- 资源：默认 `CUDA_VISIBLE_DEVICES=0,1`，SGLang TP=2
- 主训练：显式 60 optimization steps；batch=32，rollout n=4，PPO mini-batch=16
- 长度：prompt=1024，response=512，单卡 token budget=8192
- 外部评测：MATH-500 和 AIME-2024

完整协议见 `/home/wubowen/development-docs/01-baseline/grpo-baseline.md`。

## 执行顺序

```bash
# 1. 建 Conda 环境 kt-rflt-baseline（需要网络）
bash scripts/setup_env.sh

# 2. 下载/固定 NekoQA revision 后生成数据（当前 parquet 已生成）
/home/wubowen/miniconda3/envs/kt-rflt-baseline/bin/python scripts/prepare_data.py \
  --dataset-id liumindmind/NekoQA-10K \
  --dataset-revision 1b2110c996a8237823b86c1a3d3e8a6762b38430 \
  --math500-revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --aime2024-revision aa49075e24ad594b79fdf0bdcefa735c2181be67

# 3. 只做配置预览，不启动训练
bash scripts/run_b0_puregpu.sh formal --dry-run

# 4. GPU 阶段依次执行，三者都从原始模型独立开始
bash scripts/run_b0_puregpu.sh smoke
bash scripts/run_b0_puregpu.sh pilot
bash scripts/run_b0_puregpu.sh formal
```

## 数学 GRPO v2 四卡能力实验

旧版 `math_grpo_2gpu_{smoke,pilot,formal}` 使用 MATH-500 400/100 本地切分，只保留为历史诊断。v2 不再用 benchmark 训练：训练集固定为官方 `BytedTsinghua-SIA/DAPO-Math-17k@65877096c24ffa7abc4e4fa5edb95cf3413a5674`，按稳定 source ID 去重并剔除与 MATH-500/AIME-2024 的 prompt 重合；完整 MATH-500 与 AIME-2024 只用于 frozen held-out 评测。

```bash
# 1. 下载、去重并生成 hard-math 训练集与固定 MATH-500 validation gate
/home/wubowen/miniconda3/envs/kt-rflt-baseline/bin/python scripts/prepare_hard_math_data.py

# 2. 用当前模型比较 4096/8192/10240 长度；formal 默认要求该 gate 已通过
bash scripts/run_math_static_gate.sh

# 3. 用完全相同协议记录训练前 MATH-500/AIME
bash scripts/run_math_eval.sh math-v2-step0 /path/to/hf-model

# 4. 四张 GPU；三个阶段均从指定模型独立开始
bash scripts/run_math_grpo_puregpu.sh smoke
bash scripts/run_math_grpo_puregpu.sh pilot
bash scripts/run_math_grpo_puregpu.sh formal

# 5. 将 final checkpoint 导出为可加载的 HF 模型后，以同协议评测
bash scripts/run_math_eval.sh math-v2-final /path/to/final-hf-model
```

数学模型和通用 RL 入口默认使用 `/mnt/qjh007/models/Qwen3-30B-A3B-Instruct-2507`，并允许用 `MATH_MODEL_PATH` 显式覆盖。若已有兼容的 attention LoRA，可额外设置 `MATH_LORA_ADAPTER_PATH` 做 warm-start。

## 数学 GRPO 前置 SFT（可选）

veRL 的 SFT 是独立入口，不能在 GRPO 配置中直接打开。当前 DAPO-Math parquet 只有题目和 verifier 的最终答案，没有教师推理轨迹，因此本流程只训练 `Answer: <answer>` 输出约定，不是推理蒸馏。

若要一条命令串行执行完整流程，可运行：

```bash
source configs/b0.env
bash scripts/run_math_sft_grpo_pipeline.sh
```

总控脚本会依次执行数据准备、SFT、checkpoint 导出、10240 长度门禁以及 GRPO formal；任一步失败都会停止。

```bash
# Load B0_ROOT/B0_* paths used below.
source configs/b0.env

# 1. 从 DAPO-Math GRPO 数据生成 veRL messages 格式（默认 1% 确定性验证集）
/home/wubowen/miniconda3/envs/kt-rflt-baseline/bin/python scripts/prepare_math_sft_data.py

# 2. 四卡 FSDP2 + LoRA SFT；默认一轮，输出到 checkpoints/math_sft_qwen3_30b_a3b_lora
bash scripts/run_math_sft_puregpu.sh

# 3. 读取最终 global_step 并导出 HF 目录，同时生成 lora_adapter/
SFT_CKPT="$B0_ROOT/checkpoints/math_sft_qwen3_30b_a3b_lora"
SFT_STEP=$(cat "$SFT_CKPT/latest_checkpointed_iteration.txt")
bash scripts/merge_math_sft_checkpoint.sh "$SFT_STEP"

# 4. 静态长度门禁（门禁只验证长度容量；SFT adapter 在 GRPO 阶段加载）
bash scripts/run_math_static_gate.sh

# 5. GRPO 从 SFT adapter warm-start；三个阶段仍各自 resume_mode=disable
SFT_HF="$SFT_CKPT/global_step_${SFT_STEP}/actor/huggingface"
MATH_MODEL_PATH="$B0_MATH_MODEL" \
MATH_LORA_ADAPTER_PATH="$SFT_HF/lora_adapter" \
bash scripts/run_math_grpo_puregpu.sh smoke
MATH_MODEL_PATH="$B0_MATH_MODEL" \
MATH_LORA_ADAPTER_PATH="$SFT_HF/lora_adapter" \
bash scripts/run_math_grpo_puregpu.sh pilot
MATH_MODEL_PATH="$B0_MATH_MODEL" \
MATH_LORA_ADAPTER_PATH="$SFT_HF/lora_adapter" \
bash scripts/run_math_grpo_puregpu.sh formal
```

当前 `/mnt/qjh007/models/Qwen3-30B-A3B` 的模型卡标注为 `Pretraining & Post-training`，不是严格意义的 SFT-only 起点。要归因 GRPO 增益，应将 `MATH_MODEL_PATH` 设置为只经过 SFT 的 checkpoint；默认路径只适合先验证工程链路或做产品侧确认实验。

数学入口采用 `FSDP2 + SGLang TP=4 + LoRA(q/k/v/o)`。v2 formal 使用 prompt batch 4、`rollout.n=8`、response 10240、最多 80 个 physical step、LR `1e-5`、KL `0.001`；训练前验证已启用。四卡仍启用 CPU offload 以停放非活跃参数和 optimizer state。

训练和评测共用 `scripts/math_verify_reward.py`：总 reward 始终为最终答案 correctness 0/1，`format_ok`、`answer_extracted`、`parser_error` 只记录诊断，不参与 reward。prompt 明确要求最后一行使用 `Answer: <answer>`，verifier 同时兼容 boxed 数学表达并检查数学等价。

`smoke` 是 2-step 工程/显存 gate；`pilot` 是 8-step 方向门禁；`formal` 是 80-step 上限的能力实验。formal 完成后会自动运行 `audit_math_rollouts.py`，只有至少 50 个 step 含有组内 correctness 方差且重编码截断率不高于 50% 才通过验收。当前固定 veRL commit 的同步 trainer 没有实现 `algorithm.filter_groups` 补采逻辑，因此 v2 使用首选的 `n=8`，并以 fail-closed 审计防止把零信号 physical step 当作有效 step。

独立评测默认对 MATH-500 使用 8 seeds、AIME-2024 使用 16 seeds，统一 temperature 1.0、8192 response、同一 verifier，并保存逐题结果、token 数、finish reason、格式/解析诊断和 bootstrap 区间。`summarize_b0.py` 会拒绝汇总数据 hash 或采样协议不同的 step-0/final 报告。

`smoke`、`pilot`、`formal` 使用不同输出目录且强制 `resume_mode=disable`。任何阶段都不会把前一阶段 checkpoint 当作 warm start。

默认使用物理 GPU 0、1。若要换成其他一对卡，应同时保持两张卡空闲，例如：

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_b0_puregpu.sh smoke
```

## 目录约定

- `configs/`：人工维护的固定变量和协议；
- `data/raw/`：下载缓存或原始数据；
- `data/processed/`：最终 parquet；
- `data/manifests/`：revision、样本 ID、哈希和统计；
- `logs/`：阶段日志；
- `checkpoints/`：互相隔离的 checkpoint；
- `eval/`：外部 benchmark 原始生成；
- `metrics/`：机器可读汇总；
- `reports/`：最终报告；
- `worktree/`：固定 veRL 源码，必须保持 clean。

数据缓存、处理后 parquet、Conda 辅助二进制、嵌套 worktree、checkpoint、日志和原始评测生成均为本机可再生产物，不纳入 Git。代码版本、依赖版本、数据 revision、样本选择和模型 shard 哈希分别保存在 `configs/`、`env/` 与 `data/manifests/` 中。

当前 MATH-500 原始结果保存在本机 `eval/step0/math500.json`，完整日志保存在 `logs/eval-step0/math500.log`。汇总指标为 `avg@1=0.6220`、`avg@4=0.6005`，4,096-token 截断率为 `0.4380`；后续恢复前应先处理或解释高截断率。
