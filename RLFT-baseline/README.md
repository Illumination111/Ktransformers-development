# Qwen3-30B-A3B veRL GRPO B0

本目录是原生 veRL 基线的唯一实验根目录。固定 veRL worktree 位于 `worktree/`，自定义代码只放在 `scripts/`、`configs/` 等相邻目录，不修改 worktree。

当前边界：使用原生 veRL，不启用 KTransformers；actor/ref 为 FSDP2，rollout 为 SGLang。当前执行配置已收缩为 2 张 H20 的 NekoQA 纯 GPU 计算方案，CPU offload 只用于停放非活跃参数和优化器状态。此前的 8 卡调试结果只作为历史记录保留，最终两卡 persona smoke 已完成，正式 60 步尚未启动。

## 固定实验

- veRL：`68e9571a9eaa20ee44954567f507a92c986e9db0`
- 模型：`/mnt/qjh007/models/Qwen3-30B-A3B`
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

## 数学 GRPO 双卡诊断实验

本实验使用本地 `math500.parquet` 做确定性 400/100 切分，AIME-2024 保持为外部评测，不参与训练。MATH-500 本身是 benchmark，因此该切分只用于本地链路和学习趋势诊断，不能作为无污染的公开成绩。

```bash
# 只需在数据切分不存在时执行一次
python scripts/prepare_math_grpo_split.py

# 两张 H20；thinking 开启，数学 verifier，独立输出目录
bash scripts/run_math_grpo_puregpu.sh smoke
bash scripts/run_math_grpo_puregpu.sh pilot
bash scripts/run_math_grpo_puregpu.sh formal
```

数学模型默认沿用 `B0_MODEL`；可用 `MATH_MODEL_PATH` 指向经过 SFT 的本地 checkpoint。若已有兼容的 attention LoRA，可额外设置 `MATH_LORA_ADAPTER_PATH` 做 warm-start；两者均会写入 resolved config，不与 NekoQA checkpoint 混用。

当前 `/mnt/qjh007/models/Qwen3-30B-A3B` 的模型卡标注为 `Pretraining & Post-training`，不是严格意义的 SFT-only 起点。要归因 GRPO 增益，应将 `MATH_MODEL_PATH` 设置为只经过 SFT 的 checkpoint；默认路径只适合先验证工程链路或做产品侧确认实验。

数学入口采用 `FSDP2 + SGLang TP=2 + LoRA(q/k/v/o)`，默认 `batch=16`、`rollout.n=4`、response 上限 2048、训练 30 步。两卡仍启用 CPU offload 以停放非活跃参数和 optimizer state；不要把该配置表述为全参数 GPU 常驻。

`smoke` 是接口/显存 gate：仅 2 步、512 response 上限且跳过验证，因此常见结果是 reward=0（模型尚未生成最终答案），不能据此判断 GRPO 学习效果。要观察非零数学奖励和趋势，应在 smoke 通过后运行 `pilot`，再考虑 `formal`。

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
