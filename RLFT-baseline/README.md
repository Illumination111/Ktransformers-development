# Qwen3-30B-A3B veRL GRPO B0

本目录是原生 veRL 基线的唯一实验根目录。固定 veRL worktree 位于 `worktree/`，自定义代码只放在 `scripts/`、`configs/` 等相邻目录，不修改 worktree。

当前边界：已建立 worktree、Conda 环境、固定数据和全部代码入口；已在 GPU 1--4 上完成原始 checkpoint 的 MATH-500 step-0 avg@4，按要求暂停，尚未运行 AIME-2024、FSDP2 smoke/pilot 或 GRPO 训练。

## 固定实验

- veRL：`cb59290ecd85565cdf200d855b45b7080f7cf34c`
- 模型：`/mnt/qjh007/models/Qwen3-30B-A3B`
- 算法/训练/rollout：GRPO + FSDP2 + vLLM
- 数据：DAPO-Math-17k 的确定性 2048 train + 256 validation 子集
- 主训练：15 epochs，理论 60 optimization steps
- 外部评测：MATH-500 和 AIME-2024

完整协议见 `/home/wubowen/development-docs/qwen3-30b-a3b-verl-grpo-b0-plan-2026-08-21.md`。

## 执行顺序

```bash
# 1. 建 Conda 环境 kt-rflt-baseline（需要网络）
bash scripts/setup_env.sh

# 2. 下载/固定数据 revision 后生成数据
/home/wubowen/miniconda3/envs/kt-rflt-baseline/bin/python scripts/prepare_data.py --help

# 3. 只做配置预览，不启动训练
bash scripts/run_b0.sh formal --dry-run

# 4. GPU 阶段依次执行，三者都从原始模型独立开始
bash scripts/run_b0.sh smoke
bash scripts/run_b0.sh pilot
bash scripts/run_b0.sh formal
```

`smoke`、`pilot`、`formal` 使用不同输出目录且强制 `resume_mode=disable`。任何阶段都不会把前一阶段 checkpoint 当作 warm start。

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
