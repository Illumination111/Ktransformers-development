#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=1
    shift
fi
[ "$#" -eq 0 ] || die "usage: $0 [--dry-run]"

contract_dir="$B0_ROOT/metrics/logprob_contract"
prompt_manifest="$contract_dir/prompt.json"
rollout_result="$contract_dir/vllm_rollout.json"
fsdp_result="$contract_dir/fsdp2_score.json"
commands=(
    "$B0_CONDA_PREFIX/bin/python $script_dir/make_contract_prompt.py --model-path $B0_MODEL --output $prompt_manifest"
    "CUDA_VISIBLE_DEVICES=0,1,2,3 $B0_CONDA_PREFIX/bin/python $script_dir/vllm_contract_rollout.py --manifest $prompt_manifest --output $rollout_result --tensor-parallel-size 4 --gpu-memory-utilization 0.6"
    "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $B0_CONDA_PREFIX/bin/torchrun --standalone --nproc_per_node=8 $script_dir/fsdp_contract_score.py --rollout $rollout_result --output $fsdp_result"
)

if [ "$dry_run" -eq 1 ]; then
    printf '%s\n' "${commands[@]}"
    exit 0
fi

require_clean_worktree
require_conda_env
require_dir "$B0_MODEL"
visible_gpus=$(count_visible_gpus)
[ "$visible_gpus" -eq 8 ] || die "logprob contract requires 8 visible GPUs, got $visible_gpus"
while IFS= read -r used_mib; do
    [ "$used_mib" -lt 1024 ] || die "at least one GPU already uses ${used_mib} MiB"
done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

mkdir -p "$contract_dir"
export PATH="$B0_CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$script_dir:$B0_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="$B0_SEED"

"$B0_CONDA_PREFIX/bin/python" "$script_dir/make_contract_prompt.py" --model-path "$B0_MODEL" --output "$prompt_manifest"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B0_CONDA_PREFIX/bin/python" "$script_dir/vllm_contract_rollout.py" \
    --manifest "$prompt_manifest" --output "$rollout_result" \
    --tensor-parallel-size 4 --gpu-memory-utilization 0.6
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$B0_CONDA_PREFIX/bin/torchrun" --standalone --nproc_per_node=8 \
    "$script_dir/fsdp_contract_score.py" --rollout "$rollout_result" --output "$fsdp_result"

"$B0_CONDA_PREFIX/bin/python" - "$fsdp_result" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
comparison = result["comparison"]
print(json.dumps(comparison, indent=2))
if comparison["grpo_pause_threshold_exceeded"]:
    raise SystemExit("PAUSE: more than 1% of token ratios are outside [0.8, 1.2]")
PY
