#!/usr/bin/env python3
"""Teacher-force the fixed rollout trajectory through native veRL FSDP2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from probability_utils import compare_logprobs, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from tensordict import TensorDict

    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.workers.utils.padding import no_padding_2_padding

    rollout = json.loads(args.rollout.read_text(encoding="utf-8"))
    prompt_ids = rollout["prompt_ids"]
    response_ids = rollout["response_ids"]
    all_ids = prompt_ids + response_ids
    model_config = HFModelConfig(
        path=rollout["model_path"],
        load_tokenizer=False,
        trust_remote_code=True,
        override_config={"attn_implementation": "flash_attention_2"},
        enable_gradient_checkpointing=False,
        use_remove_padding=True,
        lora_rank=0,
        use_fused_kernels=False,
    )
    engine_config = FSDPEngineConfig(
        strategy="fsdp2",
        forward_only=False,
        fsdp_size=world_size,
        model_dtype="bfloat16",
        dtype="bfloat16",
        mixed_precision={"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
        param_offload=False,
        optimizer_offload=False,
        offload_policy=False,
        reshard_after_forward=True,
        use_orig_params=False,
        use_dynamic_bsz=False,
        infer_micro_batch_size_per_gpu=1,
        infer_max_token_len_per_gpu=len(all_ids),
        use_remove_padding=True,
        use_fused_kernels=False,
        use_torch_compile=False,
        full_determinism=False,
    )
    engine = FSDPEngineWithLMHead(
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=FSDPOptimizerConfig(total_training_steps=1),
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    engine.module.eval()

    def nested(values: list[int]) -> torch.Tensor:
        return torch.nested.as_nested_tensor([torch.tensor(values, dtype=torch.long)], layout=torch.jagged)

    data = TensorDict(
        {
            "input_ids": nested(all_ids),
            "position_ids": nested(list(range(len(all_ids)))),
            "prompts": nested(prompt_ids),
            "responses": nested(response_ids),
            "response_mask": nested([1] * len(response_ids)),
            "loss_mask": nested([1] * len(response_ids)),
            "temperature": torch.ones(1, dtype=torch.float32),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        data,
        global_token_num=[len(all_ids)],
        compute_loss=False,
        use_remove_padding=True,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        use_fused_kernels=False,
        max_seq_len=len(all_ids),
        max_response_len=len(response_ids),
        max_response_length=len(response_ids),
    )
    with engine.eval_mode():
        output = engine.infer_batch(data, loss_function=None)
    padded = no_padding_2_padding(output["model_output"]["log_probs"].cpu(), data)
    score_logprobs = [float(value) for value in padded[0, : len(response_ids)].tolist()]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, score_logprobs)
    if rank == 0:
        result = {
            "backend": "native_verl_fsdp2_actor",
            "rollout_path": str(args.rollout.resolve()),
            "response_ids": response_ids,
            "score_logprobs": score_logprobs,
            "comparison": compare_logprobs(rollout["rollout_logprobs"], score_logprobs),
            "rank_repeatability": [compare_logprobs(gathered[0], values) for values in gathered[1:]],
            "config": {
                "world_size": world_size,
                "dtype": "bfloat16",
                "strategy": "fsdp2",
                "use_remove_padding": True,
                "attn_implementation": "flash_attention_2",
                "backward": False,
                "optimizer_step": False,
            },
        }
        write_json(args.output, result)
        print(json.dumps(result["comparison"], indent=2))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
