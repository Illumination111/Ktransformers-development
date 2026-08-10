#!/usr/bin/env python3
"""Unit checks for M2 KTCompositeLoRAManager helpers without loading full sglang/CUDA."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock

import torch


def _load_manager_with_stubs():
    kt_ep = types.ModuleType("sglang.srt.layers.moe.kt_ep_wrapper")

    @dataclass
    class KTExpertLoraWeights:
        rank: int = 8
        alpha: float = 16.0
        gate_lora_a: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 8, 4))
        gate_lora_b: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 4, 8))
        up_lora_a: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 8, 4))
        up_lora_b: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 4, 8))
        down_lora_a: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 8, 4))
        down_lora_b: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 4, 8))

    class KTEPWrapperMethod:
        def __init__(self, layer_idx: int = 0):
            self.kt_config = types.SimpleNamespace(layer_idx=layer_idx)
            self.kt_expert_lora_enabled = True
            self.kt_expert_lora_weights = KTExpertLoraWeights()
            self.kt_lora_slot_weights: Dict[int, object] = {}
            self.kt_lora_dispatch = "single"
            self.kt_lora_token_slots_for_batch = None

        def register_kt_lora_slot(self, slot, weights):
            self.kt_lora_slot_weights[slot] = weights

        def activate_kt_lora_slot(self, slot):
            pass

        def unload_kt_lora_slot(self, slot):
            self.kt_lora_slot_weights.pop(slot, None)

    kt_ep.KTEPWrapperMethod = KTEPWrapperMethod
    kt_ep.KTExpertLoraWeights = KTExpertLoraWeights
    kt_ep._make_zero_kt_expert_lora_weights = lambda **kwargs: KTExpertLoraWeights()
    kt_ep._load_kt_expert_lora_weights = lambda **kwargs: KTExpertLoraWeights()

    registry = types.ModuleType("sglang.srt.lora.lora_registry")

    @dataclass
    class LoRARef:
        lora_id: str
        lora_name: str
        lora_path: str
        pinned: bool = False
        source_lora_path: Optional[str] = None
        kt_expert_lora_path: Optional[str] = None
        adapter_kind: str = "ordinary"

    registry.LoRARef = LoRARef

    for name, mod in [
        ("sglang", types.ModuleType("sglang")),
        ("sglang.srt", types.ModuleType("sglang.srt")),
        ("sglang.srt.layers", types.ModuleType("sglang.srt.layers")),
        ("sglang.srt.layers.moe", types.ModuleType("sglang.srt.layers.moe")),
        ("sglang.srt.layers.moe.kt_ep_wrapper", kt_ep),
        ("sglang.srt.lora", types.ModuleType("sglang.srt.lora")),
        ("sglang.srt.lora.lora_registry", registry),
    ]:
        sys.modules[name] = mod

    path = Path(
        "/mnt/data2/wbw/ktransformers/third_party/sglang/python/sglang/srt/lora/"
        "kt_composite_lora_manager.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sglang.srt.lora.kt_composite_lora_manager", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sglang.srt.lora.kt_composite_lora_manager"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod, LoRARef


def main() -> int:
    mgr_mod, LoRARef = _load_manager_with_stubs()
    KTCompositeLoRAManager = mgr_mod.KTCompositeLoRAManager
    KTExpertLoRAPool = mgr_mod.KTExpertLoRAPool
    BASE_KT_LORA_SLOT = mgr_mod.BASE_KT_LORA_SLOT

    refs = [
        LoRARef(
            lora_id=f"id-{n}",
            lora_name=n,
            lora_path=f"/tmp/{n}",
            kt_expert_lora_path=f"/tmp/{n}",
            adapter_kind="kt_composite",
        )
        for n in ("A", "B", "C")
    ]

    server_args = MagicMock()
    server_args.lora_paths = refs
    server_args.kt_max_loaded_loras = 4
    server_args.kt_max_loras_per_batch = 4
    server_args.kt_lora_dispatch = "grouped"

    runner = MagicMock()
    runner.server_args = server_args
    runner.tp_rank = 0

    mgr = KTCompositeLoRAManager.__new__(KTCompositeLoRAManager)
    mgr.model_runner = runner
    mgr.server_args = server_args
    mgr.tp_rank = 0
    mgr.composite_refs = {r.lora_id: r for r in refs}
    mgr.pool = KTExpertLoRAPool(4)
    mgr.layers = []
    mgr._active_slot = BASE_KT_LORA_SLOT
    mgr._initialized = True
    mgr.dispatch_mode = "grouped"
    mgr.max_loras_per_batch = 4
    mgr.current_kt_lora_token_slots = None
    for r in refs:
        mgr.pool.mark_ready(mgr.pool.alloc_slot(r))

    assert mgr.validate_batch({refs[0].lora_id, refs[1].lora_id, refs[2].lora_id})
    assert not mgr.validate_batch(
        {refs[0].lora_id, refs[1].lora_id, refs[2].lora_id, "extra"}
    )

    @dataclass
    class Mode:
        def is_decode(self):
            return True

        def is_target_verify(self):
            return False

    fb = MagicMock()
    fb.lora_ids = [r.lora_id for r in refs]
    fb.forward_mode = Mode()
    fb.input_ids = torch.zeros(3, dtype=torch.long)
    fb.extend_seq_lens_cpu = None
    slots = mgr.build_token_slots(fb)
    assert int(torch.unique(slots).numel()) == 3

    fb2 = MagicMock()
    fb2.lora_ids = [refs[0].lora_id, refs[1].lora_id]
    fb2.forward_mode = types.SimpleNamespace(
        is_decode=lambda: False, is_target_verify=lambda: False
    )
    fb2.input_ids = torch.zeros(5, dtype=torch.long)
    fb2.extend_seq_lens_cpu = [2, 3]
    fb2.extend_seq_lens = None
    slots2 = mgr.build_token_slots(fb2)
    assert slots2.tolist() == [
        mgr.pool.get_slot(refs[0].lora_id),
        mgr.pool.get_slot(refs[0].lora_id),
        mgr.pool.get_slot(refs[1].lora_id),
        mgr.pool.get_slot(refs[1].lora_id),
        mgr.pool.get_slot(refs[1].lora_id),
    ]

    mgr.dispatch_mode = "single"
    mgr.max_loras_per_batch = 1
    assert mgr.validate_batch({refs[0].lora_id})
    assert not mgr.validate_batch({refs[0].lora_id, refs[1].lora_id})

    print("PASS: M2 manager validate_batch + build_token_slots (N=2/3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
