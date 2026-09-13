# KT VLM Conv3D local validation

This suite holds the broad numerical and compatibility matrix for the
instance-scoped KT Conv3D fallback. The upstream KTransformers repository keeps
only a small forward/backward regression and an unsupported-contract check.
It also verifies the cross-repository installation contract: LlamaFactory keeps
the single ordinary `ktransformers[sft]` requirement, KTransformers owns patch
installation, and LlamaFactory only validates the per-instance readiness marker.

Run it with:

```bash
/mnt/data2/wbw/conda/envs/Kllama/bin/python -m pytest -q \
  /mnt/data2/wbw/Ktransformers-development/VLM-FT-test/conv3d_compat
```

Override the source under test when needed:

```bash
VLM_KT_CONV3D_COMPAT=/path/to/kt-kernel/python/sft/conv3d_compat.py \
VLM_KTRANSFORMERS_ROOT=/path/to/ktransformers \
VLM_LLAMAFACTORY_ROOT=/path/to/LlamaFactory \
  python -m pytest -q VLM-FT-test/conv3d_compat
```
