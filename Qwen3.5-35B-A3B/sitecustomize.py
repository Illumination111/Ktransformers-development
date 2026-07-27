"""Benchmark-local Python startup customization.

This module is imported automatically only when this directory is present on
PYTHONPATH.  It changes behavior solely when FFT_CUDA_CACHE_HOLD_MARKER is set.
"""

from __future__ import annotations

import os
from pathlib import Path


_MARKER_VALUE = os.environ.get("FFT_CUDA_CACHE_HOLD_MARKER")
if _MARKER_VALUE:
    import torch

    _marker = Path(_MARKER_VALUE)
    _original_empty_cache = torch.cuda.empty_cache

    def _profile_scoped_empty_cache() -> None:
        if _marker.exists():
            return
        _original_empty_cache()

    torch.cuda.empty_cache = _profile_scoped_empty_cache
