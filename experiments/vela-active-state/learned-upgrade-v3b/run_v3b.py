from __future__ import annotations

import copy
import importlib.util
import io
from pathlib import Path

import torch

ORIGINAL_DEEPCOPY = copy.deepcopy


def cache_safe_deepcopy(obj, memo=None):
    # PyTorch 2.13 rejects deepcopy() for non-leaf tensors inside MambaCache.
    # A torch serialization round-trip produces an independent cache with leaf tensors,
    # matching the checkpoint mechanism used by the earlier successful M0 probes.
    if obj is not None and (hasattr(obj, "conv_states") or hasattr(obj, "ssm_states")):
        buf = io.BytesIO()
        torch.save(obj, buf)
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=False)
    return ORIGINAL_DEEPCOPY(obj, memo)


copy.deepcopy = cache_safe_deepcopy

source = Path(__file__).resolve().parents[1] / "learned-upgrade-v3" / "mamba_upgrade_migration.py"
spec = importlib.util.spec_from_file_location("vela_learned_upgrade_v3", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.run()
