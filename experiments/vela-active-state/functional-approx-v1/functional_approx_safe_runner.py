from __future__ import annotations

import importlib.util
from pathlib import Path
import torch

TARGET = Path(__file__).with_name("mamba_functional_approx.py")
spec = importlib.util.spec_from_file_location("vela_functional_approx_v1", TARGET)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {TARGET}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_original_run_tokens = mod.v3.run_tokens_with_cache

def _no_grad_run_tokens(model, ids, cache, start_pos):
    with torch.no_grad():
        return _original_run_tokens(model, ids, cache, start_pos)

mod.v3.run_tokens_with_cache = _no_grad_run_tokens

if __name__ == "__main__":
    mod.run()
