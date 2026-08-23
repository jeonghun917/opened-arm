from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MAIN = HERE / "mamba_w1_w2_w3_chain.py"
spec = importlib.util.spec_from_file_location("vela_sequential_w1_w2_w3_v1", MAIN)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {MAIN}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# After training, selected model weights still have requires_grad=True. The migration
# path is inference-only, but the shared dependency helper does not wrap its recurrent
# slice call in no_grad. Mamba updates cache tensors in-place, which PyTorch rejects
# when autograd is tracking them. Keep the experiment semantics unchanged and force
# every migration/replay slice onto the inference path.
_orig_run_slice = mod.dep.run_slice

def _safe_run_slice(model, ids, start, end, cache):
    with torch.no_grad():
        return _orig_run_slice(model, ids, start, end, cache)

mod.dep.run_slice = _safe_run_slice

if __name__ == "__main__":
    mod.run()
