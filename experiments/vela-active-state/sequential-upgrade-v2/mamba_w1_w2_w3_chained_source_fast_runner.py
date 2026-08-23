from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE / "mamba_w1_w2_w3_chained_source.py"
spec = importlib.util.spec_from_file_location("vela_g4_chained_source_v2", MAIN)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {MAIN}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Equivalent checkpoint material, but build all requested prefix checkpoints in one
# chronological pass instead of recomputing every prefix from token zero.
def _sequential_full_lineage(model, ids, positions):
    out = {}
    cur = None
    last = 0
    for p in sorted(set(positions)):
        if p < last:
            raise ValueError("positions must be monotonic")
        if p > last:
            cur = mod.dep.run_slice(model, ids, last, p, cur)
        out[p] = mod.clone_cache(cur)
        last = p
    return out

mod.full_lineage = _sequential_full_lineage

if __name__ == "__main__":
    mod.run()
