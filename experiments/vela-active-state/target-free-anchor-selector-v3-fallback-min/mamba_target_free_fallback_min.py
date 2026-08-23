from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE.parent / "target-free-anchor-selector-v3-fallback" / "mamba_target_free_fallback.py"
spec = importlib.util.spec_from_file_location("vela_target_free_fallback_full", FULL)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {FULL}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Gate G5 requires at least three nontrivial injected-miss cases. Reuse the exact
# full fallback implementation, but limit the CPU probe to the first three stress
# fixtures (single_early, single_middle, single_late) to avoid re-running all 15.
mod.stress.FIXTURES = mod.stress.FIXTURES[:3]

if __name__ == "__main__":
    mod.run()
