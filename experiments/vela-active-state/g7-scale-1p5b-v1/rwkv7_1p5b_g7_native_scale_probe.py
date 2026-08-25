from __future__ import annotations

import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g7 = load_module(
    "vela_g7_cause_isolation_v2_for_1p5b_scale",
    BASE / "g7-cause-isolation-v2" / "rwkv7_0p4b_g7_cause_isolation_v2.py",
)
chain = g7.chain
_orig_write_report = g7.write_report

# Focused scale probe: same W1->W2->W3 adaptation recipe and same two neutral
# future streams, but only the fixture that was W3-native unstable on 0.4B.
# This avoids spending a full 1.5B G7 run before the scale hypothesis is tested.
chain.scale.WEIGHT_REPO = "BlinkDL/rwkv7-g1"
chain.scale.WEIGHT_REVISION = "ede85bf8ab2e59aff7d7ca909fbbc73317866d89"
chain.scale.WEIGHT_FILE = "rwkv7-g1i-1.5b-20260805-ctx16384.pth"
chain.scale.N_LAYER = 24
chain.scale.N_EMBD = 2048
chain.scale.HEAD_SIZE = 64
chain.scale.VOCAB_SIZE = 65536

g7.CORE_FIXTURE_IDS = ("superseded_old_value_echoes",)
g7.NATIVE_STABLE_MIN_CASES = 2


def write_report(report):
    report["status"] = "VELA_G7_RWKV7_1P5B_NATIVE_SCALE_PROBE_V1"
    report["scale_probe"] = {
        "purpose": "test whether the 0.4B W3-native old_value_echoes instability is materially reduced by scaling to the released RWKV-7 1.5B checkpoint",
        "reference_0p4b": {
            "fixture": "superseded_old_value_echoes",
            "native_stable_streams": "0/2",
            "g7_v2_total_native_stable": "6/8",
        },
        "model": "RWKV-7 g1i 1.5B, 24 layers, width 2048, ctx16384",
        "protocol": "same W2/W3 adaptation recipe; same telemetry/inventory future streams; horizons 128/512/2048",
        "primary_readout": "native_stable_case_count out of 2",
        "decision_rule": {
            "2/2": "strong evidence that 0.4B capacity/robustness is an important native-stability bottleneck; then run full 1.5B G7 qualification",
            "1/2": "partial scale benefit; consider full 1.5B + 2.9B direction only after selector-v3 result",
            "0/2": "no focused native-stability benefit; do not scale blindly",
        },
        "paid_gpu": False,
    }
    report["claim_boundary"] = (
        "Focused CPU scale diagnostic on one preregistered native-unstable fixture and two neutral future streams. "
        "It is not a full 1.5B G7 qualification and does not freeze the backbone."
    )
    _orig_write_report(report)


g7.write_report = write_report

if __name__ == "__main__":
    g7.run()
