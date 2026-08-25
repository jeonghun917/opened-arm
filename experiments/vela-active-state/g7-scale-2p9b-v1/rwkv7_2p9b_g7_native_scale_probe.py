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
    "vela_g7_cause_isolation_v2_for_2p9b_scale",
    BASE / "g7-cause-isolation-v2" / "rwkv7_0p4b_g7_cause_isolation_v2.py",
)
chain = g7.chain
_orig_write_report = g7.write_report

# Focused scale probe: same W1->W2->W3 adaptation recipe and same two neutral
# future streams, but only the fixture that was W3-native unstable on 0.4B and
# remained unstable in the 1.5B focused probe.
chain.scale.WEIGHT_REPO = "BlinkDL/rwkv7-g1"
chain.scale.WEIGHT_REVISION = "ede85bf8ab2e59aff7d7ca909fbbc73317866d89"
chain.scale.WEIGHT_FILE = "rwkv7-g1i-2.9b-20260805-ctx16384.pth"
chain.scale.N_LAYER = 32
chain.scale.N_EMBD = 2560
chain.scale.HEAD_SIZE = 64
chain.scale.VOCAB_SIZE = 65536

g7.CORE_FIXTURE_IDS = ("superseded_old_value_echoes",)
g7.NATIVE_STABLE_MIN_CASES = 2


def write_report(report):
    report["status"] = "VELA_G7_RWKV7_2P9B_NATIVE_SCALE_PROBE_V1"
    report["scale_probe"] = {
        "purpose": "test whether the old_value_echoes native instability seen at 0.4B and 1.5B is materially reduced by the released RWKV-7 2.9B checkpoint",
        "reference_results": {
            "0p4b_native_stable_streams": "0/2",
            "1p5b_native_stable_streams": "0/2",
            "1p5b_initial_chain_qualified": "0/1 fixture",
        },
        "model": "RWKV-7 g1i 2.9B, 32 layers, width 2560, ctx16384",
        "protocol": "same W2/W3 adaptation recipe; same telemetry/inventory future streams; horizons 128/512/2048",
        "primary_readout": "native_stable_case_count out of 2, with immediate qualification reported separately",
        "decision_rule": {
            "2/2": "strong scale evidence; then consider a full 2.9B G7 qualification without freezing the backbone",
            "1/2": "partial scale benefit; compare failure horizon/margins before any larger-scale move",
            "0/2": "no focused native-stability benefit on this route; do not infer that larger RWKV is impossible, but stop blind scaling under this recipe",
        },
        "paid_gpu": False,
    }
    report["claim_boundary"] = (
        "Focused CPU scale diagnostic on one preregistered native-unstable fixture and two neutral future streams. "
        "A runner OOM/timeout/download failure is infrastructure failure, not a scientific failure. "
        "This is not a full 2.9B G7 qualification and does not freeze the backbone."
    )
    _orig_write_report(report)


g7.write_report = write_report

if __name__ == "__main__":
    g7.run()
