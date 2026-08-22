from pathlib import Path
import subprocess, sys, json, time

def ensure_transformers():
    try:
        import transformers
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40,<5"])

ensure_transformers()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.rwkv_state_restore import run as run_rwkv
from shared.hard_gate_smoke import evaluate

t0 = time.time()
report = {
    "provider": "lightning",
    "rwkv": run_rwkv(),
    "hard_gate": evaluate({
        "canonical_writers": ["VELA_SHARED_STATE"],
        "external_action_authority": "CONTROL_ONLY",
        "canonical_goal_conflicts": 0,
        "cross_workstream_contamination": 0,
        "untraceable_state_mutations": 0,
        "hypothesis_conflict_count": 2,
        "conflicts_visible": True,
    }),
}
report["elapsed_s"] = time.time() - t0
Path("vela_lightning_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
