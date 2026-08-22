import subprocess, sys, json, os
from pathlib import Path

def ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40,<5"])

ensure("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.rwkv_state_restore import run as run_rwkv
from shared.hard_gate_smoke import evaluate

report = {
    "provider": "kaggle",
    "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
Path("/kaggle/working/vela_result.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
