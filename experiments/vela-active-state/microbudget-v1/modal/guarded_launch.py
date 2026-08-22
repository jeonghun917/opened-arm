from pathlib import Path
import sys, subprocess
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.budget_guard import assert_can_start

T4_USD_PER_HOUR_REFERENCE = 0.5904
PROJECTED_COST = T4_USD_PER_HOUR_REFERENCE * (1200 / 3600)
assert_can_start("modal", PROJECTED_COST)
subprocess.run(["modal", "run", str(ROOT/"modal"/"modal_rwkv_smoke.py")], check=True)
print("Record provider-reported billed cost before the next paid batch when available.")
