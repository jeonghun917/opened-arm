from pathlib import Path
import sys, time, subprocess
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.budget_guard import assert_can_start, record

T4_USD_PER_HOUR = 0.19
MAX_RUNTIME_HOURS = 4.0
PROJECTED_COST = T4_USD_PER_HOUR * MAX_RUNTIME_HOURS

assert_can_start("lightning", PROJECTED_COST)
t0 = time.time()
try:
    subprocess.run([sys.executable, str(ROOT/"lightning"/"run_in_studio.py")], check=True)
finally:
    elapsed_h = (time.time() - t0) / 3600.0
    record("lightning", elapsed_h * T4_USD_PER_HOUR, "guarded_run elapsed-time estimate")
