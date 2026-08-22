from __future__ import annotations
from pathlib import Path
import json, sys, time

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "microbudget.json").read_text(encoding="utf-8"))
LEDGER = ROOT / "runs" / "paid_budget_ledger.json"
PAID = ("lightning", "modal")

def load():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"lightning": 0.0, "modal": 0.0, "events": []}

def total_paid(ledger):
    return sum(float(ledger.get(p, 0.0)) for p in PAID)

def assert_can_start(provider: str, projected_cost_usd: float):
    if provider not in PAID:
        return
    ledger = load()
    projected = max(0.0, float(projected_cost_usd))
    provider_after = float(ledger.get(provider, 0.0)) + projected
    total_after = total_paid(ledger) + projected
    provider_ceiling = float(CFG["absolute_provider_hard_ceiling_usd"])
    total_ceiling = float(CFG["paid_total_hard_ceiling_usd"])
    if provider_after > provider_ceiling + 1e-9:
        raise SystemExit(
            f"BUDGET_BLOCK_PROVIDER: {provider} projected ${provider_after:.4f} > ${provider_ceiling:.2f}"
        )
    if total_after > total_ceiling + 1e-9:
        raise SystemExit(
            f"BUDGET_BLOCK_TOTAL: projected paid total ${total_after:.4f} > ${total_ceiling:.2f}"
        )

def reserve_state():
    ledger = load()
    base_total = sum(float(CFG["base_caps"][p]) for p in PAID)
    paid = total_paid(ledger)
    extra_used = max(0.0, paid - base_total)
    return {
        "paid_total": paid,
        "base_total": base_total,
        "extra_used": extra_used,
        "extra_remaining": max(0.0, float(CFG["shared_extra_reserve_usd"]) - extra_used),
        "hard_total": float(CFG["paid_total_hard_ceiling_usd"]),
    }

def record(provider: str, actual_or_estimated_cost_usd: float, note: str):
    amount = max(0.0, float(actual_or_estimated_cost_usd))
    assert_can_start(provider, amount)
    ledger = load()
    ledger[provider] = float(ledger.get(provider, 0.0)) + amount
    ledger["events"].append({
        "ts": time.time(),
        "provider": provider,
        "amount_usd": amount,
        "note": note,
    })
    LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(json.dumps(reserve_state(), indent=2))
        raise SystemExit(0)
    if len(sys.argv) != 3:
        raise SystemExit("usage: budget_guard.py [<provider> <projected_cost_usd>]")
    assert_can_start(sys.argv[1], float(sys.argv[2]))
    print(json.dumps(reserve_state(), indent=2))
