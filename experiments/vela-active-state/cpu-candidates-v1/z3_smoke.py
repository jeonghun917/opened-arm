from __future__ import annotations
import json
from z3 import Int, Solver, Distinct, sat, unsat


def run():
    # Tiny planning-like constraint problem: three tasks, three distinct slots,
    # with A before B. This checks the solver as a deterministic specialist organ.
    a, b, c = Int("a"), Int("b"), Int("c")
    s = Solver()
    for x in (a, b, c):
        s.add(x >= 0, x <= 2)
    s.add(Distinct(a, b, c), a < b)
    base = s.check()
    model = s.model() if base == sat else None

    # Add an explicit contradiction and verify it is rejected.
    s.push()
    s.add(a == b)
    contradiction = s.check()
    s.pop()

    report = {
        "candidate": "Z3",
        "role": "constraint_verification_specialist",
        "base_sat": base == sat,
        "contradiction_unsat": contradiction == unsat,
        "example_assignment": None if model is None else {
            "a": model[a].as_long(),
            "b": model[b].as_long(),
            "c": model[c].as_long(),
        },
        "claim_boundary": "Runtime feasibility only; Z3 is not treated as the VELA subject.",
    }
    print(json.dumps(report, indent=2))
    if not (report["base_sat"] and report["contradiction_unsat"]):
        raise SystemExit(1)


if __name__ == "__main__":
    run()
