from __future__ import annotations

import importlib.util
import json
import os
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
    "vela_g7_cause_isolation_v2_for_selector_v3",
    BASE / "g7-cause-isolation-v2" / "rwkv7_0p4b_g7_cause_isolation_v2.py",
)
chain = g7.chain
v2 = g7.v2
_orig_select = chain.target_free_select
_orig_write_report = g7.write_report


def conservative_target_free_select(model, ns, tok, ids, starts, ends, lineage, specs, segments):
    """Target-free selector v3 diagnostic.

    Keep chain-v2 candidate detection and immediate functional-equivalence logic,
    but when hop-2 has multiple carried W2-origin anchors that are immediately
    functionally equivalent to the chain-v2 baseline result, prefer the EARLIEST
    such W2-origin anchor instead of the latest one.

    This is deliberately conservative and threshold-free. It uses no W3-native
    or future trajectory outcome for selection. Its purpose is to test whether
    the G7 migration-only failures are caused by over-aggressive late-anchor
    pruning while preserving the carried-generation provenance requirement.
    """
    out = _orig_select(model, ns, tok, ids, starts, ends, lineage, specs, segments)
    trace = out.get("provenance_tiebreak") or {}
    candidates = trace.get("w2_candidates") or []
    eligible = [
        int(row["segment"])
        for row in candidates
        if row.get("origin") == "W2"
        and row.get("same_functional_signature_as_v1_result") is True
    ]
    if not eligible:
        out["selector_v3"] = {
            "used": False,
            "reason": "no_equivalent_w2_candidate",
            "chain_v2_chosen_seg": out.get("chosen_seg"),
        }
        return out

    chain_v2_chosen = int(out["chosen_seg"])
    chosen = min(eligible)
    pos = starts[chosen]
    state = chain.state_to_end(model, ns, ids, pos, lineage[pos])
    rows = chain.ca.score_specs(model, ns, tok, state, specs)

    out["chosen_seg"] = chosen
    out["anchor_pos"] = pos
    out["state"] = state
    out["rows"] = rows
    out["selected_segments"] = sorted(set(out.get("selected_segments", [])) | {chosen})
    out["selector_v3"] = {
        "used": chosen != chain_v2_chosen,
        "chain_v2_chosen_seg": chain_v2_chosen,
        "final_chosen_seg": chosen,
        "eligible_equivalent_w2_segments": sorted(eligible),
        "policy": "earliest carried W2-origin anchor among immediate-functional-equivalent candidates",
        "oracle_usage": "none",
    }
    return out


def write_report(report):
    report["status"] = "VELA_G7_RWKV7_0P4B_SELECTOR_V3_CONSERVATIVE"
    report["selector_v3_policy"] = {
        "name": "earliest-carried-W2 functional-equivalence guard",
        "basis": "G7 anchor-depth v3 showed all four native-stable migration failures are rescuable by a carried W2-origin anchor",
        "oracle_usage": "none for selection; W3-native remains evaluation-only in the inherited G7 protocol",
        "intent": "test whether conservative replay depth eliminates migration-only excess failures without changing the backbone or adaptation recipe",
        "freeze_status": "diagnostic only; do not freeze this policy even if it passes",
    }
    summary = report.get("summary")
    if isinstance(summary, dict):
        summary["selector_v3_migration_subgate_pass"] = (
            summary.get("initial_chain_qualified") == report.get("summary", {}).get("fixture_count")
            and summary.get("migration_only_excess_failure_count") == 0
        )
    report["claim_boundary"] = (
        "Synthetic target-free selector-v3 diagnostic on the same RWKV-7 0.4B G7 cause-isolation suite. "
        "The policy uses no future/native oracle for anchor choice. Passing the migration-only denominator would "
        "show that conservative bounded replay can fix the observed migration drift; it would not resolve native "
        "0.4B instability or justify final selector/backbone freeze."
    )
    _orig_write_report(report)


chain.target_free_select = conservative_target_free_select
g7.write_report = write_report

if __name__ == "__main__":
    g7.run()
