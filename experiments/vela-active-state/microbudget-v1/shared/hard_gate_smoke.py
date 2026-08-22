from __future__ import annotations
import json

def evaluate(state):
    fails = []
    if len(state.get("canonical_writers", [])) != 1:
        fails.append("duplicate_canonical_writers")
    if state.get("external_action_authority") != "CONTROL_ONLY":
        fails.append("external_action_authority_bypass")
    if state.get("canonical_goal_conflicts", 0) > 0:
        fails.append("canonical_goal_conflicts")
    if state.get("cross_workstream_contamination", 0) > 0:
        fails.append("cross_workstream_contamination")
    if state.get("untraceable_state_mutations", 0) > 0:
        fails.append("untraceable_state_mutations")
    if state.get("hypothesis_conflict_count", 0) > 0 and not state.get("conflicts_visible", False):
        fails.append("hidden_hypothesis_conflicts")
    return {"status": "PASS" if not fails else "FAIL", "failed": fails}

if __name__ == "__main__":
    sample = {
        "canonical_writers": ["VELA_SHARED_STATE"],
        "external_action_authority": "CONTROL_ONLY",
        "canonical_goal_conflicts": 0,
        "cross_workstream_contamination": 0,
        "untraceable_state_mutations": 0,
        "hypothesis_conflict_count": 2,
        "conflicts_visible": True,
    }
    print(json.dumps(evaluate(sample), indent=2))
