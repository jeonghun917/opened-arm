from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_shell(binary: str, lines: list[str]) -> str:
    payload = "\n".join(lines + ["quit"]) + "\n"
    cp = subprocess.run(
        [binary, "shell"],
        input=payload,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    return (cp.stdout or "") + (cp.stderr or "")


def block(output: str, name: str) -> str:
    start = f"//*{name}"
    lines = output.splitlines()
    inside = False
    kept: list[str] = []
    for line in lines:
        if line.strip() == start:
            inside = True
            continue
        if inside and line.strip() == "//*done":
            break
        if inside:
            kept.append(line)
    return "\n".join(kept)


def contains_term(text: str, atom: str) -> bool:
    return atom in text


def run() -> dict:
    if len(sys.argv) != 2:
        raise SystemExit("usage: ona_state_scope.py /path/to/NAR")
    binary = str(Path(sys.argv[1]).resolve())

    durable = "<{vela} --> durable>"
    active = "<{vela} --> active>"
    target = "<{vela} --> target>"

    full_prelude = [
        "*volume=0",
        durable + ".",
        active + ". :|:",
        target + "! :|:",
    ]
    inspect = ["*concepts", "*cycling_belief_events", "*cycling_goal_events"]

    native = run_shell(binary, full_prelude + inspect)
    # persistentNAR-style control: restore durable declarative content only.
    durable_reload = run_shell(binary, ["*volume=0", durable + "."] + inspect)
    # Matched transcript replay: fresh process, but reconstruct every pre-cut input.
    transcript_replay = run_shell(binary, full_prelude + inspect)

    native_beliefs = block(native, "cycling_belief_events")
    reload_beliefs = block(durable_reload, "cycling_belief_events")
    replay_beliefs = block(transcript_replay, "cycling_belief_events")

    native_goals = block(native, "cycling_goal_events")
    reload_goals = block(durable_reload, "cycling_goal_events")
    replay_goals = block(transcript_replay, "cycling_goal_events")

    reload_concepts = block(durable_reload, "concepts")

    report = {
        "candidate": "ONA/OpenNARS for Applications",
        "test": "ordinary durable-memory reconstruction vs active-state scope",
        "native_active_belief_present": contains_term(native_beliefs, "active"),
        "durable_reload_active_belief_present": contains_term(reload_beliefs, "active"),
        "transcript_replay_active_belief_present": contains_term(replay_beliefs, "active"),
        "native_active_goal_present": contains_term(native_goals, "target"),
        "durable_reload_active_goal_present": contains_term(reload_goals, "target"),
        "transcript_replay_active_goal_present": contains_term(replay_goals, "target"),
        "durable_reload_retains_durable_concept": contains_term(reload_concepts, "durable"),
        "expected_scope_pattern": False,
        "interpretation": (
            "The ordinary persistent-memory style can restore durable declarative content, "
            "but does not restore ONA's live cycling belief/goal queues. Full transcript replay "
            "can rebuild those observable states, but that is re-execution rather than a native checkpoint."
        ),
        "claim_boundary": "State-scope evidence only; not a VELA continuity or identity proof.",
    }
    report["expected_scope_pattern"] = bool(
        report["native_active_belief_present"]
        and not report["durable_reload_active_belief_present"]
        and report["transcript_replay_active_belief_present"]
        and report["native_active_goal_present"]
        and not report["durable_reload_active_goal_present"]
        and report["transcript_replay_active_goal_present"]
        and report["durable_reload_retains_durable_concept"]
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["expected_scope_pattern"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    run()
