from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def wait_branch(path: Path) -> str:
    while True:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value in {"native", "restored"}:
                return value
        time.sleep(0.05)


def block(output: str, name: str) -> list[str]:
    start = f"//*{name}"
    lines = output.splitlines()
    inside = False
    kept: list[str] = []
    for line in lines:
        text = line.strip()
        if text == start:
            inside = True
            continue
        if inside and text == "//*done":
            break
        if inside and text:
            kept.append(" ".join(text.split()))
    return kept


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: ona_checkpoint_target.py /path/to/NAR workdir")
    binary = str(Path(sys.argv[1]).resolve())
    work = Path(sys.argv[2]).resolve()
    ready = work / "ready"
    branch_file = work / "branch"

    proc = subprocess.Popen(
        [binary, "shell"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None

    prelude = [
        "*volume=0",
        "<{vela} --> durable>.",
        "<{vela} --> active>. :|:",
        "<{vela} --> target>! :|:",
    ]
    for line in prelude:
        proc.stdin.write(line + "\n")
    proc.stdin.flush()

    # Stable cut: Python controller is waiting and ONA is blocked on stdin with its live
    # belief/goal queues already populated. DMTCP snapshots both processes and the pipe state.
    ready.write_text("ready\n", encoding="utf-8")
    branch = wait_branch(branch_file)

    for line in ["*cycling_belief_events", "*cycling_goal_events", "quit"]:
        proc.stdin.write(line + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    output = proc.stdout.read()
    rc = proc.wait(timeout=20)

    beliefs = block(output, "cycling_belief_events")
    goals = block(output, "cycling_goal_events")
    state_signature = {
        "belief_queue": beliefs,
        "goal_queue": goals,
    }
    expected_live_state = bool(
        any("active" in line for line in beliefs)
        and any("target" in line for line in goals)
    )
    report = {
        "candidate": "ONA/OpenNARS for Applications",
        "branch": branch,
        "process_rc": rc,
        "expected_live_state": expected_live_state,
        "state_signature": state_signature,
        "claim_boundary": "Whole-process checkpoint fixture; not ordinary persistentNAR behavior and not VELA identity evidence.",
    }
    (work / f"{branch}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not expected_live_state or rc != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
