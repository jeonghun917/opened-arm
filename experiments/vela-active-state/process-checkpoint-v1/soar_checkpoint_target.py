from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import soar_sml as sml


def wait_branch(path: Path) -> str:
    while True:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value in {"native", "restored"}:
                return value
        time.sleep(0.05)


def norm(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: soar_checkpoint_target.py workdir")
    work = Path(sys.argv[1]).resolve()
    ready = work / "ready"
    branch_file = work / "branch"
    prod_file = work / "checkpoint-rule.soar"
    prod_file.write_text(
        "sp {vela*dmtcp-scope\n"
        "   (state <s> ^superstate nil)\n"
        "-->\n"
        "   (<s> ^vela-rule-loaded yes)\n"
        "}\n",
        encoding="utf-8",
    )

    kernel = sml.Kernel.CreateKernelInNewThread()
    if not kernel:
        raise RuntimeError("Soar kernel creation failed")
    agent = kernel.CreateAgent("vela")
    if not agent:
        kernel.Shutdown()
        raise RuntimeError("Soar agent creation failed")

    try:
        if not agent.LoadProductions(str(prod_file)):
            raise RuntimeError("fixture production failed to load")
        agent.GetInputLink().CreateStringWME("vela-marker", "active")
        agent.RunSelf(1)

        # Stable cut: the kernel thread, agent working memory, input-link value and
        # production state are all live inside this process.
        ready.write_text("ready\n", encoding="utf-8")
        branch = wait_branch(branch_file)

        marker = agent.GetInputLink().GetParameterValue("vela-marker")
        marker = None if marker is None else str(marker)
        production = agent.ExecuteCommandLine("print vela*dmtcp-scope")
        state = agent.ExecuteCommandLine("print --depth 4 s1")
        expected_live_state = bool(
            marker == "active"
            and "vela*dmtcp-scope" in production
        )
        report = {
            "candidate": "Soar 9.6.5",
            "branch": branch,
            "expected_live_state": expected_live_state,
            "state_signature": {
                "input_marker": marker,
                "production": norm(production),
                "state": norm(state),
            },
            "claim_boundary": "Whole-process checkpoint fixture; not ordinary save-agent behavior and not VELA identity evidence.",
        }
        (work / f"{branch}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not expected_live_state:
            raise SystemExit(1)
    finally:
        try:
            kernel.Shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
