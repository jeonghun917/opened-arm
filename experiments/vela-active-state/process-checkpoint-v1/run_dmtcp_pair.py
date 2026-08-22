from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def wait_for(path: Path, timeout: float, *, nonempty: bool = False) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and (not nonempty or path.stat().st_size > 0):
            return
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {path}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--result", required=True)
    ap.add_argument("target", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    target = list(args.target)
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        raise SystemExit("target command required after --")

    work = Path(args.workdir).resolve()
    ckpt = work / "ckpt"
    ready = work / "ready"
    branch = work / "branch"
    native_result = work / "native.json"
    restored_result = work / "restored.json"
    result_path = Path(args.result).resolve()

    if work.exists():
        import shutil
        shutil.rmtree(work)
    ckpt.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["DMTCP_COORD_HOST"] = "127.0.0.1"
    env["DMTCP_COORD_PORT"] = str(args.port)
    env["DMTCP_CHECKPOINT_DIR"] = str(ckpt)
    env["DMTCP_GZIP"] = "0"

    launch_log = (work / "launch.log").open("w", encoding="utf-8")
    native_rc = restored_rc = None
    checkpoint_ok = restart_script_ok = False
    checkpoint_command = ""
    try:
        proc = subprocess.Popen(
            ["dmtcp_launch"] + target,
            cwd=str(work),
            env=env,
            stdout=launch_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_for(ready, 90)

        ck = subprocess.run(
            ["dmtcp_command", "--bcheckpoint"],
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        checkpoint_command = (ck.stdout or "") + (ck.stderr or "")
        checkpoint_ok = ck.returncode == 0
        restart_script = ckpt / "dmtcp_restart_script.sh"
        wait_for(restart_script, 30, nonempty=True)
        restart_script_ok = True

        branch.write_text("native\n", encoding="utf-8")
        wait_for(native_result, 60, nonempty=True)
        native_rc = proc.wait(timeout=30)

        branch.unlink(missing_ok=True)
        restart_log = (work / "restart.log").open("w", encoding="utf-8")
        try:
            restarted = subprocess.Popen(
                ["bash", str(restart_script)],
                cwd=str(ckpt),
                env=env,
                stdout=restart_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            # The restored target resumes at the checkpointed wait loop. The label file is
            # deliberately external to the checkpoint and changes only result routing.
            branch.write_text("restored\n", encoding="utf-8")
            wait_for(restored_result, 90, nonempty=True)
            try:
                restored_rc = restarted.wait(timeout=30)
            except subprocess.TimeoutExpired:
                restarted.terminate()
                restored_rc = restarted.wait(timeout=10)
        finally:
            restart_log.close()
    finally:
        launch_log.close()
        try:
            subprocess.run(
                ["dmtcp_command", "--quit"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except Exception:
            pass

    native = load_json(native_result) if native_result.exists() else {}
    restored = load_json(restored_result) if restored_result.exists() else {}
    equivalent = bool(
        native
        and restored
        and native.get("state_signature") == restored.get("state_signature")
        and native.get("expected_live_state") is True
        and restored.get("expected_live_state") is True
    )
    report = {
        "checkpoint_command_ok": checkpoint_ok,
        "restart_script_created": restart_script_ok,
        "native_rc": native_rc,
        "restored_rc": restored_rc,
        "native": native,
        "restored": restored,
        "state_signature_equivalent": equivalent,
        "checkpoint_command_tail": checkpoint_command[-500:],
        "claim_boundary": (
            "External whole-process checkpoint reference only. Equivalence demonstrates that "
            "preserving the full causally active runtime state is sufficient for this fixture; "
            "it does not prove VELA identity or make DMTCP the target architecture."
        ),
    }
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not (checkpoint_ok and restart_script_ok and equivalent):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
