from __future__ import annotations

import base64
import json
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path

from lightning_sdk import Job, Machine, Studio

ORG = "jeonghun917-org"
TEAMSPACE = "default-project"
STUDIO_NAME = "c3-cori-e100-e200"
STATE_PATH = Path("ops/cori-selfheal-state.json")
RUNNER_PATH = Path("runner/lightning_cori_continue_to550.py")
ENTRY_PATH = Path("runner/lightning_cori_segment_entry.py")
BASELINE_EPOCH = 100
BASELINE_STEP = 50200
BASELINE_SHA = "f4409103780820e356b609ec79c425cb1cffd3059fed163e1f60bfe926438273"
BASELINE_PATH = "/teamspace/studios/this_studio/c3-migration/c3-stage-src/handoff/cori_matcha_epoch100.ckpt"
MATCHA_COMMIT = "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
FINAL_EPOCH = 550
SEGMENT = 10
STALE_MINUTES = 120
MAX_CONSECUTIVE_FAILURES = 3


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def init_state() -> dict:
    return {
        "schema": "opened-arm-cori-selfheal-v1",
        "accepted_epoch": BASELINE_EPOCH,
        "accepted_global_step": BASELINE_STEP,
        "accepted_sha256": BASELINE_SHA,
        "accepted_checkpoint_path": BASELINE_PATH,
        "accepted_source_kind": "baseline",
        "current_target": 110,
        "current_job_name": None,
        "current_job_id": None,
        "current_job_submitted_at": None,
        "attempt_counter": 0,
        "consecutive_failures": 0,
        "last_status": "initialized",
        "updated_at": now_iso(),
        "history": [],
    }


def load_state() -> dict:
    if not STATE_PATH.exists():
        return init_state()
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def record(state: dict, event: str, **fields) -> None:
    item = {"at": now_iso(), "event": event, **fields}
    history = list(state.get("history", []))
    history.append(item)
    state["history"] = history[-80:]
    state["last_status"] = event
    state["updated_at"] = item["at"]


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_studio_ready(studio: Studio) -> None:
    if str(studio.status).lower() not in {"running", "started"}:
        studio.start(Machine.CPU)
    last = None
    for _ in range(36):
        try:
            out = str(studio.run("echo OPENED_ARM_STUDIO_READY"))
            if "OPENED_ARM_STUDIO_READY" in out:
                return
        except Exception as exc:
            last = exc
        time.sleep(5)
    raise RuntimeError(f"Studio never became shell-ready: {last!r}")


def stop_studio(studio: Studio) -> None:
    try:
        if str(studio.status).lower() not in {"stopped", "stopping"}:
            studio.stop()
    except Exception as exc:
        print("OPENED_ARM_STUDIO_STOP_ERROR", repr(exc), flush=True)


def verify_environment(studio: Studio) -> None:
    cmd = f'''set -euo pipefail
HOME_ROOT=/teamspace/studios/this_studio
ROOT="$HOME_ROOT/c3-migration/c3-stage-src"
CKPT="$ROOT/handoff/cori_matcha_epoch100.ckpt"
test -f "$CKPT"
test "$(sha256sum "$CKPT" | awk '{{print $1}}')" = "{BASELINE_SHA}"
test "$(git -C "$HOME_ROOT/src/Matcha-TTS" rev-parse HEAD)" = "{MATCHA_COMMIT}"
FREEZE=$(find "$ROOT/cori_dataset" -type f -path '*/metadata/FREEZE.json' -print -quit)
test -n "$FREEZE"
python -c "import sys,torch,lightning,matcha; assert sys.version_info[:2]==(3,11); assert torch.__version__.split('+')[0]=='2.5.1'; assert lightning.__version__=='2.6.5'"
echo OPENED_ARM_ENV_PASS
'''
    out = str(studio.run(cmd))
    if "OPENED_ARM_ENV_PASS" not in out:
        raise RuntimeError(f"environment gate failed: {out[-2000:]}")


def scan_checkpoints(studio: Studio, root: str) -> list[dict]:
    cmd = f'''set -euo pipefail
ROOT={shlex.quote(root)}
if [ ! -d "$ROOT" ]; then echo 'OPENED_ARM_SCAN=[]'; exit 0; fi
python - <<'PYSCAN'
import hashlib, json, pathlib, torch
root=pathlib.Path({root!r})
rows=[]
for p in root.rglob('*.ckpt'):
    try:
        x=torch.load(p,map_location='cpu',weights_only=False)
        epoch=int(x.get('epoch',-1))+1
        step=int(x.get('global_step',-1))
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
        rows.append({{'path':str(p),'epoch':epoch,'step':step,'sha256':h.hexdigest()}})
    except Exception as exc:
        print('OPENED_ARM_BAD_CKPT',p,type(exc).__name__,str(exc)[:300])
rows.sort(key=lambda r:(r['step'],r['epoch']))
print('OPENED_ARM_SCAN='+json.dumps(rows,separators=(',',':')))
PYSCAN
'''
    out = str(studio.run(cmd))
    for line in reversed(out.splitlines()):
        if line.startswith("OPENED_ARM_SCAN="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError("checkpoint scan produced no JSON marker")


def next_target(epoch: int) -> int | None:
    if epoch >= FINAL_EPOCH:
        return None
    return min(FINAL_EPOCH, ((epoch // SEGMENT) + 1) * SEGMENT)


def choose_checkpoint(rows: list[dict], accepted_epoch: int, accepted_step: int, target: int) -> dict | None:
    valid = [r for r in rows if accepted_epoch < int(r.get("epoch", -1)) <= target and int(r.get("step", -1)) > accepted_step]
    return max(valid, key=lambda r: (int(r["step"]), int(r["epoch"]))) if valid else None


def build_job_command(state: dict, target: int) -> str:
    runner_b64 = base64.b64encode(RUNNER_PATH.read_bytes()).decode("ascii")
    entry_b64 = base64.b64encode(ENTRY_PATH.read_bytes()).decode("ascii")
    resume_source = str(state["accepted_checkpoint_path"])
    accepted_sha = str(state["accepted_sha256"])
    accepted_epoch = int(state["accepted_epoch"])
    accepted_step = int(state["accepted_global_step"])
    return f'''set -euo pipefail
JOB_WORK_ROOT="$PWD"
HOME_ROOT=/teamspace/studios/this_studio
ROOT="$HOME_ROOT/c3-migration/c3-stage-src"
FREEZE=$(find "$ROOT/cori_dataset" -type f -path '*/metadata/FREEZE.json' -print -quit)
test -n "$FREEZE"
DATASET_ROOT=$(dirname "$(dirname "$FREEZE")")
HANDOFF="$ROOT/handoff"
BASELINE="$HANDOFF/cori_matcha_epoch100.ckpt"
test "$(sha256sum "$BASELINE" | awk '{{print $1}}')" = "{BASELINE_SHA}"
python -c "import sys,torch,lightning,matcha; assert sys.version_info[:2]==(3,11); assert torch.__version__.split('+')[0]=='2.5.1'; assert lightning.__version__=='2.6.5'; assert torch.cuda.is_available(); print('C3_GPU='+torch.cuda.get_device_name(0))"
RUNNER_TMP=/tmp/opened-arm-cori-runner
rm -rf "$RUNNER_TMP" && mkdir -p "$RUNNER_TMP"
python -c "import base64,pathlib; pathlib.Path('$RUNNER_TMP/lightning_cori_continue_to550.py').write_bytes(base64.b64decode('{runner_b64}')); pathlib.Path('$RUNNER_TMP/lightning_cori_segment_entry.py').write_bytes(base64.b64decode('{entry_b64}'))"
cd "$RUNNER_TMP"
python -m py_compile lightning_cori_continue_to550.py lightning_cori_segment_entry.py
OUTPUT_ROOT="$JOB_WORK_ROOT/c3-cori-lightning-runs"
RUN_DIR="$OUTPUT_ROOT/cori-e100-to-e550-b16"
mkdir -p "$RUN_DIR/resume-seed"
RESUME_SOURCE={shlex.quote(resume_source)}
test -f "$RESUME_SOURCE"
test "$(sha256sum "$RESUME_SOURCE" | awk '{{print $1}}')" = "{accepted_sha}"
cp "$RESUME_SOURCE" "$RUN_DIR/resume-seed/accepted-e{accepted_epoch:03d}-s{accepted_step}.ckpt"
exec python lightning_cori_segment_entry.py --handoff-dir "$HANDOFF" --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT" --matcha-checkout "$HOME_ROOT/src/Matcha-TTS" --target-max-epochs "{target}"
'''


def launch_segment(state: dict, target: int) -> None:
    if int(state.get("consecutive_failures", 0)) >= MAX_CONSECUTIVE_FAILURES:
        record(state, "halted_after_repeated_failures", target=target)
        save_state(state)
        raise RuntimeError("refusing paid retry after repeated failures")
    studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, org=ORG, create_ok=False)
    try:
        ensure_studio_ready(studio)
        verify_environment(studio)
        state["attempt_counter"] = int(state.get("attempt_counter", 0)) + 1
        attempt = int(state["attempt_counter"])
        accepted_epoch = int(state["accepted_epoch"])
        job_name = f"c3-cori-e{accepted_epoch:03d}-e{target:03d}-b16-oa{attempt:03d}"
        job = Job.run(command=build_job_command(state, target), name=job_name, machine=Machine.L4, studio=studio, interruptible=False)
        state.update({"current_target": target, "current_job_name": job_name, "current_job_id": getattr(job, "id", None), "current_job_submitted_at": now_iso()})
        record(state, "segment_submitted", target=target, job_name=job_name, job_id=getattr(job, "id", None), accepted_epoch=accepted_epoch, accepted_step=int(state["accepted_global_step"]))
        save_state(state)
        print("OPENED_ARM_SUBMITTED", job_name, getattr(job, "id", None), job.status, flush=True)
    finally:
        stop_studio(studio)


def process_terminal_job(state: dict, job: Job, status: str) -> bool:
    try:
        logs = str(job.logs)
    except Exception:
        logs = ""
    studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, org=ORG, create_ok=False)
    try:
        ensure_studio_ready(studio)
        rows = []
        for _ in range(18):
            rows = scan_checkpoints(studio, str(job.artifact_path))
            if rows:
                break
            time.sleep(5)
    finally:
        stop_studio(studio)

    accepted_epoch = int(state["accepted_epoch"])
    accepted_step = int(state["accepted_global_step"])
    target = int(state["current_target"])
    best = choose_checkpoint(rows, accepted_epoch, accepted_step, target)
    if best:
        state.update({"accepted_epoch": int(best["epoch"]), "accepted_global_step": int(best["step"]), "accepted_sha256": str(best["sha256"]), "accepted_checkpoint_path": str(best["path"]), "accepted_source_kind": "managed_job_artifact"})
        record(state, "checkpoint_recovered", job_name=job.name, terminal_status=status, epoch=int(best["epoch"]), global_step=int(best["step"]), sha256=str(best["sha256"]), path=str(best["path"]))

    reached = int(state["accepted_epoch"]) >= target
    if status == "completed" and reached:
        state["consecutive_failures"] = 0
        record(state, "segment_verified", target=target, epoch=int(state["accepted_epoch"]), step=int(state["accepted_global_step"]))
    else:
        state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
        record(state, "segment_needs_retry", target=target, terminal_status=status, recovered_epoch=int(state["accepted_epoch"]), recovered_step=int(state["accepted_global_step"]), failure_count=int(state["consecutive_failures"]), log_tail=logs[-3000:])

    state.update({"current_job_name": None, "current_job_id": None, "current_job_submitted_at": None})
    state["current_target"] = next_target(int(state["accepted_epoch"]))
    save_state(state)
    if state["current_target"] is None:
        record(state, "e550_verified_complete", epoch=int(state["accepted_epoch"]), step=int(state["accepted_global_step"]), sha256=state["accepted_sha256"])
        save_state(state)
        return False
    return True


def main() -> None:
    state = load_state()
    if not STATE_PATH.exists():
        save_state(state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    if int(state["accepted_epoch"]) >= FINAL_EPOCH:
        record(state, "already_complete", epoch=int(state["accepted_epoch"]), step=int(state["accepted_global_step"]))
        save_state(state)
        return

    current_name = state.get("current_job_name")
    if current_name:
        job = Job(str(current_name), teamspace=TEAMSPACE, org=ORG)
        status = str(job.status).lower()
        print("OPENED_ARM_CURRENT", current_name, status, flush=True)
        if status in {"pending", "running"}:
            submitted = parse_iso(state.get("current_job_submitted_at"))
            age_minutes = (datetime.now(timezone.utc) - submitted).total_seconds() / 60.0 if submitted else None
            if age_minutes is not None and age_minutes > STALE_MINUTES:
                job.stop()
                record(state, "stale_job_stop_requested", job_name=current_name, status=status, age_minutes=age_minutes)
                save_state(state)
            return
        if status in {"completed", "failed", "stopped", "cancelled", "canceled"}:
            if not process_terminal_job(state, job, status):
                return
            state = load_state()
        else:
            record(state, "unexpected_job_status", job_name=current_name, status=status)
            save_state(state)
            return

    target = state.get("current_target") or next_target(int(state["accepted_epoch"]))
    if target is not None:
        launch_segment(state, int(target))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        state = load_state()
        record(state, "controller_exception", error_type=type(exc).__name__, error=str(exc)[:4000])
        save_state(state)
        raise
