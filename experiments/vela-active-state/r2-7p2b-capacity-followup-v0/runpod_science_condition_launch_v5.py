#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_LAUNCHER_BLOB = "8f5b721d5d1e91888b335ed6756b13f13687c697"
BASE_LAUNCHER_PATH = Path(__file__).resolve().parent.parent / "r2-2p9b-full-control-v0" / "runpod_science_condition_launch_v4.sh"
CURRENT_DIR = Path(__file__).resolve().parent

def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()

def adapted_source() -> str:
    raw = BASE_LAUNCHER_PATH.read_bytes()
    actual = gitblob(raw)
    if actual != BASE_LAUNCHER_BLOB:
        raise RuntimeError(f"frozen RunPod v4 launcher drift: {actual}")
    source = raw.decode("utf-8")
    replacements = [
        ('MAX_RUNTIME_SECONDS=480','MAX_RUNTIME_SECONDS=900'),
        ('WORKLOAD_TIMEOUT_SECONDS=420','WORKLOAD_TIMEOUT_SECONDS=840'),
        ('--container-disk-in-gb 20','--container-disk-in-gb 50'),
        ('experiments/vela-active-state/r2-2p9b-full-control-v0','experiments/vela-active-state/r2-7p2b-capacity-followup-v0'),
        ('name="vela-p-r2-02-science-${VELA_CONDITION}-${VELA_SOURCE_COMMIT:0:10}"','name="vela-p-r2-03-science-${VELA_CONDITION}-${VELA_SOURCE_COMMIT:0:10}"'),
        ("'max_runtime_seconds':480","'max_runtime_seconds':900"),
    ]
    for old,new in replacements:
        count=source.count(old)
        if count < 1:
            raise RuntimeError(f"expected launcher seam missing: {old}")
        source=source.replace(old,new)
    for token in ['MAX_RUNTIME_SECONDS=480','WORKLOAD_TIMEOUT_SECONDS=420','--container-disk-in-gb 20','experiments/vela-active-state/r2-2p9b-full-control-v0','name="vela-p-r2-02-science-',"'max_runtime_seconds':480"]:
        if token in source:
            raise RuntimeError(f"unadapted launcher seam remains: {token}")
    for required in ['MAX_RUNTIME_SECONDS=900','WORKLOAD_TIMEOUT_SECONDS=840','--container-disk-in-gb 50','RUNPOD_HTTPS_PROXY_V1','CONTAINER_RESTART_DETECTED_BY_CONTROL_PLANE','cleanup_pod','--ports "${RESULT_PORT}/http"']:
        if required not in source:
            raise RuntimeError(f"required inherited v4 behavior missing: {required}")
    return source

def main() -> None:
    source=adapted_source()
    if len(sys.argv)==3 and sys.argv[1]=='--emit-adapted':
        Path(sys.argv[2]).write_text(source,encoding='utf-8')
        return
    if len(sys.argv)!=1:
        raise SystemExit('usage: runpod_science_condition_launch_v5.py [--emit-adapted PATH]')
    env=os.environ.copy()
    env.setdefault('VELA_RUNPOD_CAPACITY_SELECTOR',str(CURRENT_DIR/'runpod_select_a40_capacity_v1.py'))
    env.setdefault('VELA_RUNPOD_PROXY_SUPERVISOR',str(CURRENT_DIR/'runpod_science_proxy_supervisor_v1.py'))
    fd,tmp=tempfile.mkstemp(prefix='vela-r2-03-launch-',suffix='.sh')
    os.close(fd)
    path=Path(tmp); path.write_text(source,encoding='utf-8'); path.chmod(0o700)
    try:
        proc=subprocess.run(['bash',str(path)],env=env,check=False)
        raise SystemExit(proc.returncode)
    finally:
        path.unlink(missing_ok=True)

if __name__=='__main__':
    main()
