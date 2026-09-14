#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SOURCE = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def common(root: Path, mode: str) -> tuple[Path, dict[str, str]]:
    bindir = root / "bin"
    evidence = root / "evidence"
    bindir.mkdir(); evidence.mkdir()
    selector = root / "selector.py"
    write(selector, """#!/usr/bin/env python3
import json,sys
x={'status':'PASS','paid_mutation':False,'gpuId':'NVIDIA A40','displayName':'A40','memoryInGb':48,'secureCloud':True,'securePricePerHr':0.49,'inventoryStockStatus':'High','selectedDataCenter':{'dataCenterId':'EU-SE-1','stockStatus':'High'}}
json.dump(x,open(sys.argv[2],'w')); print(json.dumps(x,separators=(',',':')))
""")
    write(bindir / "docker", """#!/bin/sh
printf '%s\\n' '{"manifests":[{"digest":"sha256:abc123","platform":{"os":"linux","architecture":"amd64"}}]}'
""")
    state = root / "get-count"
    uptime_logic = "u=$((20+n*3))" if mode == "success" else "case $n in 1) u=30;; 2) u=35;; *) u=4;; esac"
    write(bindir / "runpodctl", f"""#!/bin/sh
set -eu
state='{state}'
cmd1=${{1-}}; cmd2=${{2-}}
case \"$cmd1 $cmd2\" in
  'version ') echo 'runpodctl 2.14.0-test';;
  'user ') echo '{{}}';;
  'gpu list') echo '[{{\"gpuId\":\"NVIDIA A40\"}}]';;
  'pod create') echo '{{\"id\":\"testpod123\"}}';;
  'pod get') n=0; [ -f \"$state\" ] && n=$(cat \"$state\"); n=$((n+1)); echo \"$n\" > \"$state\"; {uptime_logic}; echo \"{{\\\"id\\\":\\\"testpod123\\\",\\\"runtimeStatus\\\":\\\"running\\\",\\\"uptimeSeconds\\\":$u,\\\"env\\\":{{\\\"VELA_RESULT_TOKEN\\\":\\\"secret\\\",\\\"VELA_SUPERVISOR_B64\\\":\\\"blob\\\"}}}}\";;
  'pod delete') echo '{{\"deleted\":true,\"id\":\"testpod123\"}}';;
  *) echo \"unexpected runpodctl $*\" >&2; exit 2;;
esac
""")
    env = os.environ.copy()
    env.update({
        "PATH": str(bindir) + os.pathsep + env["PATH"],
        "RUNPOD_API_KEY": "x",
        "VELA_SOURCE_COMMIT": SOURCE,
        "VELA_CONDITION": "B",
        "VELA_RUNPOD_DATA_CENTER_ID": "EU-SE-1",
        "VELA_RUNPOD_SCIENCE_AUTHORIZED": "true",
        "VELA_SCIENCE_EVIDENCE_DIR": str(evidence),
        "VELA_RUNPOD_CAPACITY_SELECTOR": str(selector),
        "VELA_RUNPOD_PROXY_SUPERVISOR": str(Path(__file__).with_name("runpod_science_proxy_supervisor_v1.py")),
    })
    return evidence, env


def test_success(launcher: Path) -> None:
    root = Path(tempfile.mkdtemp(prefix="vela-launch-v4-success-"))
    evidence, env = common(root, "success")
    count = root / "curl-count"
    result = {
        "status": "P-R2-02_CONDITION_RESULT",
        "scientific_evidence": True,
        "condition": "B",
        "source_commit": SOURCE,
        "scales": {"0.4B": {}, "2.9B": {}},
        "runtime": {"device": "NVIDIA A40", "device_capability": [8, 6]},
    }
    raw = json.dumps(result, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    status = {
        "schema": "vela-runpod-science-proxy-status:v1",
        "state": "COMPLETE",
        "phase": "RESULT_READY",
        "condition": "B",
        "source_commit": SOURCE,
        "boot_id": "x",
        "result_bytes": len(raw),
        "result_sha256": __import__("hashlib").sha256(raw).hexdigest(),
    }
    curl = root / "bin" / "curl"
    write(curl, f"""#!/bin/sh
set -eu
out=''; url=''; prev=''
for a in \"$@\"; do
  if [ \"$prev\" = o ]; then out=\"$a\"; prev=''; continue; fi
  [ \"$a\" = --output ] && prev=o
  case \"$a\" in http*) url=\"$a\";; esac
done
if echo \"$url\" | grep -q '/v1/result/'; then
  cat > \"$out\" <<'JSON'
{raw.decode().rstrip()}
JSON
  printf '200'; exit 0
fi
n=0; [ -f '{count}' ] && n=$(cat '{count}'); n=$((n+1)); echo \"$n\" > '{count}'
if [ \"$n\" -lt 3 ]; then
  cat > \"$out\" <<'JSON'
{{"schema":"vela-runpod-science-proxy-status:v1","state":"RUNNING","phase":"RUN_CONDITION","condition":"B","source_commit":"{SOURCE}","boot_id":"x"}}
JSON
else
  cat > \"$out\" <<'JSON'
{json.dumps(status,separators=(',',':'))}
JSON
fi
printf '200'
""")
    proc = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, timeout=45)
    if proc.returncode != 0:
        raise AssertionError(proc.stdout + "\n" + proc.stderr)
    envelope = json.load(open(evidence / "condition_envelope_B.json"))
    assert envelope["status"] == "PASS"
    assert envelope["pod_deleted"] is True
    assert envelope["transport"] == "RUNPOD_HTTPS_PROXY_V1"
    for path in (evidence / "polls").glob("*-get.json"):
        txt = path.read_text()
        assert "secret" not in txt and "blob" not in txt
    shutil.rmtree(root)


def test_restart_fail_closed(launcher: Path) -> None:
    root = Path(tempfile.mkdtemp(prefix="vela-launch-v4-restart-"))
    evidence, env = common(root, "restart")
    write(root / "bin" / "curl", """#!/bin/sh
out=''; prev=''
for a in "$@"; do
  if [ "$prev" = o ]; then out="$a"; prev=''; continue; fi
  [ "$a" = --output ] && prev=o
done
[ -n "$out" ] && : > "$out"
printf '000'
exit 1
""")
    proc = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, timeout=45)
    assert proc.returncode == 78, proc.stdout + "\n" + proc.stderr
    assert (evidence / "terminal-reason.txt").read_text().strip() == "CONTAINER_RESTART_DETECTED_BY_CONTROL_PLANE"
    failure = json.load(open(evidence / "condition_failure_envelope_B.json"))
    assert failure["pod_deleted"] is True and failure["scientific_evidence"] is False
    shutil.rmtree(root)


def main() -> None:
    launcher = Path(__file__).with_name("runpod_science_condition_launch_v4.sh")
    test_success(launcher)
    test_restart_fail_closed(launcher)
    print("RUNPOD_SCIENCE_LAUNCHER_V4_SYNTHETIC_PASS")


if __name__ == "__main__":
    main()
