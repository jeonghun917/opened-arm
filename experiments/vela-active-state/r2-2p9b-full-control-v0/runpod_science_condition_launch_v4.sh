#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required}"
: "${VELA_SOURCE_COMMIT:?VELA_SOURCE_COMMIT is required}"
: "${VELA_CONDITION:?VELA_CONDITION is required}"
: "${VELA_RUNPOD_DATA_CENTER_ID:?VELA_RUNPOD_DATA_CENTER_ID is required}"

if [[ "${VELA_RUNPOD_SCIENCE_AUTHORIZED:-false}" != "true" ]]; then
  echo "RUNPOD_SCIENCE_BLOCKED: canonical scientific authority not present" >&2
  exit 77
fi
case "$VELA_CONDITION" in B|C|G|M|X) ;; *) echo "invalid condition $VELA_CONDITION" >&2; exit 76;; esac

GPU_ID="NVIDIA A40"
GPU_PRICE_CAP="0.49"
IMAGE_TAG="python:3.12.13-slim-bookworm"
MAX_RUNTIME_SECONDS=480
WORKLOAD_TIMEOUT_SECONDS=420
PROXY_STARTUP_GRACE_SECONDS=120
POLL_INTERVAL_SECONDS=3
RESULT_PORT=8000
EVIDENCE_DIR="${VELA_SCIENCE_EVIDENCE_DIR:-/tmp/runpod-science-${VELA_CONDITION}}"
CAPACITY_SELECTOR="${VELA_RUNPOD_CAPACITY_SELECTOR:-$(dirname "$0")/runpod_select_a40_capacity_v1.py}"
SUPERVISOR="${VELA_RUNPOD_PROXY_SUPERVISOR:-$(dirname "$0")/runpod_science_proxy_supervisor_v1.py}"

mkdir -p "$EVIDENCE_DIR/polls"
CONTROLLER_START_EPOCH="$(date +%s)"
POD_ID=""
DELETED=0
WATCHDOG_PID=""
RENT_START_EPOCH=""

sanitize_pod_get() {
  local raw="$1" out="$2"
  python - "$raw" "$out" <<'PY'
import json,sys
src,out=sys.argv[1:]
try: x=json.load(open(src))
except Exception:
    open(out,'w').write('{}\n'); raise SystemExit(0)
env=x.get('env')
if isinstance(env,dict):
    for k in ('VELA_RESULT_TOKEN','VELA_SUPERVISOR_B64'):
        if k in env: env[k]='<redacted>'
json.dump(x,open(out,'w'),indent=2,sort_keys=True)
PY
}

cleanup_pod() {
  set +e
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
    wait "$WATCHDOG_PID" >/dev/null 2>&1 || true
    WATCHDOG_PID=""
  fi
  if [[ -n "$POD_ID" && "$DELETED" != "1" ]]; then
    local raw="/tmp/vela-final-pod-get-${POD_ID}.json"
    runpodctl pod get "$POD_ID" --include-machine > "$raw" 2> "$EVIDENCE_DIR/pod-final-get.err" || true
    sanitize_pod_get "$raw" "$EVIDENCE_DIR/pod-final-get.json" || true
    rm -f "$raw"
    if runpodctl pod delete "$POD_ID" > "$EVIDENCE_DIR/pod-delete.json" 2> "$EVIDENCE_DIR/pod-delete.err"; then
      DELETED=1
    elif grep -q '"code":"not_found"' "$EVIDENCE_DIR/pod-delete.err" 2>/dev/null; then
      DELETED=1
    fi
  fi
  set -e
}
trap cleanup_pod EXIT INT TERM

runpodctl version > "$EVIDENCE_DIR/runpodctl-version.txt"
runpodctl user >/dev/null
runpodctl gpu list --include-unavailable > "$EVIDENCE_DIR/gpu-list.json"
python "$CAPACITY_SELECTOR" "$EVIDENCE_DIR/gpu-list.json" "$EVIDENCE_DIR/selected-gpu.json" --require-data-center "$VELA_RUNPOD_DATA_CENTER_ID" > "$EVIDENCE_DIR/capacity-validation.json"

docker buildx imagetools inspect "$IMAGE_TAG" --raw > "$EVIDENCE_DIR/image-manifest.json"
IMAGE_DIGEST="$(python - "$EVIDENCE_DIR/image-manifest.json" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read(); j=json.loads(raw)
if 'manifests' in j:
    for m in j['manifests']:
        p=m.get('platform') or {}
        if p.get('os')=='linux' and p.get('architecture')=='amd64':
            print(m['digest']); break
    else: raise SystemExit('linux/amd64 manifest missing')
else: print('sha256:'+hashlib.sha256(raw).hexdigest())
PY
)"
IMAGE_REF="python@${IMAGE_DIGEST}"
printf '%s\n' "$IMAGE_REF" > "$EVIDENCE_DIR/image-ref.txt"
BASE_RAW="https://raw.githubusercontent.com/jeonghun917/opened-arm/${VELA_SOURCE_COMMIT}/experiments/vela-active-state/r2-2p9b-full-control-v0"
SUPERVISOR_B64="$(base64 -w0 "$SUPERVISOR")"
RESULT_TOKEN="$(python - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)"
printf '%s' "$RESULT_TOKEN" | sha256sum | awk '{print $1}' > "$EVIDENCE_DIR/result-token-sha256.txt"

ENV_JSON="$(python - "$SUPERVISOR_B64" "$RESULT_TOKEN" "$VELA_SOURCE_COMMIT" "$BASE_RAW" "$VELA_CONDITION" "$RESULT_PORT" "$WORKLOAD_TIMEOUT_SECONDS" <<'PY'
import json,sys
sup,token,source,base,condition,port,timeout=sys.argv[1:]
print(json.dumps({
  'VELA_SUPERVISOR_B64':sup,
  'VELA_RESULT_TOKEN':token,
  'VELA_SOURCE_COMMIT':source,
  'VELA_BASE_RAW':base,
  'VELA_CONDITION':condition,
  'VELA_RESULT_PORT':port,
  'VELA_WORKLOAD_TIMEOUT_SECONDS':timeout,
  'PYTHONUNBUFFERED':'1'
},separators=(',',':')))
PY
)"
DOCKER_ARGS="sh -lc 'echo \"\$VELA_SUPERVISOR_B64\" | base64 -d > /tmp/vela_proxy_supervisor.py && exec python /tmp/vela_proxy_supervisor.py'"
name="vela-p-r2-02-science-${VELA_CONDITION}-${VELA_SOURCE_COMMIT:0:10}"

set +e
runpodctl pod create \
  --name "$name" \
  --image "$IMAGE_REF" \
  --gpu-id "$GPU_ID" \
  --gpu-count 1 \
  --cloud-type SECURE \
  --data-center-ids "$VELA_RUNPOD_DATA_CENTER_ID" \
  --container-disk-in-gb 20 \
  --ports "${RESULT_PORT}/http" \
  --ssh=false \
  --min-cuda-version 12.6 \
  --env "$ENV_JSON" \
  --docker-args "$DOCKER_ARGS" \
  > "$EVIDENCE_DIR/pod-create.json" 2> "$EVIDENCE_DIR/pod-create.err"
CREATE_RC=$?
set -e
if [[ "$CREATE_RC" != "0" ]]; then
  python - "$EVIDENCE_DIR" "$VELA_CONDITION" "$VELA_RUNPOD_DATA_CENTER_ID" "$CREATE_RC" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); err=(p/'pod-create.err').read_text(errors='replace') if (p/'pod-create.err').exists() else ''
json.dump({'status':'PROVIDER_CREATE_FAILED_BEFORE_POD_ID','scientific_evidence':False,'condition':sys.argv[2],'dataCenterId':sys.argv[3],'exitCode':int(sys.argv[4]),'stderr':err[-4000:]},open(p/'provider-create-failure.json','w'),indent=2,sort_keys=True)
PY
  cat "$EVIDENCE_DIR/pod-create.err" >&2 || true
  exit "$CREATE_RC"
fi

POD_ID="$(python - "$EVIDENCE_DIR/pod-create.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); v=x.get('id') or x.get('podId')
if not v: raise SystemExit(f'pod id missing: {x}')
print(v)
PY
)"
printf '%s\n' "$POD_ID" > "$EVIDENCE_DIR/pod-id.txt"
RENT_START_EPOCH="$(date +%s)"
PROXY_BASE="https://${POD_ID}-${RESULT_PORT}.proxy.runpod.net"
printf '%s\n' "https://${POD_ID}-${RESULT_PORT}.proxy.runpod.net" > "$EVIDENCE_DIR/proxy-base.txt"

python - "$MAX_RUNTIME_SECONDS" "$POD_ID" "$EVIDENCE_DIR" <<'PYWATCH' &
import subprocess,sys,time
seconds=int(sys.argv[1]); pod_id=sys.argv[2]; evidence=sys.argv[3]
time.sleep(seconds)
open(evidence+"/watchdog-fired.txt","w").write("WATCHDOG_FIRED\n")
with open(evidence+"/watchdog-delete.json","w") as out, open(evidence+"/watchdog-delete.err","w") as err:
    subprocess.run(["runpodctl","pod","delete",pod_id],stdout=out,stderr=err,check=False)
PYWATCH
WATCHDOG_PID=$!

DEADLINE=$((RENT_START_EPOCH + MAX_RUNTIME_SECONDS))
FIRST_RUNNING_EPOCH=""
FIRST_PROXY_OK_EPOCH=""
LAST_UPTIME=""
POLL=0
FOUND=0
TERMINAL_REASON=""
TERMINAL_STATE=""

while (( $(date +%s) < DEADLINE )); do
  POLL=$((POLL + 1))
  poll_base="$EVIDENCE_DIR/polls/poll-$(printf '%04d' "$POLL")"
  raw_get="/tmp/vela-pod-get-${POD_ID}.json"
  runpodctl pod get "$POD_ID" --include-machine > "$raw_get" 2> "${poll_base}-get.err" || true
  sanitize_pod_get "$raw_get" "${poll_base}-get.json" || true
  read -r runtime uptime <<EOF
$(python - "$raw_get" <<'PY'
import json,sys
try: x=json.load(open(sys.argv[1]))
except Exception: print('unknown NA'); raise SystemExit
u=x.get('uptimeSeconds'); print(x.get('runtimeStatus') or 'unknown', u if isinstance(u,(int,float)) else 'NA')
PY
)
EOF
  rm -f "$raw_get"

  now="$(date +%s)"
  if [[ "$runtime" == "running" && -z "$FIRST_RUNNING_EPOCH" ]]; then FIRST_RUNNING_EPOCH="$now"; fi
  if [[ "$uptime" != "NA" ]]; then
    if [[ -n "$LAST_UPTIME" ]] && (( uptime + 10 < LAST_UPTIME )) && (( LAST_UPTIME >= 20 )); then
      TERMINAL_REASON="CONTAINER_RESTART_DETECTED_BY_CONTROL_PLANE"
      TERMINAL_STATE="FAILED"
      break
    fi
    LAST_UPTIME="$uptime"
  fi

  status_file="${poll_base}-proxy-status.json"
  status_err="${poll_base}-proxy-status.err"
  http_code="$(curl --silent --show-error --location --connect-timeout 3 --max-time 5 \
    --output "$status_file" --write-out '%{http_code}' \
    "$PROXY_BASE/v1/status/$RESULT_TOKEN" 2> "$status_err" || true)"
  printf '%s\n' "$http_code" > "${poll_base}-proxy-status.code"

  if [[ "$http_code" == "200" ]]; then
    [[ -n "$FIRST_PROXY_OK_EPOCH" ]] || FIRST_PROXY_OK_EPOCH="$now"
    read -r state condition source schema <<EOF
$(python - "$status_file" <<'PY'
import json,sys
try: x=json.load(open(sys.argv[1]))
except Exception: print('INVALID INVALID INVALID INVALID'); raise SystemExit
print(x.get('state','INVALID'),x.get('condition','INVALID'),x.get('source_commit','INVALID'),x.get('schema','INVALID'))
PY
)
EOF
    if [[ "$schema" != "vela-runpod-science-proxy-status:v1" || "$condition" != "$VELA_CONDITION" || "$source" != "$VELA_SOURCE_COMMIT" ]]; then
      TERMINAL_REASON="PROXY_STATUS_IDENTITY_MISMATCH"
      TERMINAL_STATE="FAILED"
      cp "$status_file" "$EVIDENCE_DIR/terminal-proxy-status.json" || true
      break
    fi
    if [[ "$state" == "FAILED" ]]; then
      TERMINAL_REASON="SUPERVISOR_REPORTED_FAILED"
      TERMINAL_STATE="FAILED"
      cp "$status_file" "$EVIDENCE_DIR/terminal-proxy-status.json"
      break
    fi
    if [[ "$state" == "COMPLETE" ]]; then
      cp "$status_file" "$EVIDENCE_DIR/terminal-proxy-status.json"
      result_tmp="$EVIDENCE_DIR/condition_result_${VELA_CONDITION}.json.tmp"
      result_code="$(curl --silent --show-error --location --connect-timeout 3 --max-time 10 \
        --output "$result_tmp" --write-out '%{http_code}' \
        "$PROXY_BASE/v1/result/$RESULT_TOKEN" 2> "$EVIDENCE_DIR/result-fetch.err" || true)"
      printf '%s\n' "$result_code" > "$EVIDENCE_DIR/result-fetch.code"
      if [[ "$result_code" != "200" ]]; then
        TERMINAL_REASON="COMPLETE_STATUS_RESULT_FETCH_FAILED_${result_code}"
        TERMINAL_STATE="FAILED"
        break
      fi
      if python - "$status_file" "$result_tmp" "$VELA_CONDITION" "$VELA_SOURCE_COMMIT" <<'PY'
import hashlib,json,sys
status_path,result_path,condition,source=sys.argv[1:]
s=json.load(open(status_path)); raw=open(result_path,'rb').read(); r=json.loads(raw)
assert len(raw)==int(s['result_bytes'])
assert hashlib.sha256(raw).hexdigest()==s['result_sha256']
assert r.get('status')=='P-R2-02_CONDITION_RESULT'
assert r.get('scientific_evidence') is True
assert r.get('condition')==condition and r.get('source_commit')==source
PY
      then
        mv "$result_tmp" "$EVIDENCE_DIR/condition_result_${VELA_CONDITION}.json"
        FOUND=1
        TERMINAL_REASON="RESULT_FETCHED_VIA_RUNPOD_HTTPS_PROXY"
        TERMINAL_STATE="COMPLETE"
      else
        rm -f "$result_tmp"
        TERMINAL_REASON="PROXY_RESULT_INTEGRITY_OR_IDENTITY_FAILURE"
        TERMINAL_STATE="FAILED"
      fi
      break
    fi
  fi

  if [[ -n "$FIRST_RUNNING_EPOCH" && -z "$FIRST_PROXY_OK_EPOCH" ]] && (( now - FIRST_RUNNING_EPOCH >= PROXY_STARTUP_GRACE_SECONDS )); then
    TERMINAL_REASON="PROXY_UNREACHABLE_AFTER_RUNNING_GRACE"
    TERMINAL_STATE="FAILED"
    break
  fi
  sleep "$POLL_INTERVAL_SECONDS"
done

if [[ -z "$TERMINAL_REASON" ]]; then
  TERMINAL_REASON="CONTROLLER_MAX_RUNTIME_EXPIRED"
  TERMINAL_STATE="FAILED"
fi

cleanup_pod
if [[ "$DELETED" != "1" ]]; then
  echo "RUNPOD_POD_DELETE_FAILED condition=$VELA_CONDITION" >&2
  exit 79
fi
trap - EXIT INT TERM
printf '%s\n' "$TERMINAL_REASON" > "$EVIDENCE_DIR/terminal-reason.txt"
printf '%s\n' "$TERMINAL_STATE" > "$EVIDENCE_DIR/terminal-state.txt"

END_EPOCH="$(date +%s)"
ELAPSED=$(( END_EPOCH - RENT_START_EPOCH ))
if (( ELAPSED < 0 )); then ELAPSED=0; fi
if (( ELAPSED > MAX_RUNTIME_SECONDS )); then ELAPSED=$MAX_RUNTIME_SECONDS; fi

if [[ "$FOUND" != "1" ]]; then
  python - "$EVIDENCE_DIR" "$VELA_CONDITION" "$VELA_SOURCE_COMMIT" "$ELAPSED" "$GPU_PRICE_CAP" "$TERMINAL_REASON" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); c,source=sys.argv[2],sys.argv[3]; elapsed=int(sys.argv[4]); price=float(sys.argv[5]); reason=sys.argv[6]
status={}
try: status=json.load(open(p/'terminal-proxy-status.json'))
except Exception: pass
env={'status':'FAIL','scientific_evidence':False,'provider':'RUNPOD','phase':'SCIENTIFIC_CONDITION','condition':c,'source_commit':source,'pod_deleted':True,'elapsed_seconds':elapsed,'max_runtime_seconds':480,'observed_secure_price_per_hr_usd':price,'estimated_gpu_cost_upper_bound_usd':round(elapsed*price/3600,6),'transport':'RUNPOD_HTTPS_PROXY_V1','transport_terminal_reason':reason,'supervisor_status':status}
json.dump(env,open(p/f'condition_failure_envelope_{c}.json','w'),indent=2,sort_keys=True)
PY
  echo "RUNPOD_SCIENCE_TRANSPORT_FAILED condition=$VELA_CONDITION reason=$TERMINAL_REASON" >&2
  exit 78
fi

python - "$EVIDENCE_DIR" "$VELA_CONDITION" "$VELA_SOURCE_COMMIT" "$ELAPSED" "$GPU_PRICE_CAP" "$TERMINAL_REASON" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); c,source=sys.argv[2],sys.argv[3]; elapsed=int(sys.argv[4]); price=float(sys.argv[5]); terminal=sys.argv[6]
rp=p/f'condition_result_{c}.json'; raw=rp.read_bytes(); r=json.loads(raw)
assert r['status']=='P-R2-02_CONDITION_RESULT' and r['scientific_evidence'] is True
assert r['condition']==c and r['source_commit']==source
env={'status':'PASS','scientific_evidence':True,'provider':'RUNPOD','phase':'SCIENTIFIC_CONDITION','condition':c,'source_commit':source,'pod_deleted':True,'elapsed_seconds':elapsed,'max_runtime_seconds':480,'observed_secure_price_per_hr_usd':price,'estimated_gpu_cost_upper_bound_usd':round(elapsed*price/3600,6),'gpu':json.load(open(p/'selected-gpu.json')),'image_ref':(p/'image-ref.txt').read_text().strip(),'condition_result_sha256':hashlib.sha256(raw).hexdigest(),'transport':'RUNPOD_HTTPS_PROXY_V1','transport_terminal_reason':terminal}
json.dump(env,open(p/f'condition_envelope_{c}.json','w'),indent=2,sort_keys=True)
print(json.dumps(env,separators=(',',':'),sort_keys=True))
PY
