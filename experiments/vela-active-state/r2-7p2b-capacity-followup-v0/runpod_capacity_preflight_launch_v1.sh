#!/usr/bin/env bash
set -euo pipefail
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required}"
: "${VELA_SOURCE_COMMIT:?VELA_SOURCE_COMMIT required}"
: "${VELA_RUNPOD_DATA_CENTER_ID:?VELA_RUNPOD_DATA_CENTER_ID required}"
if [[ "${VELA_RUNPOD_PREFLIGHT_AUTHORIZED:-false}" != "true" ]]; then echo "RUNPOD_PREFLIGHT_BLOCKED" >&2; exit 77; fi
GPU_ID="NVIDIA A40"; GPU_PRICE_CAP="0.49"; IMAGE_TAG="python:3.12.13-slim-bookworm"; MAX_RUNTIME_SECONDS=900; WORKLOAD_TIMEOUT_SECONDS=840; PROXY_STARTUP_GRACE_SECONDS=120; RESULT_PORT=8000; POLL_INTERVAL_SECONDS=3
EVIDENCE_DIR="${VELA_PREFLIGHT_EVIDENCE_DIR:-/tmp/runpod-capacity-preflight}"; CAPACITY_SELECTOR="${VELA_RUNPOD_CAPACITY_SELECTOR:-$(dirname "$0")/runpod_select_a40_capacity_v1.py}"; SUPERVISOR="${VELA_RUNPOD_PREFLIGHT_SUPERVISOR:-$(dirname "$0")/runpod_capacity_preflight_supervisor_v1.py}"
mkdir -p "$EVIDENCE_DIR/polls"; POD_ID=""; DELETED=0; WATCHDOG_PID=""; RENT_START_EPOCH=""
cleanup_pod(){ set +e; if [[ -n "$WATCHDOG_PID" ]]; then kill "$WATCHDOG_PID" >/dev/null 2>&1||true; wait "$WATCHDOG_PID" >/dev/null 2>&1||true; WATCHDOG_PID=""; fi; if [[ -n "$POD_ID" && "$DELETED" != 1 ]]; then runpodctl pod get "$POD_ID" --include-machine > "$EVIDENCE_DIR/pod-final-get.json" 2> "$EVIDENCE_DIR/pod-final-get.err"||true; if runpodctl pod delete "$POD_ID" > "$EVIDENCE_DIR/pod-delete.json" 2> "$EVIDENCE_DIR/pod-delete.err"; then DELETED=1; elif grep -q '"code":"not_found"' "$EVIDENCE_DIR/pod-delete.err" 2>/dev/null; then DELETED=1; fi; fi; set -e; }
trap cleanup_pod EXIT INT TERM
runpodctl version > "$EVIDENCE_DIR/runpodctl-version.txt"; runpodctl user >/dev/null; runpodctl gpu list --include-unavailable > "$EVIDENCE_DIR/gpu-list.json"; python "$CAPACITY_SELECTOR" "$EVIDENCE_DIR/gpu-list.json" "$EVIDENCE_DIR/selected-gpu.json" --require-data-center "$VELA_RUNPOD_DATA_CENTER_ID" > "$EVIDENCE_DIR/capacity-validation.json"
docker buildx imagetools inspect "$IMAGE_TAG" --raw > "$EVIDENCE_DIR/image-manifest.json"
IMAGE_DIGEST="$(python - "$EVIDENCE_DIR/image-manifest.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]));
for m in j.get('manifests',[]):
 p=m.get('platform') or {}
 if p.get('os')=='linux' and p.get('architecture')=='amd64': print(m['digest']); break
else: raise SystemExit('linux/amd64 image missing')
PY
)"; IMAGE_REF="python@${IMAGE_DIGEST}"; echo "$IMAGE_REF" > "$EVIDENCE_DIR/image-ref.txt"
BASE_RAW="https://raw.githubusercontent.com/jeonghun917/opened-arm/${VELA_SOURCE_COMMIT}/experiments/vela-active-state/r2-7p2b-capacity-followup-v0"; SUPERVISOR_B64="$(base64 -w0 "$SUPERVISOR")"; RESULT_TOKEN="$(python - <<'PY'
import secrets; print(secrets.token_hex(32))
PY
)"; printf '%s' "$RESULT_TOKEN"|sha256sum|awk '{print $1}' > "$EVIDENCE_DIR/result-token-sha256.txt"
ENV_JSON="$(python - "$SUPERVISOR_B64" "$RESULT_TOKEN" "$VELA_SOURCE_COMMIT" "$BASE_RAW" "$RESULT_PORT" "$WORKLOAD_TIMEOUT_SECONDS" <<'PY'
import json,sys
sup,tok,src,base,port,timeout=sys.argv[1:]; print(json.dumps({'VELA_SUPERVISOR_B64':sup,'VELA_RESULT_TOKEN':tok,'VELA_SOURCE_COMMIT':src,'VELA_BASE_RAW':base,'VELA_RESULT_PORT':port,'VELA_PREFLIGHT_WORKLOAD_TIMEOUT_SECONDS':timeout,'PYTHONUNBUFFERED':'1'},separators=(',',':')))
PY
)"; DOCKER_ARGS="sh -lc 'echo \"\$VELA_SUPERVISOR_B64\" | base64 -d > /tmp/vela_preflight_supervisor.py && exec python /tmp/vela_preflight_supervisor.py'"; name="vela-p-r2-03-preflight-${VELA_SOURCE_COMMIT:0:10}"
set +e; runpodctl pod create --name "$name" --image "$IMAGE_REF" --gpu-id "$GPU_ID" --gpu-count 1 --cloud-type SECURE --data-center-ids "$VELA_RUNPOD_DATA_CENTER_ID" --container-disk-in-gb 50 --ports "${RESULT_PORT}/http" --ssh=false --min-cuda-version 12.6 --env "$ENV_JSON" --docker-args "$DOCKER_ARGS" > "$EVIDENCE_DIR/pod-create.json" 2> "$EVIDENCE_DIR/pod-create.err"; CREATE_RC=$?; set -e
if [[ "$CREATE_RC" != 0 ]]; then echo "PREFLIGHT_PROVIDER_CREATE_FAILED" >&2; exit "$CREATE_RC"; fi
POD_ID="$(python - "$EVIDENCE_DIR/pod-create.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x.get('id') or x.get('podId') or (_ for _ in ()).throw(RuntimeError('pod id missing')))
PY
)"; echo "$POD_ID" > "$EVIDENCE_DIR/pod-id.txt"; RENT_START_EPOCH="$(date +%s)"; PROXY_BASE="https://${POD_ID}-${RESULT_PORT}.proxy.runpod.net"
python - "$MAX_RUNTIME_SECONDS" "$POD_ID" "$EVIDENCE_DIR" <<'PY' &
import subprocess,sys,time
time.sleep(int(sys.argv[1])); open(sys.argv[3]+'/watchdog-fired.txt','w').write('WATCHDOG_FIRED\n'); subprocess.run(['runpodctl','pod','delete',sys.argv[2]],stdout=open(sys.argv[3]+'/watchdog-delete.json','w'),stderr=open(sys.argv[3]+'/watchdog-delete.err','w'),check=False)
PY
WATCHDOG_PID=$!
DEADLINE=$((RENT_START_EPOCH+MAX_RUNTIME_SECONDS)); FIRST_RUNNING=""; FIRST_PROXY=""; LAST_UPTIME=""; FOUND=0; REASON=""; POLL=0
while (( $(date +%s)<DEADLINE )); do POLL=$((POLL+1)); base="$EVIDENCE_DIR/polls/poll-$(printf '%04d' "$POLL")"; runpodctl pod get "$POD_ID" --include-machine > "$base-get.json" 2> "$base-get.err"||true; read -r runtime uptime <<EOF
$(python - "$base-get.json" <<'PY'
import json,sys
try:x=json.load(open(sys.argv[1]))
except:print('unknown NA');raise SystemExit
u=x.get('uptimeSeconds'); print(x.get('runtimeStatus') or 'unknown',u if isinstance(u,(int,float)) else 'NA')
PY
)
EOF
now="$(date +%s)"; [[ "$runtime" != running || -n "$FIRST_RUNNING" ]]||FIRST_RUNNING="$now"; if [[ "$uptime" != NA ]]; then if [[ -n "$LAST_UPTIME" ]]&&(( uptime+10<LAST_UPTIME ))&&(( LAST_UPTIME>=20 )); then REASON="CONTAINER_RESTART_DETECTED_BY_CONTROL_PLANE"; break; fi; LAST_UPTIME="$uptime"; fi
code="$(curl -sS -L --connect-timeout 3 --max-time 5 -o "$base-status.json" -w '%{http_code}' "$PROXY_BASE/v1/status/$RESULT_TOKEN" 2> "$base-status.err"||true)"; echo "$code" > "$base-status.code"
if [[ "$code" == 200 ]]; then [[ -n "$FIRST_PROXY" ]]||FIRST_PROXY="$now"; read -r state source schema <<EOF
$(python - "$base-status.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x.get('state','INVALID'),x.get('source_commit','INVALID'),x.get('schema','INVALID'))
PY
)
EOF
if [[ "$source" != "$VELA_SOURCE_COMMIT" || "$schema" != vela-runpod-capacity-preflight-proxy:v1 ]]; then REASON="PROXY_IDENTITY_MISMATCH"; break; fi; if [[ "$state" == FAILED ]]; then cp "$base-status.json" "$EVIDENCE_DIR/terminal-status.json"; REASON="SUPERVISOR_REPORTED_FAILED"; break; fi; if [[ "$state" == COMPLETE ]]; then cp "$base-status.json" "$EVIDENCE_DIR/terminal-status.json"; rcode="$(curl -sS -L --connect-timeout 3 --max-time 15 -o "$EVIDENCE_DIR/capacity_preflight.json.tmp" -w '%{http_code}' "$PROXY_BASE/v1/result/$RESULT_TOKEN" 2> "$EVIDENCE_DIR/result-fetch.err"||true)"; if [[ "$rcode" == 200 ]]; then FOUND=1; REASON="RESULT_FETCHED_VIA_RUNPOD_HTTPS_PROXY"; fi; break; fi; fi
if [[ -n "$FIRST_RUNNING" && -z "$FIRST_PROXY" ]]&&(( now-FIRST_RUNNING>=PROXY_STARTUP_GRACE_SECONDS )); then REASON="PROXY_UNREACHABLE_AFTER_RUNNING_GRACE"; break; fi; sleep "$POLL_INTERVAL_SECONDS"; done
[[ -n "$REASON" ]]||REASON="CONTROLLER_MAX_RUNTIME_EXPIRED"; cleanup_pod; [[ "$DELETED" == 1 ]]||{ echo "POD_DELETE_FAILED" >&2; exit 79; }; trap - EXIT INT TERM; echo "$REASON" > "$EVIDENCE_DIR/terminal-reason.txt"
if [[ "$FOUND" != 1 ]]; then echo "CAPACITY_PREFLIGHT_FAILED reason=$REASON" >&2; exit 78; fi
python - "$EVIDENCE_DIR/terminal-status.json" "$EVIDENCE_DIR/capacity_preflight.json.tmp" "$VELA_SOURCE_COMMIT" <<'PY'
import hashlib,json,sys
s=json.load(open(sys.argv[1])); raw=open(sys.argv[2],'rb').read(); r=json.loads(raw); src=sys.argv[3]
assert len(raw)==int(s['result_bytes']) and hashlib.sha256(raw).hexdigest()==s['result_sha256']
assert r['status']=='PASS' and r['scientific_evidence'] is False and r['purpose']=='P-R2-03_RESULT_BLIND_CAPACITY_PREFLIGHT' and r['source_commit']==src
assert r['matched_profile']=='g1i-20260805'; assert r['runtime']['device_capability']==[8,6] and 'A40' in r['runtime']['device']; m={x['scale']:x for x in r['models']}; assert set(m)=={'2.9B','7.2B'}
assert m['2.9B']['actual_sha256']=='ac1ae23d0e65c1d35ba523eacd81a2a4dacb7b886479909bbff34f312e766320' and m['2.9B']['actual_size']==5896273469 and m['2.9B']['state_restore_tensor_exact'] is True
assert m['7.2B']['actual_sha256']=='0d09d8961448032501c4d432c33a224c66356d43c10174386ea86b0da2b127d8' and m['7.2B']['actual_size']==14400007869 and m['7.2B']['state_restore_tensor_exact'] is True
PY
mv "$EVIDENCE_DIR/capacity_preflight.json.tmp" "$EVIDENCE_DIR/capacity_preflight.json"; END="$(date +%s)"; ELAPSED=$((END-RENT_START_EPOCH)); ((ELAPSED>MAX_RUNTIME_SECONDS))&&ELAPSED=$MAX_RUNTIME_SECONDS; python - "$EVIDENCE_DIR" "$ELAPSED" "$GPU_PRICE_CAP" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); e=int(sys.argv[2]); price=float(sys.argv[3]); r=json.load(open(p/'capacity_preflight.json')); x={'status':'PASS','scientific_evidence':False,'provider':'RUNPOD','phase':'RESULT_BLIND_CAPACITY_PREFLIGHT','pod_deleted':True,'elapsed_seconds':e,'max_runtime_seconds':900,'observed_secure_price_per_hr_usd':price,'estimated_gpu_cost_upper_bound_usd':round(e*price/3600,6),'transport':'RUNPOD_HTTPS_PROXY_V1','transport_terminal_reason':'RESULT_FETCHED_VIA_RUNPOD_HTTPS_PROXY','matched_profile':r['matched_profile'],'models':r['models']}; json.dump(x,open(p/'preflight_envelope.json','w'),indent=2,sort_keys=True)
PY
