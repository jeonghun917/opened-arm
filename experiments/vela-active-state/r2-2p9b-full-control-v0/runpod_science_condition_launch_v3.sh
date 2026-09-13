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
POST_RESULT_HOLD_SECONDS=45
POLL_INTERVAL_SECONDS=3
MAX_CHUNK_CHARS=3500
EVIDENCE_DIR="${VELA_SCIENCE_EVIDENCE_DIR:-/tmp/runpod-science-${VELA_CONDITION}}"
TRANSPORT_PARSER="${VELA_SCIENCE_TRANSPORT_PARSER:-$(dirname "$0")/runpod_science_transport_v2.py}"
CAPACITY_SELECTOR="${VELA_RUNPOD_CAPACITY_SELECTOR:-$(dirname "$0")/runpod_select_a40_capacity_v1.py}"

mkdir -p "$EVIDENCE_DIR/polls"
START_EPOCH="$(date +%s)"
POD_ID=""
DELETED=0
WATCHDOG_PID=""

cleanup_pod() {
  set +e
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
    wait "$WATCHDOG_PID" >/dev/null 2>&1 || true
    WATCHDOG_PID=""
  fi
  if [[ -n "$POD_ID" && "$DELETED" != "1" ]]; then
    runpodctl pod get "$POD_ID" --include-machine > "$EVIDENCE_DIR/pod-final-get.json" 2> "$EVIDENCE_DIR/pod-final-get.err" || true
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
    else:
        raise SystemExit('linux/amd64 manifest missing')
else:
    print('sha256:'+hashlib.sha256(raw).hexdigest())
PY
)"
IMAGE_REF="python@${IMAGE_DIGEST}"
printf '%s\n' "$IMAGE_REF" > "$EVIDENCE_DIR/image-ref.txt"
BASE_RAW="https://raw.githubusercontent.com/jeonghun917/opened-arm/${VELA_SOURCE_COMMIT}/experiments/vela-active-state/r2-2p9b-full-control-v0"

BOOTSTRAP="$(cat <<'SH'
#!/bin/sh
set -u
mkdir -p /workspace/vela/stimuli
cd /workspace/vela

emit_result() {
python - <<'PY'
import base64,hashlib,json,os
from pathlib import Path
c=os.environ['VELA_CONDITION']; p=Path(f'condition_result_{c}.json')
if not p.exists():
    raw=json.dumps({'status':'TRANSPORT_NO_CONDITION_RESULT','scientific_evidence':False,'condition':c},separators=(',',':')).encode()
else:
    raw=p.read_bytes()
chunk=int(os.environ['VELA_MAX_CHUNK_CHARS'])
b64=base64.b64encode(raw).decode(); chunks=[b64[i:i+chunk] for i in range(0,len(b64),chunk)]
meta={'condition':c,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'chunks':len(chunks)}
print('VELA_CONDITION_RESULT_META='+json.dumps(meta,separators=(',',':'),sort_keys=True),flush=True)
for i,x in enumerate(chunks,1): print(f'VELA_CONDITION_RESULT_CHUNK={i:04d}/{len(chunks):04d}:'+x,flush=True)
print('VELA_SCIENCE_RESULT_READY='+json.dumps({'condition':c,'sha256':meta['sha256'],'chunks':meta['chunks'],'scientific_evidence':bool(json.loads(raw).get('scientific_evidence')) if raw else False},separators=(',',':'),sort_keys=True),flush=True)
PY
}

if [ -f ".vela-science-complete-${VELA_CONDITION}" ] && [ -f "condition_result_${VELA_CONDITION}.json" ]; then
  emit_result
  echo "VELA_SCIENCE_BOOTSTRAP_EXIT=0"
  sleep "$VELA_POST_RESULT_HOLD_SECONDS"
  exit 0
fi

rc=0
python -m pip install --quiet --extra-index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1+cu126 rwkv==0.8.32 tokenizers==0.21.4 numpy==2.4.6 huggingface_hub==0.36.0 || rc=$?
if [ "$rc" -eq 0 ]; then
python - <<'PY' || rc=$?
import os, urllib.request
base=os.environ['VELA_BASE_RAW']
files=['runpod_condition_adapter_v1.py','condition_runner.py','science_common.py','EVALUATOR.json','EXTERNAL_TRACE.json','MODEL_STIMULI_INDEX.json']+[f'stimuli/episode_{i}.json' for i in range(1,7)]
for rel in files:
    url=base+'/'+rel
    os.makedirs(os.path.dirname(rel) or '.',exist_ok=True)
    urllib.request.urlretrieve(url,rel)
PY
fi
if [ "$rc" -eq 0 ]; then
  VELA_CONDITION="$VELA_CONDITION" VELA_SOURCE_COMMIT="$VELA_SOURCE_COMMIT" python runpod_condition_adapter_v1.py || rc=$?
fi

emit_result
if [ "$rc" -eq 0 ] && [ -f "condition_result_${VELA_CONDITION}.json" ]; then
  touch ".vela-science-complete-${VELA_CONDITION}"
fi
echo "VELA_SCIENCE_BOOTSTRAP_EXIT=$rc"
sleep "$VELA_POST_RESULT_HOLD_SECONDS"
exit "$rc"
SH
)"

BOOTSTRAP_B64="$(printf '%s' "$BOOTSTRAP" | base64 -w0)"
ENV_JSON="$(python - "$BOOTSTRAP_B64" "$VELA_SOURCE_COMMIT" "$BASE_RAW" "$VELA_CONDITION" "$POST_RESULT_HOLD_SECONDS" "$MAX_CHUNK_CHARS" <<'PY'
import json,sys
b64,source,base,condition,hold,chunk=sys.argv[1:]
print(json.dumps({'VELA_BOOTSTRAP_B64':b64,'VELA_SOURCE_COMMIT':source,'VELA_BASE_RAW':base,'VELA_CONDITION':condition,'VELA_POST_RESULT_HOLD_SECONDS':hold,'VELA_MAX_CHUNK_CHARS':chunk,'PYTHONUNBUFFERED':'1'},separators=(',',':')))
PY
)"
DOCKER_ARGS="sh -lc 'echo \"\$VELA_BOOTSTRAP_B64\" | base64 -d | timeout ${WORKLOAD_TIMEOUT_SECONDS}s sh'"
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

(
  sleep "$MAX_RUNTIME_SECONDS"
  runpodctl pod delete "$POD_ID" > "$EVIDENCE_DIR/watchdog-delete.json" 2> "$EVIDENCE_DIR/watchdog-delete.err" || true
) &
WATCHDOG_PID=$!

ACCUM="$EVIDENCE_DIR/pod-logs-accumulated.jsonl"
: > "$ACCUM"
DEADLINE=$((START_EPOCH + MAX_RUNTIME_SECONDS))
FOUND=0
POLL=0
TERMINAL_REASON=""
while (( $(date +%s) < DEADLINE )); do
  POLL=$((POLL + 1))
  poll_base="$EVIDENCE_DIR/polls/poll-$(printf '%04d' "$POLL")"
  runpodctl pod get "$POD_ID" --include-machine > "${poll_base}-get.json" 2> "${poll_base}-get.err" || true
  runpodctl pod logs "$POD_ID" --source container --tail 400 --max-wait 3s > "${poll_base}-logs.jsonl" 2> "${poll_base}-logs.err" || true
  if [[ -s "${poll_base}-logs.jsonl" ]]; then
    cat "${poll_base}-logs.jsonl" >> "$ACCUM"
    awk '!seen[$0]++' "$ACCUM" > "${ACCUM}.tmp"; mv "${ACCUM}.tmp" "$ACCUM"
  fi
  python "$TRANSPORT_PARSER" "$ACCUM" "$EVIDENCE_DIR/condition_result_${VELA_CONDITION}.json" "$VELA_CONDITION" "$VELA_SOURCE_COMMIT" > "$EVIDENCE_DIR/transport-status.json"
  read -r ready_seen exit_seen exit_code reconstructed validation_error <<EOF
$(python - "$EVIDENCE_DIR/transport-status.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print('1' if x.get('ready_seen') else '0','1' if x.get('bootstrap_exit_seen') else '0',x.get('bootstrap_exit') if x.get('bootstrap_exit') is not None else 'NA','1' if x.get('result_reconstructed') else '0','1' if x.get('validation_errors') else '0')
PY
)
EOF
  if [[ "$ready_seen" == "1" || "$exit_seen" == "1" ]]; then
    runpodctl pod logs "$POD_ID" --source container --tail 800 --max-wait 3s > "$EVIDENCE_DIR/result-ready-final-logs.jsonl" 2> "$EVIDENCE_DIR/result-ready-final-logs.err" || true
    if [[ -s "$EVIDENCE_DIR/result-ready-final-logs.jsonl" ]]; then
      cat "$EVIDENCE_DIR/result-ready-final-logs.jsonl" >> "$ACCUM"; awk '!seen[$0]++' "$ACCUM" > "${ACCUM}.tmp"; mv "${ACCUM}.tmp" "$ACCUM"
    fi
    cleanup_pod
    python "$TRANSPORT_PARSER" "$ACCUM" "$EVIDENCE_DIR/condition_result_${VELA_CONDITION}.json" "$VELA_CONDITION" "$VELA_SOURCE_COMMIT" > "$EVIDENCE_DIR/transport-status-final.json"
    read -r exit_code reconstructed validation_error <<EOF
$(python - "$EVIDENCE_DIR/transport-status-final.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x.get('bootstrap_exit') if x.get('bootstrap_exit') is not None else 'NA','1' if x.get('result_reconstructed') else '0','1' if x.get('validation_errors') else '0')
PY
)
EOF
    if [[ "$validation_error" == "1" ]]; then TERMINAL_REASON="TRANSPORT_VALIDATION_ERROR"; break; fi
    if [[ "$exit_code" != "NA" && "$exit_code" != "0" ]]; then TERMINAL_REASON="BOOTSTRAP_EXIT_${exit_code}"; break; fi
    if [[ "$reconstructed" == "1" ]]; then FOUND=1; TERMINAL_REASON="RESULT_RECONSTRUCTED_AFTER_PROVIDER_DELETE"; break; fi
    TERMINAL_REASON="RESULT_READY_BUT_RECONSTRUCTION_FAILED_AFTER_PROVIDER_DELETE"; break
  fi
  sleep "$POLL_INTERVAL_SECONDS"
done

if [[ "$DELETED" != "1" ]]; then cleanup_pod; fi
if [[ "$DELETED" != "1" ]]; then echo "RUNPOD_POD_DELETE_FAILED condition=$VELA_CONDITION" >&2; exit 79; fi
trap - EXIT INT TERM
printf '%s\n' "$TERMINAL_REASON" > "$EVIDENCE_DIR/terminal-reason.txt"
if [[ "$FOUND" != "1" ]]; then echo "RUNPOD_SCIENCE_TRANSPORT_FAILED condition=$VELA_CONDITION reason=$TERMINAL_REASON" >&2; exit 78; fi

END_EPOCH="$(date +%s)"
python - "$EVIDENCE_DIR" "$VELA_CONDITION" "$VELA_SOURCE_COMMIT" "$START_EPOCH" "$END_EPOCH" "$GPU_PRICE_CAP" "$MAX_RUNTIME_SECONDS" "$TERMINAL_REASON" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); c,source=sys.argv[2],sys.argv[3]; start,end=int(sys.argv[4]),int(sys.argv[5]); price=float(sys.argv[6]); max_runtime=int(sys.argv[7]); terminal=sys.argv[8]
rp=p/f'condition_result_{c}.json'; raw=rp.read_bytes(); r=json.loads(raw); elapsed=max(0,end-start)
env={'status':'PASS' if r.get('status')=='P-R2-02_CONDITION_RESULT' and r.get('scientific_evidence') is True else 'FAIL','scientific_evidence':bool(r.get('scientific_evidence')),'provider':'RUNPOD','phase':'SCIENTIFIC_CONDITION','condition':c,'source_commit':source,'pod_deleted':True,'elapsed_seconds':elapsed,'max_runtime_seconds':max_runtime,'observed_secure_price_per_hr_usd':price,'estimated_gpu_cost_upper_bound_usd':round(elapsed*price/3600,6),'gpu':json.load(open(p/'selected-gpu.json')),'image_ref':(p/'image-ref.txt').read_text().strip(),'condition_result_sha256':hashlib.sha256(raw).hexdigest(),'transport_terminal_reason':terminal}
json.dump(env,open(p/f'condition_envelope_{c}.json','w'),indent=2,sort_keys=True)
print(json.dumps(env,separators=(',',':'),sort_keys=True))
if env['status']!='PASS': raise SystemExit(80)
PY
