#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required}"
: "${VELA_SOURCE_COMMIT:?VELA_SOURCE_COMMIT is required}"
if [[ "${VELA_RUNPOD_PREFLIGHT_AUTHORIZED:-false}" != "true" ]]; then
  echo "RUNPOD_PREFLIGHT_BLOCKED: canonical preflight authority not present" >&2
  exit 77
fi

GPU_ID="NVIDIA A40"
GPU_PRICE_CAP="0.49"
IMAGE_TAG="python:3.12.13-slim-bookworm"
MAX_RUNTIME_SECONDS=1800
WORKLOAD_TIMEOUT_SECONDS=1500
EVIDENCE_DIR="${VELA_EVIDENCE_DIR:-/tmp/runpod-preflight-evidence}"
mkdir -p "$EVIDENCE_DIR"
START_EPOCH="$(date +%s)"
POD_ID=""
DELETED=0

cleanup_pod() {
  set +e
  if [[ -n "$POD_ID" && "$DELETED" != "1" ]]; then
    runpodctl pod get "$POD_ID" --include-machine > "$EVIDENCE_DIR/pod-final-get.json" 2> "$EVIDENCE_DIR/pod-final-get.err" || true
    runpodctl pod delete "$POD_ID" > "$EVIDENCE_DIR/pod-delete.json" 2> "$EVIDENCE_DIR/pod-delete.err"
    rc=$?
    if [[ "$rc" == "0" ]]; then DELETED=1; fi
  fi
  set -e
}
trap cleanup_pod EXIT INT TERM

runpodctl version > "$EVIDENCE_DIR/runpodctl-version.txt"
runpodctl user >/dev/null
runpodctl gpu list --include-unavailable > "$EVIDENCE_DIR/gpu-list.json"
python - "$EVIDENCE_DIR/gpu-list.json" "$EVIDENCE_DIR/selected-gpu.json" "$GPU_ID" "$GPU_PRICE_CAP" <<'PY'
import json, sys
src,out,gpu_id,cap=sys.argv[1:]
gpus=json.load(open(src))
cap=float(cap)
match=next((g for g in gpus if g.get('gpuId')==gpu_id),None)
if not match: raise SystemExit(f'GPU_NOT_LISTED: {gpu_id}')
price=match.get('securePricePerHr')
if match.get('secureCloud') is not True or match.get('available') is not True:
    raise SystemExit(f'GPU_NOT_AVAILABLE_SECURE: {match}')
if not isinstance(price,(int,float)) or float(price)>cap:
    raise SystemExit(f'GPU_PRICE_CAP_EXCEEDED: observed={price} cap={cap}')
if int(match.get('memoryInGb') or 0) < 48:
    raise SystemExit(f'GPU_MEMORY_DRIFT: {match.get("memoryInGb")}')
json.dump({k:match.get(k) for k in ['gpuId','displayName','memoryInGb','secureCloud','securePricePerHr','stockStatus','dataCenterAvailability']},open(out,'w'),indent=2,sort_keys=True)
PY

# Resolve the linux/amd64 image manifest once, then create the pod by immutable digest.
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
else:
    print('sha256:'+hashlib.sha256(raw).hexdigest())
PY
)"
IMAGE_REF="python@${IMAGE_DIGEST}"
printf '%s\n' "$IMAGE_REF" > "$EVIDENCE_DIR/image-ref.txt"

BASE_RAW="https://raw.githubusercontent.com/jeonghun917/opened-arm/${VELA_SOURCE_COMMIT}/experiments/vela-active-state/r2-2p9b-full-control-v0"
GPU_PREFLIGHT_URL="$BASE_RAW/gpu_preflight.py"
ROUNDTRIP_URL="$BASE_RAW/runpod_state_roundtrip_v1.py"

BOOTSTRAP="$(cat <<'SH'
#!/bin/sh
set -u
mkdir -p /workspace/vela
cd /workspace/vela
python - <<'PY'
import os, urllib.request
for env,dst in [('VELA_GPU_PREFLIGHT_URL','gpu_preflight.py'),('VELA_ROUNDTRIP_URL','runpod_state_roundtrip_v1.py')]:
    urllib.request.urlretrieve(os.environ[env], dst)
PY
rc=0
python gpu_preflight.py || rc=$?
if [ -f gpu_preflight.json ]; then
  python - <<'PY'
import json
r=json.load(open('gpu_preflight.json'))
print('VELA_GPU_PREFLIGHT_JSON='+json.dumps(r,separators=(',',':'),sort_keys=True),flush=True)
PY
else
  echo 'VELA_GPU_PREFLIGHT_JSON={"status":"NO_REPORT","scientific_evidence":false}'
fi
if [ "$rc" -eq 0 ]; then
  VELA_SOURCE_COMMIT="$VELA_SOURCE_COMMIT" VELA_ROUNDTRIP_DIR=/workspace/vela python runpod_state_roundtrip_v1.py || rc=$?
fi
echo "VELA_BOOTSTRAP_EXIT=$rc"
sleep 60
exit "$rc"
SH
)"
BOOTSTRAP_B64="$(printf '%s' "$BOOTSTRAP" | base64 -w0)"
ENV_JSON="$(python - "$BOOTSTRAP_B64" "$VELA_SOURCE_COMMIT" "$GPU_PREFLIGHT_URL" "$ROUNDTRIP_URL" <<'PY'
import json,sys
b64,source,gpu_url,rt_url=sys.argv[1:]
print(json.dumps({'VELA_BOOTSTRAP_B64':b64,'VELA_SOURCE_COMMIT':source,'VELA_GPU_PREFLIGHT_URL':gpu_url,'VELA_ROUNDTRIP_URL':rt_url,'PYTHONUNBUFFERED':'1'},separators=(',',':')))
PY
)"
DOCKER_ARGS="sh -lc 'echo \"\$VELA_BOOTSTRAP_B64\" | base64 -d | timeout ${WORKLOAD_TIMEOUT_SECONDS}s sh'"

name="vela-p-r2-02-preflight-${VELA_SOURCE_COMMIT:0:12}"
runpodctl pod create \
  --name "$name" \
  --image "$IMAGE_REF" \
  --gpu-id "$GPU_ID" \
  --gpu-count 1 \
  --cloud-type SECURE \
  --container-disk-in-gb 20 \
  --ssh=false \
  --min-cuda-version 12.6 \
  --env "$ENV_JSON" \
  --docker-args "$DOCKER_ARGS" \
  > "$EVIDENCE_DIR/pod-create.json"
POD_ID="$(python - "$EVIDENCE_DIR/pod-create.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); v=x.get('id') or x.get('podId')
if not v: raise SystemExit(f'pod id missing: {x}')
print(v)
PY
)"
printf '%s\n' "$POD_ID" > "$EVIDENCE_DIR/pod-id.txt"

DEADLINE=$((START_EPOCH + WORKLOAD_TIMEOUT_SECONDS))
FOUND=0
while (( $(date +%s) < DEADLINE )); do
  runpodctl pod get "$POD_ID" --include-machine > "$EVIDENCE_DIR/pod-get.json" 2> "$EVIDENCE_DIR/pod-get.err" || true
  runpodctl pod logs "$POD_ID" --source container --tail 5000 --max-wait 3s > "$EVIDENCE_DIR/pod-logs.jsonl" 2> "$EVIDENCE_DIR/pod-logs.err" || true
  if python - "$EVIDENCE_DIR/pod-logs.jsonl" "$EVIDENCE_DIR/gpu-preflight-result.json" "$EVIDENCE_DIR/state-roundtrip-result.json" <<'PY'
import json,sys
src,gout,rout=sys.argv[1:]
gpu=rt=None
try:
  lines=open(src,encoding='utf-8').read().splitlines()
except FileNotFoundError:
  raise SystemExit(1)
for raw in lines:
  try: obj=json.loads(raw); line=obj.get('line','')
  except Exception: continue
  if line.startswith('VELA_GPU_PREFLIGHT_JSON='):
    gpu=json.loads(line.split('=',1)[1])
  elif line.startswith('VELA_STATE_ROUNDTRIP_JSON='):
    rt=json.loads(line.split('=',1)[1])
if gpu is not None: json.dump(gpu,open(gout,'w'),indent=2,sort_keys=True)
if rt is not None: json.dump(rt,open(rout,'w'),indent=2,sort_keys=True)
raise SystemExit(0 if gpu is not None and rt is not None else 1)
PY
  then FOUND=1; break; fi
  sleep 5
done

runpodctl pod logs "$POD_ID" --source both --tail 5000 --max-wait 3s > "$EVIDENCE_DIR/pod-logs-final.jsonl" 2> "$EVIDENCE_DIR/pod-logs-final.err" || true
if [[ "$FOUND" != "1" ]]; then
  echo "RUNPOD_PREFLIGHT_RESULT_TIMEOUT" >&2
  exit 78
fi

python - "$EVIDENCE_DIR/gpu-preflight-result.json" "$EVIDENCE_DIR/state-roundtrip-result.json" "$VELA_SOURCE_COMMIT" <<'PY'
import json,sys
gpu=json.load(open(sys.argv[1])); rt=json.load(open(sys.argv[2])); source=sys.argv[3]
assert gpu.get('status')=='PASS', gpu
assert gpu.get('scientific_evidence') is False, gpu
assert (gpu.get('runtime') or {}).get('torch')=='2.7.1+cu126', gpu
assert (gpu.get('runtime') or {}).get('rwkv')=='0.8.32', gpu
assert (gpu.get('runtime') or {}).get('tokenizers')=='0.21.4', gpu
assert (gpu.get('runtime') or {}).get('numpy')=='2.4.6', gpu
assert 'A40' in ((gpu.get('runtime') or {}).get('device') or ''), gpu
m=next(x for x in gpu.get('models',[]) if x.get('scale')=='2.9B')
assert m.get('actual_size')==5896274949, m
assert m.get('actual_sha256')=='df5716263b617e7da83590446fb2a98b6663447cfd52e8e9b49e11ce7f4faa3e', m
assert rt.get('status')=='PASS', rt
assert rt.get('scientific_evidence') is False, rt
assert rt.get('source_commit')==source, rt
s=rt.get('state') or {}
assert s.get('serialize_restore_exact') is True, rt
assert s.get('continuation_logits_exact') is True, rt
assert s.get('continuation_state_exact') is True, rt
PY

# Capture final provider identity before deletion, then delete exactly once.
runpodctl pod get "$POD_ID" --include-machine > "$EVIDENCE_DIR/pod-final-before-delete.json"
cleanup_pod
if [[ "$DELETED" != "1" ]]; then
  echo "RUNPOD_POD_DELETE_FAILED" >&2
  exit 79
fi
trap - EXIT INT TERM
END_EPOCH="$(date +%s)"
python - "$EVIDENCE_DIR" "$START_EPOCH" "$END_EPOCH" "$GPU_PRICE_CAP" "$MAX_RUNTIME_SECONDS" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); start,end=int(sys.argv[2]),int(sys.argv[3]); price=float(sys.argv[4]); max_runtime=int(sys.argv[5])
elapsed=max(0,end-start)
env={
 'status':'PASS','scientific_evidence':False,'provider':'RUNPOD','phase':'PREFLIGHT_ONLY',
 'pod_deleted':True,'elapsed_seconds':elapsed,'max_runtime_seconds':max_runtime,
 'observed_secure_price_per_hr_usd':price,
 'estimated_gpu_cost_upper_bound_usd':round(elapsed*price/3600,6),
 'gpu':json.load(open(p/'selected-gpu.json')),
 'gpu_preflight':json.load(open(p/'gpu-preflight-result.json')),
 'state_roundtrip':json.load(open(p/'state-roundtrip-result.json')),
 'image_ref':(p/'image-ref.txt').read_text().strip()
}
json.dump(env,open(p/'preflight-envelope.json','w'),indent=2,sort_keys=True)
print(json.dumps(env,separators=(',',':'),sort_keys=True))
PY
