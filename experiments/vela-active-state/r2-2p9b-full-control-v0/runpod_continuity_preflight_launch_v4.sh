#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required}"
: "${VELA_SOURCE_COMMIT:?VELA_SOURCE_COMMIT is required}"
if [[ "${VELA_RUNPOD_CONTINUITY_V4_AUTHORIZED:-false}" != "true" ]]; then
  echo "RUNPOD_CONTINUITY_V4_BLOCKED: canonical authority not present" >&2
  exit 77
fi

GPU_ID="NVIDIA A100-SXM4-80GB"
GPU_PRICE_CAP="1.59"
MIN_MEMORY_GB=80
IMAGE_REF="python@sha256:6e13e65c55e33adf203d77ee371cf8bf5d81bd4902ef07565721f46bf44917af"
MAX_RUNTIME_SECONDS=900
WORKLOAD_TIMEOUT_SECONDS=600
POST_RESULT_HOLD_SECONDS=900
EVIDENCE_DIR="${VELA_CONTINUITY_V4_EVIDENCE_DIR:-/tmp/runpod-continuity-v4-evidence}"
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
python - "$EVIDENCE_DIR/gpu-list.json" "$EVIDENCE_DIR/selected-gpu.json" "$GPU_ID" "$GPU_PRICE_CAP" "$MIN_MEMORY_GB" <<'PY'
import json,sys
src,out,gpu_id,cap,min_mem=sys.argv[1:]
gpus=json.load(open(src)); cap=float(cap); min_mem=int(min_mem)
match=next((g for g in gpus if g.get('gpuId')==gpu_id),None)
if not match: raise SystemExit(f'GPU_NOT_LISTED: {gpu_id}')
price=match.get('securePricePerHr')
if match.get('secureCloud') is not True or match.get('available') is not True:
    raise SystemExit(f'GPU_NOT_AVAILABLE_SECURE: {match}')
if match.get('stockStatus') not in {'High','Medium'}:
    raise SystemExit(f'GPU_STOCK_NOT_SUFFICIENT: {match.get("stockStatus")}')
if not isinstance(price,(int,float)) or float(price)>cap:
    raise SystemExit(f'GPU_PRICE_CAP_EXCEEDED: observed={price} cap={cap}')
if int(match.get('memoryInGb') or 0)<min_mem:
    raise SystemExit(f'GPU_MEMORY_DRIFT: {match.get("memoryInGb")}')
json.dump({k:match.get(k) for k in ['gpuId','displayName','memoryInGb','secureCloud','securePricePerHr','stockStatus','dataCenterAvailability']},open(out,'w'),indent=2,sort_keys=True)
PY

printf '%s\n' "$IMAGE_REF" > "$EVIDENCE_DIR/image-ref.txt"
BASE_RAW="https://raw.githubusercontent.com/jeonghun917/opened-arm/${VELA_SOURCE_COMMIT}/experiments/vela-active-state/r2-2p9b-full-control-v0"
PREFLIGHT_URL="$BASE_RAW/runpod_continuity_preflight_v2.py"

BOOTSTRAP="$(cat <<'SH'
#!/bin/sh
set -u
mkdir -p /workspace/vela
cd /workspace/vela
rc=0
python -m pip install --quiet --extra-index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1+cu126 rwkv==0.8.32 tokenizers==0.21.4 numpy==2.4.6 huggingface_hub==0.36.0 || rc=$?
if [ "$rc" -ne 0 ]; then
  echo "VELA_CONTINUITY_V4_BOOTSTRAP_EXIT=$rc"
  exit "$rc"
fi
python - <<'PY' || rc=$?
import json, torch
name=torch.cuda.get_device_name(0)
cap=list(torch.cuda.get_device_capability(0))
arches=list(torch.cuda.get_arch_list())
report={"device":name,"device_capability":cap,"torch":torch.__version__,"cuda":torch.version.cuda,"torch_arch_list":arches}
print("VELA_CONTINUITY_V4_RUNTIME_IDENTITY="+json.dumps(report,separators=(",",":"),sort_keys=True),flush=True)
if cap != [8,0]:
    raise SystemExit(f"unexpected A100 capability {cap}")
if "sm_80" not in arches:
    raise SystemExit(f"torch wheel missing sm_80: {arches}")
if "A100" not in name:
    raise SystemExit(f"unexpected device name {name}")
PY
if [ "$rc" -ne 0 ]; then
  echo "VELA_CONTINUITY_V4_BOOTSTRAP_EXIT=$rc"
  exit "$rc"
fi
python - <<'PY' || rc=$?
import os, urllib.request
urllib.request.urlretrieve(os.environ['VELA_CONTINUITY_V4_URL'], 'runpod_continuity_preflight_v2.py')
PY
if [ "$rc" -ne 0 ]; then
  echo "VELA_CONTINUITY_V4_BOOTSTRAP_EXIT=$rc"
  exit "$rc"
fi
VELA_SOURCE_COMMIT="$VELA_SOURCE_COMMIT" VELA_CONTINUITY_PREFLIGHT_DIR=/workspace/vela python runpod_continuity_preflight_v2.py || rc=$?
echo "VELA_CONTINUITY_V4_BOOTSTRAP_EXIT=$rc"
if [ "$rc" -eq 0 ]; then sleep "$VELA_POST_RESULT_HOLD_SECONDS"; fi
exit "$rc"
SH
)"
BOOTSTRAP_B64="$(printf '%s' "$BOOTSTRAP" | base64 -w0)"
ENV_JSON="$(python - "$BOOTSTRAP_B64" "$VELA_SOURCE_COMMIT" "$PREFLIGHT_URL" "$POST_RESULT_HOLD_SECONDS" <<'PY'
import json,sys
b64,source,url,hold=sys.argv[1:]
print(json.dumps({'VELA_BOOTSTRAP_B64':b64,'VELA_SOURCE_COMMIT':source,'VELA_CONTINUITY_V4_URL':url,'VELA_POST_RESULT_HOLD_SECONDS':hold,'PYTHONUNBUFFERED':'1'},separators=(',',':')))
PY
)"
DOCKER_ARGS="sh -lc 'echo \"\$VELA_BOOTSTRAP_B64\" | base64 -d | timeout ${WORKLOAD_TIMEOUT_SECONDS}s sh'"

name="vela-p-r2-02-cont-v4-${VELA_SOURCE_COMMIT:0:12}"
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
  set +e
  python - "$EVIDENCE_DIR/pod-logs.jsonl" "$EVIDENCE_DIR/continuity-v4-result.json" "$EVIDENCE_DIR/bootstrap-exit.json" "$EVIDENCE_DIR/runtime-identity.json" <<'PY'
import json,sys
src,out,exitout,identityout=sys.argv[1:]; result=None; bootstrap_exit=None; identity=None
try: lines=open(src,encoding='utf-8').read().splitlines()
except FileNotFoundError: raise SystemExit(1)
for raw in lines:
  try: obj=json.loads(raw); line=obj.get('line','')
  except Exception: continue
  if line.startswith('VELA_CONTINUITY_V4_RUNTIME_IDENTITY='):
    identity=json.loads(line.split('=',1)[1])
  if line.startswith('VELA_CONTINUITY_PREFLIGHT_V2_SUMMARY='):
    result=json.loads(line.split('=',1)[1])
  if line.startswith('VELA_CONTINUITY_V4_BOOTSTRAP_EXIT='):
    try: bootstrap_exit=int(line.rsplit('=',1)[1])
    except ValueError: bootstrap_exit=255
if identity is not None:
  json.dump(identity,open(identityout,'w'),indent=2,sort_keys=True)
if result is not None:
  json.dump(result,open(out,'w'),indent=2,sort_keys=True); raise SystemExit(0)
if bootstrap_exit is not None:
  json.dump({'bootstrapExit':bootstrap_exit},open(exitout,'w'),indent=2,sort_keys=True); raise SystemExit(2)
raise SystemExit(1)
PY
  parse_rc=$?
  set -e
  if [[ "$parse_rc" == "0" ]]; then FOUND=1; break; fi
  if [[ "$parse_rc" == "2" ]]; then
    echo "RUNPOD_CONTINUITY_V4_BOOTSTRAP_FAILED" >&2
    exit 76
  fi
  sleep 5
done

runpodctl pod logs "$POD_ID" --source both --tail 5000 --max-wait 3s > "$EVIDENCE_DIR/pod-logs-final.jsonl" 2> "$EVIDENCE_DIR/pod-logs-final.err" || true
if [[ "$FOUND" != "1" ]]; then
  echo "RUNPOD_CONTINUITY_V4_RESULT_TIMEOUT" >&2
  exit 78
fi

python - "$EVIDENCE_DIR/continuity-v4-result.json" "$EVIDENCE_DIR/runtime-identity.json" "$VELA_SOURCE_COMMIT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); ident=json.load(open(sys.argv[2])); source=sys.argv[3]
assert ident['device_capability']==[8,0], ident
assert 'sm_80' in ident['torch_arch_list'], ident
assert 'A100' in ident['device'], ident
assert ident['torch']=='2.7.1+cu126', ident
assert r.get('status')=='PASS', r
assert r.get('scientific_evidence') is False, r
assert r.get('source_commit')==source, r
assert 'A100' in (r.get('device') or ''), r
assert r.get('torch')=='2.7.1+cu126', r
assert r.get('model_sha256')=='df5716263b617e7da83590446fb2a98b6663447cfd52e8e9b49e11ce7f4faa3e', r
for k in ['serialization_exact','metadata_equal','original_repeat_exact','restored_repeat_exact','cross_first_exact','cross_second_exact']:
    assert r.get(k) is True, (k,r)
assert max(float(x) for x in (r.get('max_abs') or {}).values())==0.0, r
PY

runpodctl pod get "$POD_ID" --include-machine > "$EVIDENCE_DIR/pod-final-before-delete.json" || true
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
elapsed=max(0,end-start); result=json.load(open(p/'continuity-v4-result.json')); identity=json.load(open(p/'runtime-identity.json'))
env={
 'status':result.get('status'),'classification':result.get('classification'),
 'scientific_evidence':False,'provider':'RUNPOD','phase':'CONTINUITY_PREFLIGHT_V4_ONLY',
 'pod_deleted':True,'elapsed_seconds':elapsed,'max_runtime_seconds':max_runtime,
 'observed_secure_price_per_hr_usd':price,
 'estimated_gpu_cost_upper_bound_usd':round(elapsed*price/3600,6),
 'gpu':json.load(open(p/'selected-gpu.json')),
 'runtime_identity':identity,
 'result':result,'image_ref':(p/'image-ref.txt').read_text().strip()
}
json.dump(env,open(p/'continuity-v4-envelope.json','w'),indent=2,sort_keys=True)
print(json.dumps(env,separators=(',',':'),sort_keys=True))
PY
