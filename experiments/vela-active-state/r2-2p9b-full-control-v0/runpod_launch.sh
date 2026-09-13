#!/usr/bin/env bash
set -euo pipefail
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required}"
: "${VELA_SOURCE_COMMIT:?VELA_SOURCE_COMMIT is required}"
: "${VELA_RUNPOD_GPU_ID:?VELA_RUNPOD_GPU_ID is required}"
: "${VELA_RUNPOD_IMAGE:?VELA_RUNPOD_IMAGE is required}"
if [[ "${VELA_RUNPOD_PAID_EXECUTION_AUTHORIZED:-false}" != "true" ]]; then
  echo "RUNPOD_EXECUTION_BLOCKED: canonical paid execution authority not present" >&2
  exit 77
fi
runpodctl update >/dev/null
runpodctl version >&2
runpodctl pod create --help >&2
name="vela-p-r2-02-preflight-${VELA_SOURCE_COMMIT:0:12}"
runpodctl pod create --name "$name" --image "$VELA_RUNPOD_IMAGE" --gpu-id "$VELA_RUNPOD_GPU_ID" --terminate-after "${VELA_RUNPOD_TERMINATE_AFTER:-3600}" > runpod-create.json
python - <<'PY'
import json
x=json.load(open('runpod-create.json')); pod=x.get('id') or x.get('podId')
if not pod: raise SystemExit(f'pod id missing: {x}')
open('runpod-pod-id.txt','w').write(str(pod)+'\n')
PY
