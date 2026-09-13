#!/usr/bin/env python3
import json
from pathlib import Path
r=Path(__file__).resolve().parent
s=json.loads((r/'RUNPOD_RUNNER_SPEC.json').read_text())
assert s['schema']=='vela-runpod-runner-spec:v0'
assert s['provider']=='RUNPOD' and s['mode']=='POD'
assert s['paidExecutionAuthorized'] is False
assert s['automaticRetry'] is False and s['silentFallbackProviders']==[]
assert s['scientificGate']['enabled'] is False
assert s['model']['file']=='rwkv7-g1a-2.9b-20250924-ctx4096.pth'
assert s['model']['size']==5896274949
assert s['model']['sha256']=='df5716263b617e7da83590446fb2a98b6663447cfd52e8e9b49e11ce7f4faa3e'
p=(r/'runpod_preflight.py').read_text()
for forbidden in ['MODEL_STIMULI_INDEX','EVALUATOR.json','EXTERNAL_TRACE.json','VELA_CONDITION']:
    assert forbidden not in p, forbidden
l=(r/'runpod_launch.sh').read_text()
assert 'RUNPOD_API_KEY' in l and 'VELA_RUNPOD_PAID_EXECUTION_AUTHORIZED' in l and 'exit 77' in l and '--terminate-after' in l
print(json.dumps({'status':'PASS','provider':'RUNPOD','prep_only':True,'scientific_gate_enabled':False,'paid_execution_authorized':False},sort_keys=True))
