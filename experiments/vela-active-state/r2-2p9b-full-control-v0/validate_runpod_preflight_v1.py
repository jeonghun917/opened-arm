#!/usr/bin/env python3
import json
from pathlib import Path

r=Path(__file__).resolve().parent
s=json.loads((r/'RUNPOD_PREFLIGHT_SPEC_V1.json').read_text())
assert s['schema']=='vela-runpod-preflight-spec:v1'
assert s['provider']=='RUNPOD' and s['mode']=='POD' and s['phase']=='PREFLIGHT_ONLY'
assert s['preflightAuthorized'] is True
assert s['scientificExecutionAuthorized'] is False
assert s['automaticRetry'] is False and s['fallbackProviders']==[]
assert s['scientificGate']['enabled'] is False
assert s['target']['gpuId']=='NVIDIA A40'
assert s['target']['memoryInGb']==48
assert s['target']['cloud']=='SECURE'
assert s['target']['maxSecurePricePerHrUsd']==0.49
assert s['target']['containerImageTag']=='python:3.12.13-slim-bookworm'
assert s['target']['ssh'] is False
assert s['budget']=={'maxTotalUsd':0.50,'maxPodCount':1,'maxRuntimeSeconds':1800,'workloadTimeoutSeconds':1500}
assert s['frozenGpuPreflight']['gitBlobSha1']=='7f3801df39f0689fd4135730f4c15c780f9ebab9'
assert s['model']['size']==5896274949
assert s['model']['sha256']=='df5716263b617e7da83590446fb2a98b6663447cfd52e8e9b49e11ce7f4faa3e'

rt=(r/'runpod_state_roundtrip_v1.py').read_text()
for forbidden in ['MODEL_STIMULI_INDEX','EVALUATOR.json','EXTERNAL_TRACE.json','VELA_CONDITION','condition_runner.py','aggregate.py']:
    assert forbidden not in rt, forbidden
launch=(r/'runpod_preflight_launch_v1.sh').read_text()
for required in ['RUNPOD_API_KEY','VELA_RUNPOD_PREFLIGHT_AUTHORIZED','NVIDIA A40','GPU_PRICE_CAP="0.49"','python:3.12.13-slim-bookworm','--cloud-type SECURE','--ssh=false','--gpu-count 1','pod delete','WORKLOAD_TIMEOUT_SECONDS=1500','automaticRetry']:
    if required=='automaticRetry':
        continue
    assert required in launch, required
assert '--terminate-after' not in launch
assert 'NVIDIA GeForce RTX' not in launch
assert 'KAGGLE' not in launch and 'MODAL' not in launch and 'AWS_LAMBDA' not in launch
print(json.dumps({'status':'PASS','provider':'RUNPOD','phase':'PREFLIGHT_ONLY','gpu':'NVIDIA A40','scientific_gate_enabled':False,'automatic_retry':False},sort_keys=True))
