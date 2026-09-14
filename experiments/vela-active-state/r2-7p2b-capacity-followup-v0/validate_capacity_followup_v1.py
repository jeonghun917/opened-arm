#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXPECTED_BLOBS={"MODEL_STIMULI_INDEX.json":"95b2371ac7bb9416ad5b9a7d33f2df4de40d4d4c","EVALUATOR.json":"55d28e6564396408f293f67ebabfc258e41ae9ef","EXTERNAL_TRACE.json":"fb056e547feaedfead6b2bb67f87e31449d21466","PROTOCOL.json":"3dbf20fd96082221c90e348d15d943d5213ec8d1","condition_runner.py":"70b696bcfdb3ba7fc7f4010a1143aac1063c452b","runpod_condition_adapter_v1.py":"fd5145246a519ae45992df21b6c3acdf7ede5273","runpod_science_proxy_supervisor_v1.py":"d005b8eb0652859238c23ed240ab1c469f53d751","runpod_select_a40_capacity_v1.py":"56a92af4fb52fb33f85cb8d7dcaabcaaf3d6d5d8","install_runpodctl_pinned_v1.sh":"9eb674a5f0e6ca7e2a7cd8dc5b98db19ec22a901","test_runpod_science_proxy_v1.py":"ceff7e7d8ee90d80eee8cde2cfe0613adaade557","stimuli/episode_1.json":"9b91ed85c5a3f0e1495fbcda627b660857743235","stimuli/episode_2.json":"22e590dd8318ac64cc12bd3d41fc765e6ce7db46","stimuli/episode_3.json":"426055ca45b40476e0baa9917c72a7d1aba0332f","stimuli/episode_4.json":"fb232b0b6683276d8d323cc306071d7e5cf83f55","stimuli/episode_5.json":"8c00d81963e25f998348aac88ebdf92d45077e95","stimuli/episode_6.json":"8091870b4f8ac512ac835d9db48f16a80033ad15"}
def gitblob(raw:bytes)->str:return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def main():
  for rel,expected in EXPECTED_BLOBS.items():
    actual=gitblob((HERE/rel).read_bytes())
    if actual!=expected: raise RuntimeError(f"blob drift {rel}: {actual}")
  contract=json.loads((HERE/'CAPACITY_FOLLOWUP_CONTRACT_V1.json').read_text()); auth=json.loads((HERE/'RUNPOD_SCIENCE_MATRIX_AUTH_V1.json').read_text()); pauth=json.loads((HERE/'RUNPOD_CAPACITY_PREFLIGHT_AUTH_V1.json').read_text())
  assert contract['schema']=='vela-p-r2-03-capacity-followup:v1' and contract['status']=='PREPARED_UNARMED' and contract['scientificExecutionAuthorized'] is False
  assert contract['design']['conditions']==['B','C','G','M','X'] and contract['design']['episodes']==6
  assert contract['design']['classifierInheritedExactlyFromAggregateBlob']=='ac86908be386b0159f52bcf45d4dd0e0786daae4'
  models={x['scale']:x for x in contract['models']}; assert set(models)=={'2.9B','7.2B'}
  assert models['2.9B']['revision']==models['7.2B']['revision']=='ede85bf8ab2e59aff7d7ca909fbbc73317866d89'
  assert models['2.9B']['sha256']=='ac1ae23d0e65c1d35ba523eacd81a2a4dacb7b886479909bbff34f312e766320' and models['2.9B']['sizeBytes']==5896273469
  assert models['7.2B']['sha256']=='0d09d8961448032501c4d432c33a224c66356d43c10174386ea86b0da2b127d8' and models['7.2B']['sizeBytes']==14400007869
  orch=contract['orchestration']; assert orch['transport']=='RUNPOD_HTTPS_PROXY_V1' and orch['matrixMaxParallel']==1 and orch['automaticRetry'] is False and orch['fallbackProviders']==[] and orch['containerDiskGb']==50 and orch['maxRuntimeSecondsPerPod']==900 and orch['workloadTimeoutSeconds']==840
  assert contract['gates']['preflightStatus']=='NOT_RUN' and contract['gates']['scienceStatus']=='NOT_AUTHORIZED'
  assert pauth['status'] in {'PREPARED_UNARMED','AUTHORIZED_PREFLIGHT','CONSUMED_SUCCESS','CONSUMED_INCOMPLETE'} and pauth['maxPodCount']==1 and float(pauth['maxTotalUsd'])<=0.20 and pauth['automaticRetry'] is False and pauth['fallbackProviders']==[] and pauth['scientificEvidence'] is False
  if pauth['status']=='PREPARED_UNARMED': assert pauth['paidExecutionAuthorized'] is False
  assert auth['status'] in {'PREPARED_UNARMED','AUTHORIZED_MATRIX','CONSUMED_SUCCESS','CONSUMED_INCOMPLETE'}
  if auth['status']=='PREPARED_UNARMED': assert auth['scientificExecutionAuthorized'] is False and auth['preflightStatus']=='NOT_RUN' and auth['scientificResultExists'] is False
  if auth['status']=='AUTHORIZED_MATRIX': assert auth['scientificExecutionAuthorized'] is True and auth['preflightStatus']=='PASS' and auth['maxPodCount']==5 and float(auth['maxTotalUsd'])<=0.70 and auth['maxRuntimeSecondsPerPod']==900 and auth['automaticRetry'] is False and auth['fallbackProviders']==[]
  science=(HERE/'science_common.py').read_text(); agg=(HERE/'aggregate.py').read_text(); launcher=(HERE/'runpod_science_condition_launch_v5.py').read_text(); preflight_launcher=(HERE/'runpod_capacity_preflight_launch_v1.sh').read_text()
  for token in ['BASE_SCIENCE_COMMON_BLOB = "77cee99a8ff131b31165521fa8c9cfb76a7ce174"','"rwkv7-g1i-2.9b-20260805-ctx16384.pth"','"rwkv7-g1i-7.2b-20260805-ctx16384.pth"','"matched_model_profile"']: assert token in science,token
  for token in ['BASE_AGGREGATE_BLOB = "ac86908be386b0159f52bcf45d4dd0e0786daae4"','SCALES = ["2.9B", "7.2B"]','PRIMARY_SCALE = "7.2B"','"P-R2-03_RESULT"']: assert token in agg,token
  for token in ['BASE_LAUNCHER_BLOB = "8f5b721d5d1e91888b335ed6756b13f13687c697"',"'MAX_RUNTIME_SECONDS=900'","'WORKLOAD_TIMEOUT_SECONDS=840'","'--container-disk-in-gb 50'",'r2-7p2b-capacity-followup-v0']: assert token in launcher,token
  for token in ['sanitize_pod_get','VELA_RESULT_TOKEN','<redacted>','CONTAINER_RESTART_DETECTED_BY_CONTROL_PLANE','RESULT_FETCHED_VIA_RUNPOD_HTTPS_PROXY','MAX_RUNTIME_SECONDS=900','WORKLOAD_TIMEOUT_SECONDS=840']: assert token in preflight_launcher,token
  sha=os.environ.get('GITHUB_SHA','')
  if len(sha)==40:
    td=Path(tempfile.mkdtemp(prefix='vela-science-common-portable-'))
    try:
      shutil.copy2(HERE/'science_common.py',td/'science_common.py')
      env=os.environ.copy(); env['PYTHONPATH']=str(td); env['VELA_SOURCE_COMMIT']=sha
      code="import science_common as s; assert s.BASE_SCIENCE_COMMON_BLOB=='77cee99a8ff131b31165521fa8c9cfb76a7ce174'; assert [x['scale'] for x in s.MODELS]==['2.9B','7.2B']; assert s.MATCHED_PROFILE=='g1i-20260805'"
      subprocess.check_call([sys.executable,'-c',code],cwd=td,env=env)
    finally: shutil.rmtree(td)
  print('P_R2_03_CAPACITY_FOLLOWUP_STATIC_PASS')
if __name__=='__main__': main()
