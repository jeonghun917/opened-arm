#!/usr/bin/env python3
import argparse,json,re,tempfile
from pathlib import Path
import benchmark_contract as contract
import benchmark_metrics as metrics
import rwkv_state_engine as engine
REQUIRED={
 'rwkv7-g1i-2.9b-20260805':('2.9B','ac1ae23d0e65c1d35ba523eacd81a2a4dacb7b886479909bbff34f312e766320'),
 'rwkv7-g1i-7.2b-20260805':('7.2B','0d09d8961448032501c4d432c33a224c66356d43c10174386ea86b0da2b127d8')}
PROBLEM_LITERALS=('bank_a','bank_b','release_readiness','database_migration','model_serving','data_pipeline','supply_chain','backup_recovery')
def validate_measurement_boundary():
 initial={'engine_version':'16.4','pool_mode':'statement','migration_channel':'canary'}; final={'engine_version':'16.5','pool_mode':'statement','migration_channel':'canary'}; compact=lambda x:json.dumps(x,separators=(',',':'))
 if engine.extract_json('\n\nAnswer: '+compact(final))!=final:raise SystemExit('MEASUREMENT_DIRECT_JSON_REGRESSION')
 if engine.extract_json('Assistant: <think>quoted initial '+compact(initial)) is not None:raise SystemExit('MEASUREMENT_UNCLOSED_THINK_JSON_ACCEPTED')
 if engine.extract_json('Assistant: <think>quoted initial '+compact(initial)+'</think>\n\nAnswer: '+compact(final))!=final:raise SystemExit('MEASUREMENT_FINAL_JSON_AFTER_THINK_MISSED')
 if engine.extract_json('Assistant: <think>reasoning</think>\n\nAnswer: {"engine_version":"16') is not None:raise SystemExit('MEASUREMENT_INCOMPLETE_JSON_ACCEPTED')
 problem={'memory_contract':{'durable_memory_events':[]}};ep={'episode':1,'event':'DB_RETURN','domain':'database_runtime','probe':{'expected':final,'required_keys':list(final),'primary':True}};row=metrics.evaluate_probe(problem,ep,None,'truncated','MAX_NEW_TOKENS')
 if row['parse_ok'] or row['output_failure']!='NO_VALID_FINAL_JSON' or row['omissions'] or row['field_accuracy']!=0:raise SystemExit('MEASUREMENT_PARSE_FAILURE_CLASSIFICATION_DRIFT')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execution-config',required=True);ap.add_argument('--problem-pack',required=True);ap.add_argument('--workflow');a=ap.parse_args()
 validate_measurement_boundary();config_path=Path(a.execution_config);pack_path=Path(a.problem_pack)
 cfg,manifest,registry,problems,_root,tmp=contract.load_contract(config_path,pack_path,next(iter(REQUIRED)),'OFF')
 try:
  contract.validate_contract(cfg,problems);profiles={x['id']:x for x in registry['profiles']}
  for pid,(scale,sha) in REQUIRED.items():
   if pid not in profiles or profiles[pid].get('scale')!=scale or profiles[pid].get('sha256')!=sha:raise SystemExit(f'REQUIRED_MODEL_PROFILE_DRIFT:{pid}')
  try: direct=json.loads(pack_path.read_text()) if pack_path.is_file() else None
  except Exception: direct=None
  if isinstance(direct,dict) and direct.get('schema')=='vela-statebank-tabular-problem-source:v1':
   with tempfile.TemporaryDirectory(prefix='vela-extensionless-pack-test-') as td:
    shadow=Path(td)/'PACK_INPUT';shadow.write_bytes(pack_path.read_bytes())
    cfg2,manifest2,registry2,problems2,_root2,tmp2=contract.load_contract(config_path,shadow,next(iter(REQUIRED)),'OFF')
    try:
     contract.validate_contract(cfg2,problems2)
     if len(problems2)!=len(problems) or len(manifest2.get('projects',[]))!=len(manifest.get('projects',[])):raise SystemExit('EXTENSIONLESS_COMPACT_SOURCE_DRIFT')
     if registry2.get('schema')!=registry.get('schema'):raise SystemExit('EXTENSIONLESS_MODEL_REGISTRY_DRIFT')
    finally:
     if tmp2:tmp2.cleanup()
  for name in ('statebank_runner.py','benchmark_metrics.py','rwkv_state_engine.py'):
   text=Path(__file__).with_name(name).read_text()
   for token in PROBLEM_LITERALS:
    if token in text:raise SystemExit(f'RUNNER_PROBLEM_HARDCODE:{name}:{token}')
  runner=Path(__file__).with_name('statebank_runner.py').read_text()
  for token in ('primary_parse_failure_count','domain_metrics','zero_exact_domains'):
   if token not in runner:raise SystemExit(f'MEASUREMENT_REPORTING_MISSING:{token}')
  if a.workflow:
   text=Path(a.workflow).read_text()
   for token in PROBLEM_LITERALS:
    if token in text:raise SystemExit(f'WORKFLOW_PROBLEM_HARDCODE:{token}')
   for token in ('execute_paid','model_profile_id','problem_pack','execution_config'):
    if token not in text:raise SystemExit(f'WORKFLOW_INPUT_MISSING:{token}')
   for pid in REQUIRED:
    if pid not in text:raise SystemExit(f'WORKFLOW_MODEL_CHOICE_MISSING:{pid}')
   if re.search(r'matrix:\s*[\s\S]{0,300}rwkv7-g1i-2\.9b[\s\S]{0,150}rwkv7-g1i-7\.2b',text):raise SystemExit('WORKFLOW_FORCES_TWO_MODEL_MATRIX')
  print(json.dumps({'status':'PASS','measurement_boundary':'PASS','projects':len(problems),'families':len(set(p.get('family') for p in problems)),'required_models':sorted(REQUIRED),'routing_conditions':[x['id'] for x in cfg['routing_conditions']]},sort_keys=True))
 finally:
  if tmp:tmp.cleanup()
if __name__=='__main__':main()
