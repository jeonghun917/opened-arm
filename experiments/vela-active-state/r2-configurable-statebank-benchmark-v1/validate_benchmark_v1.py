#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import benchmark_contract as contract
REQUIRED={"rwkv7-g1i-2.9b-20260805":("2.9B","ac1ae23d0e65c1d35ba523eacd81a2a4dacb7b886479909bbff34f312e766320"),"rwkv7-g1i-7.2b-20260805":("7.2B","0d09d8961448032501c4d432c33a224c66356d43c10174386ea86b0da2b127d8")}
FORBIDDEN=('bank_a','bank_b','release_readiness','database_migration','model_serving','data_pipeline','supply_chain','backup_recovery','DOMAIN_A_','DOMAIN_B_')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--execution-config',required=True); ap.add_argument('--problem-pack',required=True); ap.add_argument('--workflow'); a=ap.parse_args(); cfg,manifest,registry,problems,_root,tmp=contract.load_contract(Path(a.execution_config),Path(a.problem_pack),next(iter(REQUIRED)),'OFF')
 try:
  contract.validate_contract(cfg,problems); profiles={x['id']:x for x in registry['profiles']}
  for pid,(scale,sha) in REQUIRED.items():
   if pid not in profiles or profiles[pid].get('scale')!=scale or profiles[pid].get('sha256')!=sha: raise SystemExit(f'REQUIRED_MODEL_PROFILE_DRIFT:{pid}')
  for name in ('statebank_runner.py','benchmark_contract.py','rwkv_state_engine.py','benchmark_metrics.py'):
   text=Path(__file__).with_name(name).read_text()
   for token in FORBIDDEN:
    if token in text: raise SystemExit(f'RUNNER_PROBLEM_HARDCODE:{name}:{token}')
  if a.workflow:
   text=Path(a.workflow).read_text()
   for token in FORBIDDEN:
    if token in text: raise SystemExit(f'WORKFLOW_PROBLEM_HARDCODE:{token}')
   for token in ('execute_paid','model_profile_id','problem_pack','execution_config'):
    if token not in text: raise SystemExit(f'WORKFLOW_INPUT_MISSING:{token}')
  print(json.dumps({'status':'PASS','projects':len(problems),'families':len(set(p.get('family') for p in problems)),'required_models':sorted(REQUIRED),'routing_conditions':[x['id'] for x in cfg['routing_conditions']]},sort_keys=True))
 finally:
  if tmp: tmp.cleanup()
if __name__=='__main__': main()
