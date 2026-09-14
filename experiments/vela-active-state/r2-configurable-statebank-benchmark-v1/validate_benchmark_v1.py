#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import statebank_runner as runner

REQUIRED_MODELS={
    "rwkv7-g1i-2.9b-20260805": ("2.9B", "ac1ae23d0e65c1d35ba523eacd81a2a4dacb7b886479909bbff34f312e766320"),
    "rwkv7-g1i-7.2b-20260805": ("7.2B", "0d09d8961448032501c4d432c33a224c66356d43c10174386ea86b0da2b127d8"),
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--execution-config',required=True); ap.add_argument('--problem-pack',required=True); ap.add_argument('--workflow'); args=ap.parse_args()
    cfg, manifest, registry, problems, _root, tmp=runner.load_contract(Path(args.execution_config),Path(args.problem_pack),next(iter(REQUIRED_MODELS)),"OFF")
    try:
        runner.validate_data_driven_contract(cfg,problems)
        profiles={x['id']:x for x in registry['profiles']}
        for pid,(scale,sha) in REQUIRED_MODELS.items():
            if pid not in profiles or profiles[pid].get('scale')!=scale or profiles[pid].get('sha256')!=sha:
                raise SystemExit(f'REQUIRED_MODEL_PROFILE_DRIFT:{pid}')
        if len({x['id'] for x in cfg['routing_conditions']}) != len(cfg['routing_conditions']): raise SystemExit('DUPLICATE_CONDITION')
        if args.workflow:
            text=Path(args.workflow).read_text()
            for forbidden in ('bank_a','bank_b','release_readiness','database_migration','model_serving','data_pipeline','supply_chain','backup_recovery','DOMAIN_A_','DOMAIN_B_'):
                if forbidden in text: raise SystemExit(f'WORKFLOW_PROBLEM_HARDCODE:{forbidden}')
            if 'execute_paid' not in text or 'model_profile_id' not in text or 'problem_pack' not in text: raise SystemExit('WORKFLOW_REQUIRED_INPUT_MISSING')
        runner_text=Path(__file__).with_name('statebank_runner.py').read_text()
        for forbidden in ('bank_a','bank_b','release_readiness','database_migration','model_serving','data_pipeline','supply_chain','backup_recovery','DOMAIN_A_','DOMAIN_B_'):
            if forbidden in runner_text: raise SystemExit(f'RUNNER_PROBLEM_HARDCODE:{forbidden}')
        print(json.dumps({'status':'PASS','projects':len(problems),'families':len(set(p.get('family') for p in problems)),'required_models':sorted(REQUIRED_MODELS),'routing_conditions':[x['id'] for x in cfg['routing_conditions']]},sort_keys=True))
    finally:
        if tmp: tmp.cleanup()
if __name__=='__main__': main()
