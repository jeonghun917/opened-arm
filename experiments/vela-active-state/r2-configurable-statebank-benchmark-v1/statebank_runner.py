#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,gc
from pathlib import Path
import benchmark_contract as contract
import benchmark_metrics as metrics
import rwkv_state_engine as engine

def select_project_shard(problems, shard_index, shard_count):
 if shard_count < 1 or shard_index < 0 or shard_index >= shard_count: raise ValueError('invalid shard index/count')
 selected=[p for i,p in enumerate(problems) if i % shard_count == shard_index]
 if not selected: raise ValueError(f'empty project shard {shard_index}/{shard_count}')
 return selected

def summarize_primary(probes):
 fields=sum(len(x['field_correct']) for x in probes);correct=sum(sum(int(v) for v in x['field_correct'].values()) for x in probes);term={}
 for x in probes:
  reason=x.get('termination_reason') or 'UNKNOWN';term[reason]=term.get(reason,0)+1
 return {'primary_probe_count':len(probes),'primary_exact_count':sum(int(x['exact_json']) for x in probes),'primary_exact_rate':sum(int(x['exact_json']) for x in probes)/max(1,len(probes)),'primary_field_accuracy':correct/max(1,fields),'primary_stale_value_error_count':sum(len(x['stale_value_errors']) for x in probes),'primary_omission_count':sum(len(x['omissions']) for x in probes),'primary_invention_count':sum(len(x['inventions']) for x in probes),'primary_parse_failure_count':sum(int(not x['parse_ok']) for x in probes),'termination_reason_counts':term}

def summarize_domains(probes):
 groups={}
 for x in probes: groups.setdefault(str(x.get('domain')),[]).append(x)
 out={}
 for domain,rows in sorted(groups.items()):
  s=summarize_primary(rows);out[domain]={'primary_probe_count':s['primary_probe_count'],'primary_exact_count':s['primary_exact_count'],'primary_exact_rate':s['primary_exact_rate'],'primary_field_accuracy':s['primary_field_accuracy'],'parse_failure_count':s['primary_parse_failure_count'],'stale_value_error_count':s['primary_stale_value_error_count'],'omission_count':s['primary_omission_count'],'invention_count':s['primary_invention_count'],'termination_reason_counts':s['termination_reason_counts']}
 return out

def run_condition(model,pipe,problem,condition,cfg,retrieval):
 zero=engine.snap(model.generate_zero_state()); banks={name:engine.clone_snap(zero) for name in problem['bank_profile']['banks']}; route_field=condition['route_field']; probes=[]; trace=[]; units=0; modes=cfg['state_modes']; retrieval_cfg=cfg['retrieval_modes'][retrieval]
 for ep in problem['episodes']:
  sc=ep['state_commit']; mode=modes[sc['mode']]; bank=sc.get(route_field); inp=mode['input_state']
  if inp=='zero': snapshot=engine.clone_snap(zero)
  elif inp=='selected':
   if bank not in banks: raise RuntimeError(f'missing bank {bank}')
   snapshot=engine.clone_snap(banks[bank])
  else: raise RuntimeError(f'unsupported input_state {inp}')
  before=engine.state_digest(snapshot)
  if mode.get('probe'):
   prompt=metrics.render_probe_prompt(ep,problem,retrieval_cfg); parsed,raw,u,termination_reason=engine.greedy_json_probe(model,pipe,snapshot,prompt,int(cfg['generation']['max_new_tokens'])); units+=u; probes.append(metrics.evaluate_probe(problem,ep,parsed,raw,termination_reason)); after=before
  else:
   _,out,u=engine.feed_text(model,pipe,snapshot,str(ep['model_input'])); units+=u; commit=mode['commit']
   if commit=='all':
    for name in banks: banks[name]=engine.clone_snap(out)
   elif commit=='selected':
    if bank not in banks: raise RuntimeError(f'commit bank missing {bank}')
    banks[bank]=engine.clone_snap(out)
   elif commit!='none': raise RuntimeError(f'unsupported commit policy {commit}')
   after=engine.state_digest(out)
  trace.append({'episode':ep['episode'],'event':ep.get('event'),'mode':sc['mode'],'selected_bank':bank,'input_state_sha256':before,'active_output_state_sha256':after,'bank_state_sha256':{n:engine.state_digest(v) for n,v in banks.items()}})
  gc.collect()
 primary=[x for x in probes if x['primary']]; summary=summarize_primary(primary)
 return {'condition':condition['id'],'route_field':route_field,'bank_count':len(banks),'bank_names':list(banks),'probe_results':probes,**summary,'domain_metrics':summarize_domains(primary),'proposal_units':units,'state_trace':trace}

def aggregate_results(cfg,results):
 agg={}
 for c in cfg['routing_conditions']:
  cid=c['id']; rows=[x['conditions'][cid] for x in results]; primary=[p for row in rows for p in row['probe_results'] if p['primary']]; s=summarize_primary(primary); domains=summarize_domains(primary)
  agg[cid]={'primary_probe_count':s['primary_probe_count'],'primary_exact_count':s['primary_exact_count'],'primary_exact_rate':s['primary_exact_rate'],'primary_field_accuracy':s['primary_field_accuracy'],'stale_value_error_count':s['primary_stale_value_error_count'],'omission_count':s['primary_omission_count'],'invention_count':s['primary_invention_count'],'parse_failure_count':s['primary_parse_failure_count'],'termination_reason_counts':s['termination_reason_counts'],'domain_metrics':domains,'zero_exact_domains':[d for d,v in domains.items() if v['primary_exact_count']==0],'proposal_units':sum(x['proposal_units'] for x in rows)}
 comps=[]
 for c in cfg.get('comparisons',[]):
  t,ctl=c['treatment'],c['control']; comps.append({'id':c['id'],'treatment':t,'control':ctl,'exact_rate_delta':agg[t]['primary_exact_rate']-agg[ctl]['primary_exact_rate'],'field_accuracy_delta':agg[t]['primary_field_accuracy']-agg[ctl]['primary_field_accuracy'],'stale_error_delta':agg[t]['stale_value_error_count']-agg[ctl]['stale_value_error_count'],'omission_delta':agg[t]['omission_count']-agg[ctl]['omission_count'],'invention_delta':agg[t]['invention_count']-agg[ctl]['invention_count'],'parse_failure_delta':agg[t]['parse_failure_count']-agg[ctl]['parse_failure_count']})
 return agg,comps

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--execution-config',required=True); ap.add_argument('--problem-pack',required=True); ap.add_argument('--model-profile',required=True); ap.add_argument('--retrieval-mode',required=True); ap.add_argument('--shard-index',type=int,default=0); ap.add_argument('--shard-count',type=int,default=1); ap.add_argument('--output',default='benchmark_result.json'); ap.add_argument('--validate-only',action='store_true'); a=ap.parse_args()
 cfg,manifest,registry,all_problems,_root,tmp=contract.load_contract(Path(a.execution_config),Path(a.problem_pack),a.model_profile,a.retrieval_mode)
 try:
  contract.validate_contract(cfg,all_problems); problems=select_project_shard(all_problems,a.shard_index,a.shard_count); selected=next(x for x in registry['profiles'] if x['id']==a.model_profile)
  if a.validate_only:
   print(json.dumps({'status':'VALIDATION_PASS','suite_id':manifest.get('suite_id'),'suite_project_count':len(all_problems),'selected_project_count':len(problems),'shard_index':a.shard_index,'shard_count':a.shard_count,'model_profile_id':a.model_profile,'retrieval_mode':a.retrieval_mode,'routing_conditions':[x['id'] for x in cfg['routing_conditions']],'bank_counts_observed':sorted(set(len(p['bank_profile']['banks']) for p in problems)),'families_observed':sorted(set(str(p.get('family')) for p in problems))},indent=2,sort_keys=True)); return
  os.environ['RWKV_V7_ON']='1'; os.environ['RWKV_JIT_ON']='1'; os.environ['RWKV_CUDA_ON']='0'
  import torch
  from huggingface_hub import hf_hub_download
  from rwkv.model import RWKV
  from rwkv.utils import PIPELINE
  if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
  cap=tuple(int(x) for x in torch.cuda.get_device_capability(0))
  if cap!=(8,6) or 'A40' not in torch.cuda.get_device_name(0): raise RuntimeError(f'Secure A40 required: {torch.cuda.get_device_name(0)} cap={cap}')
  path=Path(hf_hub_download(repo_id=selected['repo'],filename=selected['file'],revision=selected['revision']))
  if path.stat().st_size!=int(selected['size']) or contract.sha256_file(path)!=selected['sha256']: raise RuntimeError('model identity mismatch')
  strategy='cuda fp16'; model=RWKV(model=str(path)[:-4],strategy=strategy); pipe=PIPELINE(model,'rwkv_vocab_v20230424'); results=[]
  for p in problems:
   row={'project_id':p['project_id'],'family':p.get('family'),'conditions':{}}
   for c in cfg['routing_conditions']: row['conditions'][c['id']]=run_condition(model,pipe,p,c,cfg,a.retrieval_mode)
   results.append(row); gc.collect(); torch.cuda.empty_cache()
  agg,comps=aggregate_results(cfg,results); out={'schema':cfg['result_schema'],'status':'COMPLETE','scientific_evidence':True,'source_commit':os.environ.get('VELA_SOURCE_COMMIT'),'suite_id':manifest.get('suite_id'),'suite_project_count':len(all_problems),'project_count':len(results),'shard_index':a.shard_index,'shard_count':a.shard_count,'model_profile':selected,'retrieval_mode':a.retrieval_mode,'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'device_capability':list(cap),'strategy':strategy},'projects':results,'aggregate':agg,'comparisons':comps}; Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'status':'COMPLETE','project_count':len(results),'shard_index':a.shard_index,'shard_count':a.shard_count,'aggregate':agg,'comparisons':comps},indent=2,sort_keys=True))
 finally:
  if tmp: tmp.cleanup()
if __name__=='__main__': main()
