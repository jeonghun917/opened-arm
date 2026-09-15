from __future__ import annotations
import json

def render_probe_prompt(ep,problem,retrieval_cfg):
 prompt=str(ep['model_input']).rstrip()
 if retrieval_cfg.get('include_current_durable_snapshot_at_probe'):
  snap=(problem.get('memory_contract',{}).get('current_snapshot_by_episode') or {}).get(str(ep['episode']))
  if snap is None: raise ValueError(f'missing durable snapshot ep{ep["episode"]}')
  prompt+='\n\nAuthoritative durable-memory retrieval for this exact episode:\n'+json.dumps(snap,sort_keys=True,separators=(',',':'))
 return prompt+'\n\nReturn exactly one JSON object and no prose.'
def historical_values(problem,domain,episode):
 out={}
 for e in problem.get('memory_contract',{}).get('durable_memory_events',[]):
  if int(e.get('episode',10**9))>episode or e.get('domain')!=domain: continue
  for k,v in (e.get('record') or {}).items(): out.setdefault(str(k),set()).add(json.dumps(v,sort_keys=True))
 return out
def evaluate_probe(problem,ep,obj,raw,termination_reason=None):
 probe=ep['probe']; expected=dict(probe['expected']); required=[str(x) for x in probe['required_keys']]; parse_ok=obj is not None; got=obj or {}; corr={k:(k in got and got[k]==expected[k]) for k in required}; history=historical_values(problem,str(ep.get('domain')),int(ep['episode'])); stale=[]; inventions=[]
 if parse_ok:
  omissions=[k for k in required if k not in got or got[k] is None]
  for k in required:
   if k not in got or got[k] is None or got[k]==expected[k]: continue
   enc=json.dumps(got[k],sort_keys=True)
   (stale if enc in history.get(k,set()) else inventions).append(k)
  inventions.extend(sorted(set(got)-set(required)) if isinstance(got,dict) else [])
  output_failure=None
 else:
  omissions=[];output_failure='NO_VALID_FINAL_JSON'
 return {'episode':ep['episode'],'event':ep.get('event'),'domain':ep.get('domain'),'primary':bool(probe.get('primary')),'parse_ok':parse_ok,'output_failure':output_failure,'termination_reason':termination_reason,'exact_json':obj==expected,'field_correct':corr,'field_accuracy':sum(int(v) for v in corr.values())/max(1,len(required)),'omissions':omissions,'stale_value_errors':stale,'inventions':inventions,'raw_output':raw,'parsed_output':obj,'expected':expected}
