from __future__ import annotations
import base64,hashlib,json,tempfile,zipfile
from pathlib import Path
from typing import Any

def sha256_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 return h.hexdigest()
def load_json(path:Path)->Any: return json.loads(path.read_text(encoding='utf-8'))
def _compact_json(v:dict[str,Any])->str: return json.dumps(v,sort_keys=True,separators=(',',':'))
def _probe(keys:list[str],expected:dict[str,Any])->dict[str,Any]: return {'response_format':'json_object','required_keys':keys,'expected':expected,'primary':True}
def _expand_tabular_source(d:dict[str,Any]):
 banks=[str(x) for x in d.get('banks',[])]
 if len(banks)!=2 or len(set(banks))!=2: raise ValueError('tabular source v1 requires exactly two declared banks')
 problems=[]
 for family,f in sorted((d.get('families') or {}).items()):
  ka=[str(x) for x in f['ka']];kb=[str(x) for x in f['kb']];pka=[str(x) for x in f['pka']];pkb=[str(x) for x in f['pkb']]
  da,db=str(f['da']),str(f['db']);la,lb=str(f['la']),str(f['lb'])
  width=1+len(ka)*2+len(kb)*2
  for row in f.get('rows',[]):
   if len(row)!=width: raise ValueError(f'{family}: tabular row width')
   pid=str(row[0]);i=1
   a0=dict(zip(ka,row[i:i+len(ka)]));i+=len(ka);a1=dict(zip(ka,row[i:i+len(ka)]));i+=len(ka);b0=dict(zip(kb,row[i:i+len(kb)]));i+=len(kb);b1=dict(zip(kb,row[i:i+len(kb)]))
   snapshots={'1':{},'2':{da:a0},'3':{da:a0,db:b0},'4':{da:a1,db:b0},'5':{da:a1,db:b0},'6':{da:a1,db:b1},'7':{da:a1,db:b1},'8':{da:a1,db:b1}}
   memory_contract={'current_snapshot_by_episode':snapshots,'durable_memory_events':[{'episode':2,'domain':da,'op':'put','record':a0},{'episode':3,'domain':db,'op':'put','record':b0},{'episode':4,'domain':da,'op':'supersede','record':a1},{'episode':6,'domain':db,'op':'supersede','record':b1}],'model_retrieval_default':'OFF','supported_probe_retrieval_modes':['OFF','FULL_CURRENT_EQUALIZED'],'note':'Durable memory is independent from recurrent state; retrieval mode is an external experimental variable.'}
   problems.append({'schema':'vela-statebank-problem:v1','project_id':pid,'family':family,'episode_count':8,'bank_profile':{'banks':banks,'semantic_domain_to_bank':{da:banks[0],db:banks[1]},'control':'storage_matched_shuffled_route','bank_count':len(banks)},'memory_contract':memory_contract,'episodes':[
    {'episode':1,'kind':'event','event':'INITIAL_SHARED_SEED','domain':'shared','model_input':f"Project {pid}. Two independent work domains will evolve over time. Treat later corrections as authoritative. Do not merge one domain's facts into the other.",'state_commit':{'mode':'seed_all_banks','semantic_bank':None,'shuffled_bank':None},'probe':None},
    {'episode':2,'kind':'event','event':'DOMAIN_A_ADDITION','domain':da,'model_input':f'{la} update: {_compact_json(a0)}','state_commit':{'mode':'keep','semantic_bank':banks[0],'shuffled_bank':banks[0]},'probe':None},
    {'episode':3,'kind':'event','event':'DOMAIN_B_ADDITION','domain':db,'model_input':f'{lb} update: {_compact_json(b0)}','state_commit':{'mode':'keep','semantic_bank':banks[1],'shuffled_bank':banks[1]},'probe':None},
    {'episode':4,'kind':'event','event':'DOMAIN_A_CORRECTION','domain':da,'model_input':f'Authoritative correction for {la}. Replace the prior domain state with this complete current snapshot: {_compact_json(a1)}','state_commit':{'mode':'reset_then_commit','semantic_bank':banks[0],'shuffled_bank':banks[1]},'probe':None},
    {'episode':5,'kind':'probe','event':'DOMAIN_B_RETURN_PROBE','domain':db,'model_input':f"Return the current {lb} state as one JSON object with exactly these keys: {', '.join(pkb)}. Use only information actually available to you; do not invent missing values.",'state_commit':{'mode':'discard_probe_state','semantic_bank':banks[1],'shuffled_bank':banks[1]},'probe':_probe(pkb,b0)},
    {'episode':6,'kind':'event','event':'DOMAIN_B_RESOLUTION','domain':db,'model_input':f'Resolution update for the same previously seen {lb} record: its review_status is now "{b1["review_status"]}". The identifier and owner are unchanged.','state_commit':{'mode':'keep','semantic_bank':banks[1],'shuffled_bank':banks[1]},'probe':None},
    {'episode':7,'kind':'probe','event':'DOMAIN_A_RETURN_PROBE','domain':da,'model_input':f"Return the current {la} state as one JSON object with exactly these keys: {', '.join(pka)}. Use the latest authoritative state and do not invent missing values.",'state_commit':{'mode':'discard_probe_state','semantic_bank':banks[0],'shuffled_bank':banks[0]},'probe':_probe(pka,a1)},
    {'episode':8,'kind':'probe','event':'DOMAIN_B_POST_RESOLUTION_PROBE','domain':db,'model_input':f"Return the current {lb} state as one JSON object with exactly these keys: {', '.join(pkb)}. Use only retained state from prior events; do not invent missing values.",'state_commit':{'mode':'discard_probe_state','semantic_bank':banks[1],'shuffled_bank':banks[1]},'probe':_probe(pkb,b1)}
   ]})
 if not problems: raise ValueError('tabular source has no projects')
 manifest={'schema':'vela-statebank-benchmark-suite:v1','suite_id':'r2-statebank-construct-validity-v1','projects':[{'project_id':p['project_id'],'family':p['family']} for p in problems]}
 registry=d.get('model_registry') or {}
 return manifest,registry,problems

def unpack_problem_pack(path:Path):
 if path.is_dir(): return path,None
 tmp=tempfile.TemporaryDirectory(prefix='vela-statebank-pack-'); zip_path=Path(tmp.name)/'problem_pack.zip'
 if path.name.endswith('.pack.json') or path.name=='PACK_V1.json':
  d=load_json(path)
  if d.get('schema')!='vela-base64-zip-chunks:v1': tmp.cleanup(); raise ValueError('unsupported pack descriptor')
  encoded=''.join((path.parent/x).read_text().strip() for x in d['chunks']); raw=base64.b64decode(encoded)
  if hashlib.sha256(raw).hexdigest()!=d['zip_sha256']: tmp.cleanup(); raise ValueError('pack zip digest mismatch')
  zip_path.write_bytes(raw)
 elif path.name.endswith('.b64'): zip_path.write_bytes(base64.b64decode(path.read_text()))
 elif zipfile.is_zipfile(path): zip_path=path
 else: tmp.cleanup(); raise ValueError(f'unsupported problem pack: {path}')
 if not zipfile.is_zipfile(zip_path): tmp.cleanup(); raise ValueError('decoded problem pack is not zip')
 with zipfile.ZipFile(zip_path) as zf: zf.extractall(tmp.name)
 root=Path(tmp.name); dirs=[p for p in root.iterdir() if p.is_dir()]
 if len(dirs)==1 and not (root/'SUITE_MANIFEST_V1.json').exists(): root=dirs[0]
 return root,tmp

def load_contract(config:Path,pack:Path,profile_id:str,retrieval:str):
 cfg=load_json(config)
 if cfg.get('schema')!='vela-statebank-execution-config:v1': raise ValueError('unsupported execution config')
 tmp=None
 try: direct=load_json(pack) if pack.is_file() and pack.suffix.lower()=='.json' else None
 except Exception: direct=None
 if isinstance(direct,dict) and direct.get('schema')=='vela-statebank-tabular-problem-source:v1':
  manifest,registry,problems=_expand_tabular_source(direct);root=pack.parent
 else:
  root,tmp=unpack_problem_pack(pack); manifest=load_json(root/'SUITE_MANIFEST_V1.json'); registry=load_json(root/'MODEL_REGISTRY_V1.json');problems=[]
  for item in manifest.get('projects',[]):
   p=root/item['file']
   if sha256_file(p)!=item['sha256']: raise ValueEError(f'problem hash mismatch {item["file"]}')
   problems.append(load_json(p))
 matches=[x for x in registry.get('profiles',[]) if x.get('id')==profile_id]
 if len(matches)!=1: raise ValueError(f'model selection must resolve once: {profile_id}')
 card=registry.get('selection_cardinality') or {}
 if card.get('min')!=1 or card.get('max')!=1: raise ValueError('registry must require exactly one model')
 if retrieval not in cfg.get('retrieval_modes',{}): raise ValueError(f'unsupported retrieval mode {retrieval}')
 if not problems or len(problems)!=len(manifest.get('projects',[])): raise ValueError('incomplete problem suite')
 return cfg,manifest,registry,problems,root,tmp

def validate_contract(cfg,problems):
 cond=cfg.get('routing_conditions') or []; fields=[str(x.get('route_field')) for x in cond]
 if len(cond)<2 or len({x.get('id') for x in cond})!=len(cond): raise ValueError('invalid routing conditions')
 modes=cfg.get('state_modes') or {}
 if not modes: raise ValueError('state modes missing')
 for p in problems:
  pid=p.get('project_id'); banks=p.get('bank_profile',{}).get('banks') or []; eps=p.get('episodes') or []
  if not banks or len(banks)!=len(set(banks)) or int(p['bank_profile'].get('bank_count',-1))!=len(banks): raise ValueError(f'{pid}: invalid banks')
  if int(p.get('episode_count',-1))!=len(eps): raise ValueError(f'{pid}: episode count')
  contrast=False
  for ep in eps:
   sc=ep.get('state_commit') or {}; mode=modes.get(sc.get('mode'))
   if mode is None or bool(mode.get('probe'))!=bool(ep.get('probe')): raise ValueError(f'{pid} ep{ep.get("episode")}: mode mismatch')
   chosen=[]
   for field in fields:
    bank=sc.get(field)
    if bank is not None and bank not in banks: raise ValueError(f'{pid} ep{ep.get("episode")}: unknown bank')
    chosen.append(bank)
   if len(set(x for x in chosen if x is not None))>1: contrast=True
   probe=ep.get('probe')
   if probe:
    if mode.get('commit')!='none': raise ValueError(f'{pid}: probe state persists')
    prompt=str(ep.get('model_input',''))
    for v in (probe.get('expected') or {}).values():
     if str(v) in prompt: raise ValueError(f'{pid}: answer leaked')
  if not contrast: raise ValueError(f'{pid}: no routing contrast')
