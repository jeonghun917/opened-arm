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
def unpack_problem_pack(path:Path):
 if path.is_dir(): return path,None
 tmp=tempfile.TemporaryDirectory(prefix='vela-statebank-pack-'); zip_path=Path(tmp.name)/'problem_pack.zip'
 if path.name.endswith('.pack.json') or path.name=='PACK_V1.json':
  d=load_json(path)
  if d.get('schema')!='vela-base64-zip-chunks:v1': raise ValueError('unsupported pack descriptor')
  encoded=''.join((path.parent/x).read_text().strip() for x in d['chunks']); raw=base64.b64decode(encoded)
  if hashlib.sha256(raw).hexdigest()!=d['zip_sha256']: raise ValueError('pack zip digest mismatch')
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
 root,tmp=unpack_problem_pack(pack); manifest=load_json(root/'SUITE_MANIFEST_V1.json'); registry=load_json(root/'MODEL_REGISTRY_V1.json')
 matches=[x for x in registry.get('profiles',[]) if x.get('id')==profile_id]
 if len(matches)!=1: raise ValueError(f'model selection must resolve once: {profile_id}')
 card=registry.get('selection_cardinality') or {}
 if card.get('min')!=1 or card.get('max')!=1: raise ValueError('registry must require exactly one model')
 if retrieval not in cfg.get('retrieval_modes',{}): raise ValueError(f'unsupported retrieval mode {retrieval}')
 problems=[]
 for item in manifest.get('projects',[]):
  p=root/item['file']
  if sha256_file(p)!=item['sha256']: raise ValueError(f'problem hash mismatch {item["file"]}')
  problems.append(load_json(p))
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
