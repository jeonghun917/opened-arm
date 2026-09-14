#!/usr/bin/env python3
from __future__ import annotations
import gc, hashlib, importlib.metadata, json, os, subprocess, sys, time, traceback
from pathlib import Path

EXPECTED_TORCH="2.7.1+cu126"
MODELS=[
 {"scale":"2.9B","profile":"g1i","repo":"BlinkDL/rwkv7-g1","revision":"ede85bf8ab2e59aff7d7ca909fbbc73317866d89","file":"rwkv7-g1i-2.9b-20260805-ctx16384.pth","size":5896273469,"sha256":"ac1ae23d0e65c1d35ba523eacd81a2a4dacb7b886479909bbff34f312e766320"},
 {"scale":"7.2B","profile":"g1i","repo":"BlinkDL/rwkv7-g1","revision":"ede85bf8ab2e59aff7d7ca909fbbc73317866d89","file":"rwkv7-g1i-7.2b-20260805-ctx16384.pth","size":14400007869,"sha256":"0d09d8961448032501c4d432c33a224c66356d43c10174386ea86b0da2b127d8"},
]

def sha256_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def state_bytes(state)->int:return int(sum(x.numel()*x.element_size() for x in state))
def tensor_state_exact(a,b)->bool:return len(a)==len(b) and all(x.shape==y.shape and x.dtype==y.dtype and __import__('torch').equal(x,y) for x,y in zip(a,b))
def write_report(x:dict)->None:
 Path('capacity_preflight.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(x,ensure_ascii=False,indent=2),flush=True)

def main()->None:
 import numpy as np, torch
 from huggingface_hub import hf_hub_download
 if torch.__version__!=EXPECTED_TORCH: raise RuntimeError(f'torch drift {torch.__version__}')
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
 arches=list(torch.cuda.get_arch_list()); cap=tuple(int(x) for x in torch.cuda.get_device_capability(0)); device=torch.cuda.get_device_name(0)
 if cap!=(8,6) or 'sm_86' not in arches or 'A40' not in device: raise RuntimeError(f'A40 sm_86 required; device={device} cap={cap} arches={arches}')
 os.environ['RWKV_V7_ON']='1'; os.environ['RWKV_JIT_ON']='1'; os.environ['RWKV_CUDA_ON']='0'
 from rwkv.model import RWKV
 from rwkv.utils import PIPELINE
 results=[]
 for spec in MODELS:
  t0=time.monotonic(); p=Path(hf_hub_download(repo_id=spec['repo'],filename=spec['file'],revision=spec['revision'])); dl=time.monotonic()-t0
  actual_size=p.stat().st_size; actual_sha=sha256_file(p)
  if actual_size!=spec['size'] or actual_sha!=spec['sha256']: raise RuntimeError(f"model identity mismatch {spec['scale']}: {actual_size} {actual_sha}")
  torch.cuda.reset_peak_memory_stats(); t1=time.monotonic(); model=RWKV(model=str(p)[:-4],strategy='cuda fp16'); pipe=PIPELINE(model,'rwkv_vocab_v20230424'); load_s=time.monotonic()-t1
  zero=model.generate_zero_state()
  warm1=[int(x) for x in pipe.encode('VELA RUNPOD SCIENCE NEUTRAL WARMUP')]; warm2=[int(x) for x in pipe.encode(' CONTINUATION CHECK')]
  _,warm_state=model.forward(warm1,[x.detach().clone() for x in zero]); _,_=model.forward(warm2,[x.detach().clone() for x in warm_state]); torch.cuda.synchronize(); del warm_state
  seed=[int(x) for x in pipe.encode('VELA P-R2-03 RESULT BLIND STATE ROUNDTRIP')]
  cont=[int(x) for x in pipe.encode(' CONTINUATION')]
  if not seed or not cont: raise RuntimeError('empty probe tokenization')
  t2=time.monotonic(); _,state=model.forward(seed,[x.detach().clone() for x in zero]); cpu=[x.detach().contiguous().cpu().clone() for x in state]; restored=[x.to(device=y.device,dtype=y.dtype).contiguous() for x,y in zip(cpu,model.generate_zero_state())]
  restore_exact=tensor_state_exact(state,restored)
  logits,out_state=model.forward(cont,[x.detach().clone() for x in restored]); torch.cuda.synchronize(); fwd_s=time.monotonic()-t2
  if not restore_exact: raise RuntimeError(f"state CPU->GPU tensor restore mismatch {spec['scale']}")
  results.append({**spec,'actual_size':actual_size,'actual_sha256':actual_sha,'download_seconds':dl,'load_seconds':load_s,'probe_seconds':fwd_s,'zero_state_bytes':state_bytes(zero),'post_seed_state_bytes':state_bytes(state),'tensor_count':len(state),'state_restore_tensor_exact':restore_exact,'logits_shape':list(logits.shape),'peak_cuda_bytes':int(torch.cuda.max_memory_allocated())})
  del logits,out_state,restored,cpu,state,zero,pipe,model; gc.collect(); torch.cuda.empty_cache()
 report={'status':'PASS','scientific_evidence':False,'purpose':'P-R2-03_RESULT_BLIND_CAPACITY_PREFLIGHT','source_commit':os.environ.get('VELA_SOURCE_COMMIT'),'matched_profile':'g1i-20260805','runtime':{'python':sys.version.split()[0],'torch':torch.__version__,'torch_cuda_arch_list':arches,'rwkv':importlib.metadata.version('rwkv'),'tokenizers':importlib.metadata.version('tokenizers'),'numpy':np.__version__,'cuda':torch.version.cuda,'device':device,'device_capability':list(cap),'strategy':'cuda fp16','rwkv_cuda_kernel':False},'models':results}
 write_report(report)

if __name__=='__main__':
 try: main()
 except BaseException as exc:
  write_report({'status':'PREFLIGHT_ERROR','scientific_evidence':False,'purpose':'P-R2-03_RESULT_BLIND_CAPACITY_PREFLIGHT','source_commit':os.environ.get('VELA_SOURCE_COMMIT'),'error_type':type(exc).__name__,'error':str(exc),'traceback_tail':traceback.format_exc().splitlines()[-100:]})
  raise
