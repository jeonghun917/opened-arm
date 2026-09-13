#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.metadata, json, os, platform
from pathlib import Path

MODEL_FILE="rwkv7-g1a-2.9b-20250924-ctx4096.pth"
MODEL_SIZE=5896274949
MODEL_SHA256="df5716263b617e7da83590446fb2a98b6663447cfd52e8e9b49e11ce7f4faa3e"

def pkg(n):
    try: return importlib.metadata.version(n)
    except importlib.metadata.PackageNotFoundError: return None

def sha256_file(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024), b''): h.update(c)
    return h.hexdigest()

def main():
    import torch
    model_path=Path(os.environ.get('VELA_MODEL_PATH', f'/runpod-volume/models/{MODEL_FILE}'))
    out=Path(os.environ.get('VELA_PREFLIGHT_OUT','/runpod-volume/artifacts/p-r2-02/runpod-preflight.json'))
    out.parent.mkdir(parents=True, exist_ok=True)
    r={
      'status':'P-R2-02_RUNPOD_PREFLIGHT','scientific_evidence':False,'provider':'RUNPOD',
      'source_commit':os.environ.get('VELA_SOURCE_COMMIT'),
      'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
                 'cuda_available':torch.cuda.is_available(),'rwkv':pkg('rwkv'),'numpy':pkg('numpy'),'tokenizers':pkg('tokenizers')},
      'model':{'path':str(model_path),'expected_size':MODEL_SIZE,'expected_sha256':MODEL_SHA256},'checks':{}
    }
    if torch.cuda.is_available():
        r['runtime']['device_name']=torch.cuda.get_device_name(0)
        r['runtime']['device_capability']=list(torch.cuda.get_device_capability(0))
    c=r['checks']; c['model_exists']=model_path.is_file()
    if c['model_exists']:
        r['model']['actual_size']=model_path.stat().st_size
        r['model']['actual_sha256']=sha256_file(model_path)
        c['model_size_exact']=r['model']['actual_size']==MODEL_SIZE
        c['model_sha256_exact']=r['model']['actual_sha256']==MODEL_SHA256
    c['rwkv_version_exact']=pkg('rwkv')=='0.8.32'
    c['numpy_version_exact']=pkg('numpy')=='2.4.6'
    c['tokenizers_version_exact']=pkg('tokenizers')=='0.21.4'
    c['cuda_available']=torch.cuda.is_available()
    if all(c.get(k) for k in ['model_exists','model_size_exact','model_sha256_exact','rwkv_version_exact','numpy_version_exact','tokenizers_version_exact','cuda_available']):
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE
        strategy=os.environ.get('VELA_RWKV_STRATEGY','cuda fp16'); r['runtime']['strategy']=strategy
        model=RWKV(model=str(model_path),strategy=strategy); pipe=PIPELINE(model,'rwkv_vocab_v20230424')
        logits,state=model.forward(pipe.encode('VELA RUNPOD PREFLIGHT NEUTRAL STATE ROUNDTRIP'),None)
        c['forward_pass']=logits is not None and bool(state)
        sp=out.with_suffix('.state.pt'); torch.save(state,sp)
        restored=torch.load(sp,map_location='cpu',weights_only=False)
        ok=len(state)==len(restored)
        if ok:
            for a,b in zip(state,restored):
                aa=a.detach().cpu() if hasattr(a,'detach') else a; bb=b.detach().cpu() if hasattr(b,'detach') else b
                if hasattr(aa,'shape'):
                    if aa.shape!=bb.shape or aa.dtype!=bb.dtype or not torch.equal(aa,bb): ok=False; break
                elif aa!=bb: ok=False; break
        c['state_roundtrip_exact']=ok
        r['state']={'tensor_count':len(state),'snapshot_size':sp.stat().st_size,'snapshot_sha256':sha256_file(sp)}
    required=['model_exists','model_size_exact','model_sha256_exact','rwkv_version_exact','numpy_version_exact','tokenizers_version_exact','cuda_available','forward_pass','state_roundtrip_exact']
    r['preflight_ok']=all(c.get(k) is True for k in required)
    out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    if not r['preflight_ok']: raise SystemExit('RunPod preflight failed')

if __name__=='__main__': main()
