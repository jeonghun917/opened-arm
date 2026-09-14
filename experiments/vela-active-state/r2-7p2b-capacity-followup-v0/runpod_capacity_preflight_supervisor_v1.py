#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys,threading,time,traceback,urllib.request,uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from typing import Any

@dataclass
class Ctx:
 root:Path; token:str; source_commit:str
 @property
 def status(self): return self.root/'.vela-preflight-status.json'
 @property
 def result(self): return self.root/'capacity_preflight.json'
 @property
 def started(self): return self.root/'.vela-preflight-started'

def atomic(path:Path,obj:dict[str,Any]):
 tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,separators=(',',':'),sort_keys=True)+'\n'); tmp.replace(path)
def load(path:Path):
 try:
  x=json.loads(path.read_text()); return x if isinstance(x,dict) else None
 except Exception:return None
def valid(ctx:Ctx):
 x=load(ctx.result)
 if not x or x.get('status')!='PASS' or x.get('scientific_evidence') is not False or x.get('purpose')!='P-R2-03_RESULT_BLIND_CAPACITY_PREFLIGHT' or x.get('source_commit')!=ctx.source_commit:return None
 try: raw=ctx.result.read_bytes()
 except Exception:return None
 return x,raw
def update(ctx:Ctx,state:str,phase:str,boot_id:str,**extra):
 x={'schema':'vela-runpod-capacity-preflight-proxy:v1','state':state,'phase':phase,'source_commit':ctx.source_commit,'boot_id':boot_id,'updated_unix':int(time.time())}; x.update(extra); atomic(ctx.status,x)
def handler(ctx:Ctx):
 class H(BaseHTTPRequestHandler):
  def log_message(self,*a):return
  def sendb(self,code:int,body:bytes):
   self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body)
  def do_GET(self):
   if self.path==f'/v1/status/{ctx.token}': self.sendb(200,ctx.status.read_bytes() if ctx.status.exists() else b'{"state":"STARTING"}\n'); return
   if self.path==f'/v1/result/{ctx.token}':
    s=load(ctx.status) or {}
    if s.get('state')!='COMPLETE' or not ctx.result.exists(): self.sendb(409,b'{"error":"RESULT_NOT_READY"}\n'); return
    self.sendb(200,ctx.result.read_bytes()); return
   self.sendb(404,b'{"error":"NOT_FOUND"}\n')
 return H

def worker(ctx:Ctx,boot_id:str):
 deadline=time.monotonic()+int(os.environ.get('VELA_PREFLIGHT_WORKLOAD_TIMEOUT_SECONDS','840'))
 try:
  recovered=valid(ctx)
  if recovered:
   _,raw=recovered; update(ctx,'COMPLETE','RECOVERED_EXISTING_RESULT',boot_id,result_bytes=len(raw),result_sha256=hashlib.sha256(raw).hexdigest(),scientific_evidence=False); return
  if ctx.started.exists(): update(ctx,'FAILED','RESTART_RECOVERY',boot_id,failure_class='CONTAINER_RESTART_AFTER_PREFLIGHT_START_NO_VALID_RESULT',scientific_evidence=False); return
  ctx.started.write_text(boot_id+'\n')
  update(ctx,'RUNNING','PIP_INSTALL',boot_id)
  remaining=deadline-time.monotonic()
  if remaining<=0: raise TimeoutError('deadline before pip')
  p=subprocess.run([sys.executable,'-m','pip','install','--quiet','--extra-index-url','https://download.pytorch.org/whl/cu126','torch==2.7.1+cu126','rwkv==0.8.32','tokenizers==0.21.4','numpy==2.4.6','huggingface_hub==0.36.0'],cwd=ctx.root,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=remaining,text=True)
  if p.returncode!=0: update(ctx,'FAILED','PIP_INSTALL',boot_id,failure_class='PIP_INSTALL_FAILED',exit_code=p.returncode,failure_detail=(p.stderr or '')[-4000:],scientific_evidence=False); return
  update(ctx,'RUNNING','DOWNLOAD_PREFLIGHT',boot_id)
  url=os.environ['VELA_BASE_RAW'].rstrip('/')+'/runpod_capacity_preflight_v1.py'; dest=ctx.root/'runpod_capacity_preflight_v1.py'; urllib.request.urlretrieve(url,dest)
  remaining=deadline-time.monotonic()
  if remaining<=0: raise TimeoutError('deadline before preflight')
  update(ctx,'RUNNING','RUN_PREFLIGHT',boot_id)
  env=os.environ.copy(); proc=subprocess.run([sys.executable,str(dest)],cwd=ctx.root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=remaining,text=True)
  r=valid(ctx)
  if proc.returncode!=0 or r is None: update(ctx,'FAILED','RUN_PREFLIGHT',boot_id,failure_class='PREFLIGHT_FAILED_OR_INVALID_RESULT',exit_code=proc.returncode,failure_detail=(proc.stderr or '')[-4000:],scientific_evidence=False,result_file_present=ctx.result.exists()); return
  _,raw=r; update(ctx,'COMPLETE','RESULT_READY',boot_id,result_bytes=len(raw),result_sha256=hashlib.sha256(raw).hexdigest(),scientific_evidence=False)
 except Exception as exc: update(ctx,'FAILED','SUPERVISOR_EXCEPTION',boot_id,failure_class=type(exc).__name__,failure_detail=(str(exc)+'\n'+traceback.format_exc())[-4000:],scientific_evidence=False)

def main():
 token=os.environ.get('VELA_RESULT_TOKEN',''); source=os.environ.get('VELA_SOURCE_COMMIT','')
 if len(token)<32 or len(source)!=40: raise SystemExit('invalid preflight supervisor identity')
 root=Path(os.environ.get('VELA_WORK_ROOT','/workspace/vela-preflight')); root.mkdir(parents=True,exist_ok=True); ctx=Ctx(root,token,source); boot=uuid.uuid4().hex; update(ctx,'BOOTING','SUPERVISOR_START',boot)
 threading.Thread(target=worker,args=(ctx,boot),daemon=True).start(); port=int(os.environ.get('VELA_RESULT_PORT','8000')); server=ThreadingHTTPServer(('0.0.0.0',port),handler(ctx)); server.daemon_threads=True; server.serve_forever(poll_interval=.5)
if __name__=='__main__':main()
