#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys,threading,time,traceback,urllib.request,uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from typing import Any
@dataclass
class Ctx:
 root:Path; token:str; source:str; model:str; retrieval:str
 @property
 def status(self): return self.root/'.vela-status.json'
 @property
 def result(self): return self.root/'benchmark_result.json'
 @property
 def started(self): return self.root/'.vela-started'
def atomic(path,obj):
 t=path.with_suffix(path.suffix+'.tmp'); t.write_text(json.dumps(obj,separators=(',',':'),sort_keys=True)+'\n'); t.replace(path)
def set_status(ctx,state,phase,boot,**extra):
 x={'schema':'vela-statebank-proxy-status:v1','state':state,'phase':phase,'boot_id':boot,'source_commit':ctx.source,'model_profile_id':ctx.model,'retrieval_mode':ctx.retrieval,'updated_unix':int(time.time())}; x.update(extra); atomic(ctx.status,x)
def valid_result(ctx):
 try: raw=ctx.result.read_bytes(); obj=json.loads(raw)
 except Exception:return None
 if obj.get('schema')!='vela-statebank-benchmark-result:v1' or obj.get('status')!='COMPLETE' or obj.get('scientific_evidence') is not True:return None
 if obj.get('source_commit')!=ctx.source or obj.get('model_profile',{}).get('id')!=ctx.model or obj.get('retrieval_mode')!=ctx.retrieval:return None
 return obj,raw
def handler(ctx):
 class H(BaseHTTPRequestHandler):
  def log_message(self,*a):return
  def sendb(self,code,body):self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
  def do_GET(self):
   if self.path==f'/v1/status/{ctx.token}':self.sendb(200,ctx.status.read_bytes() if ctx.status.exists() else b'{"state":"STARTING"}\n');return
   if self.path==f'/v1/result/{ctx.token}':
    if valid_result(ctx) is None:self.sendb(409,b'{"error":"RESULT_NOT_READY"}\n');return
    self.sendb(200,ctx.result.read_bytes());return
   self.sendb(404,b'{"error":"NOT_FOUND"}\n')
 return H
def run(ctx,boot,deadline,phase,cmd,env=None):
 remain=deadline-time.monotonic()
 if remain<=0:return 124
 set_status(ctx,'RUNNING',phase,boot)
 with (ctx.root/f'{phase}.stdout').open('wb') as out,(ctx.root/f'{phase}.stderr').open('wb') as err:
  try:return subprocess.run(cmd,cwd=ctx.root,env=env,stdout=out,stderr=err,timeout=remain,check=False).returncode
  except subprocess.TimeoutExpired:return 124
def dl(url,dest):dest.parent.mkdir(parents=True,exist_ok=True);urllib.request.urlretrieve(url,dest)
def download_pack(base,rel,dest):
 target=dest/'PACK_INPUT'; dl(base+'/'+rel,target)
 try:d=json.loads(target.read_text())
 except Exception:return target
 if d.get('schema')!='vela-base64-zip-chunks:v1':return target
 packdir=dest/'pack'; packdir.mkdir(); desc=packdir/'PACK_V1.json'; desc.write_text(json.dumps(d,indent=2)+'\n')
 for chunk in d['chunks']:dl(base+'/'+str(Path(rel).parent/chunk),packdir/chunk)
 return desc
def work(ctx,boot):
 deadline=time.monotonic()+int(os.environ.get('VELA_WORKLOAD_TIMEOUT_SECONDS','840')); base=os.environ['VELA_BASE_RAW'].rstrip('/')
 try:
  if valid_result(ctx):
   _,raw=valid_result(ctx);set_status(ctx,'COMPLETE','recovered',boot,result_bytes=len(raw),result_sha256=hashlib.sha256(raw).hexdigest());return
  if ctx.started.exists():set_status(ctx,'FAILED','restart_recovery',boot,failure_class='RESTART_WITHOUT_VALID_RESULT');return
  ctx.started.write_text(boot+'\n')
  rc=run(ctx,boot,deadline,'pip_install',[sys.executable,'-m','pip','install','--quiet','--extra-index-url','https://download.pytorch.org/whl/cu126','torch==2.7.1+cu126','rwkv==0.8.32','tokenizers==0.21.4','numpy==2.4.6','huggingface_hub==0.36.0'])
  if rc!=0:set_status(ctx,'FAILED','pip_install',boot,failure_class='PIP_INSTALL_FAILED',exit_code=rc);return
  set_status(ctx,'RUNNING','download_sources',boot)
  for name in ('statebank_runner.py','benchmark_contract.py','benchmark_metrics.py','rwkv_state_engine.py'):dl(base+'/'+name,ctx.root/name)
  dl(base+'/'+os.environ['VELA_EXECUTION_CONFIG_REL'],ctx.root/'execution_config.json'); pack=download_pack(base,os.environ['VELA_PROBLEM_PACK_REL'],ctx.root/'problem_pack')
  env=os.environ.copy();env['VELA_SOURCE_COMMIT']=ctx.source
  rc=run(ctx,boot,deadline,'run_benchmark',[sys.executable,'statebank_runner.py','--execution-config','execution_config.json','--problem-pack',str(pack),'--model-profile',ctx.model,'--retrieval-mode',ctx.retrieval,'--output','benchmark_result.json'],env)
  result=valid_result(ctx)
  if rc!=0 or result is None:set_status(ctx,'FAILED','run_benchmark',boot,failure_class='RUNNER_FAILED_OR_INVALID_RESULT',exit_code=rc,result_present=ctx.result.exists());return
  _,raw=result;set_status(ctx,'COMPLETE','result_ready',boot,scientific_evidence=True,result_bytes=len(raw),result_sha256=hashlib.sha256(raw).hexdigest())
 except Exception as e:set_status(ctx,'FAILED','supervisor_exception',boot,failure_class=type(e).__name__,failure_detail=(str(e)+'\n'+traceback.format_exc())[-4000:])
def main():
 token=os.environ.get('VELA_RESULT_TOKEN','');source=os.environ.get('VELA_SOURCE_COMMIT','');model=os.environ.get('VELA_MODEL_PROFILE_ID','');retrieval=os.environ.get('VELA_RETRIEVAL_MODE','')
 if len(token)<32 or len(source)!=40 or not model or not retrieval:raise SystemExit('invalid identity environment')
 root=Path(os.environ.get('VELA_WORK_ROOT','/workspace/vela'));root.mkdir(parents=True,exist_ok=True);ctx=Ctx(root,token,source,model,retrieval);boot=uuid.uuid4().hex;set_status(ctx,'BOOTING','supervisor_start',boot);threading.Thread(target=work,args=(ctx,boot),daemon=True).start();server=ThreadingHTTPServer(('0.0.0.0',int(os.environ.get('VELA_RESULT_PORT','8000'))),handler(ctx));server.daemon_threads=True;server.serve_forever(poll_interval=.5)
if __name__=='__main__':main()
