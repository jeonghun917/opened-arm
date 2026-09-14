#!/usr/bin/env python3
from __future__ import annotations
import argparse,base64,hashlib,json,os,secrets,signal,subprocess,sys,time,urllib.error,urllib.request
from pathlib import Path
PINNED_IMAGE='pytorch/pytorch@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3'
PROXY_STARTUP_GRACE_SECONDS=120; POLL_INTERVAL_SECONDS=3
class ControllerSignal(Exception):
 def __init__(self,n): super().__init__(f'controller signal {n}'); self.signum=n
def sh(cmd,**kw): return subprocess.run(cmd,text=True,**kw)
def emit(tag,**d): print(json.dumps({'tag':tag,'unix':int(time.time()),**d},sort_keys=True),flush=True)
def safe_rel(v):
 p=Path(v)
 if not v or p.is_absolute() or '..' in p.parts or '\\' in v: raise ValueError(f'unsafe relative path {v}')
 return p.as_posix()
def req(url,timeout=6):
 try:
  with urllib.request.urlopen(url,timeout=timeout) as r:
   o=json.loads(r.read()); return r.status,o if isinstance(o,dict) else {}
 except urllib.error.HTTPError as e:
  try:o=json.loads(e.read())
  except Exception:o={}
  return e.code,o if isinstance(o,dict) else {}
 except Exception:return 0,{}
def result_get(url,path):
 try:
  with urllib.request.urlopen(url,timeout=15) as r: raw=r.read();path.write_bytes(raw);return r.status,raw
 except Exception:return 0,b''
request_json=req
fetch_result=result_get
def image_ref(evidence): (evidence/'image-ref.txt').write_text(PINNED_IMAGE+'\n');return PINNED_IMAGE
def redact(s,token,sup): return s.replace(token,'<redacted-result-token>').replace(sup,'<redacted-supervisor-b64>')
def sanitize(o,token,sup):
 if isinstance(o,dict): return {k:('<redacted>' if k in {'VELA_RESULT_TOKEN','VELA_SUPERVISOR_B64'} else sanitize(v,token,sup)) for k,v in o.items()}
 if isinstance(o,list):
  return [('VELA_RESULT_TOKEN=<redacted>' if isinstance(x,str) and x.startswith('VELA_RESULT_TOKEN=') else 'VELA_SUPERVISOR_B64=<redacted>' if isinstance(x,str) and x.startswith('VELA_SUPERVISOR_B64=') else sanitize(x,token,sup)) for x in o]
 return redact(o,token,sup) if isinstance(o,str) else o
def save_obj(raw,path,token,sup):
 try:path.write_text(json.dumps(sanitize(json.loads(raw),token,sup),indent=2,sort_keys=True)+'\n')
 except Exception:path.write_text(json.dumps({'unparsed_sha256':hashlib.sha256(raw.encode()).hexdigest(),'bytes':len(raw)})+'\n')
def logs(pod,evidence,token,sup,label):
 for src in ('system','container'):
  try:
   r=sh(['runpodctl','pod','logs',pod,'--since','3m','--source',src],capture_output=True,timeout=20); body=redact((r.stdout or '')+(('\nSTDERR:\n'+r.stderr) if r.stderr else ''),token,sup); rc=r.returncode
  except Exception as e: body=f'LOG_CAPTURE_EXCEPTION:{type(e).__name__}:{e}\n';rc=-1
  (evidence/f'{label}-{src}.log').write_text(body);emit('VELA_POD_LOG_SNAPSHOT',source=src,exit_code=rc,tail=body[-1600:].replace('\n',' | '))
def pod_get(pod,evidence,token,sup):
 try:r=sh(['runpodctl','pod','get',pod,'--include-machine'],capture_output=True,timeout=20)
 except Exception as e:(evidence/'last-pod-get.stderr').write_text(f'POD_GET_EXCEPTION:{type(e).__name__}:{e}\n');return None,None,None
 err=redact(r.stderr or '',token,sup);(evidence/'last-pod-get.stderr').write_text(err)
 if r.returncode:
  if '"code":"not_found"' in err or '"code": "not_found"' in err:return None,None,'POD_NOT_FOUND'
  return None,None,None
 try:o=json.loads(r.stdout or '')
 except Exception:save_obj(r.stdout or '',evidence/'last-pod-get.json',token,sup);return None,None,'POD_GET_INVALID_JSON'
 (evidence/'last-pod-get.json').write_text(json.dumps(sanitize(o,token,sup),indent=2,sort_keys=True)+'\n')
 status=str(o.get('runtimeStatus')).lower() if o.get('runtimeStatus') is not None else None; u=o.get('uptimeSeconds');return status,float(u) if isinstance(u,(int,float)) else None,None
def status_error(s,source,model,retrieval):
 for k,v in {'schema':'vela-statebank-proxy-status:v1','source_commit':source,'model_profile_id':model,'retrieval_mode':retrieval}.items():
  if s.get(k)!=v:return f'{k}:{s.get(k)!r}!={v!r}'
 return None if s.get('state') in {'BOOTING','RUNNING','FAILED','COMPLETE'} else f"state:{s.get('state')!r}"
def integrity_error(s,raw):
 if not isinstance(s.get('result_bytes'),int) or s['result_bytes']!=len(raw):return 'result_bytes_mismatch'
 d=hashlib.sha256(raw).hexdigest();return None if s.get('result_sha256')==d else 'result_sha256_mismatch'
def delete_pod(pod,evidence,token,sup):
 for n in range(1,4):
  emit('VELA_POD_DELETE_ATTEMPT',pod_id=pod,attempt=n)
  try:r=sh(['runpodctl','pod','delete',pod],capture_output=True,timeout=30);out=redact(r.stdout or '',token,sup);err=redact(r.stderr or '',token,sup);(evidence/'pod-delete.stdout').write_text(out);(evidence/'pod-delete.stderr').write_text(err)
  except Exception as e:r=None;(evidence/'pod-delete.stderr').write_text(f'DELETE_EXCEPTION:{type(e).__name__}:{e}\n');err=''
  if r is not None and (r.returncode==0 or '"code":"not_found"' in err or '"code": "not_found"' in err):emit('VELA_POD_DELETE_DONE',pod_id=pod,deleted=True,exit_code=r.returncode);return True
  if n<3:time.sleep(2)
 emit('VELA_POD_DELETE_DONE',pod_id=pod,deleted=False,exit_code=-1);return False
def main():
 a=argparse.ArgumentParser();a.add_argument('--dry-run',action='store_true');args=a.parse_args();need=('VELA_SOURCE_COMMIT','VELA_MODEL_PROFILE_ID','VELA_RETRIEVAL_MODE','VELA_PROBLEM_PACK_REL','VELA_EXECUTION_CONFIG_REL')
 for k in need:
  if not os.environ.get(k):raise SystemExit(f'{k} required')
 source=os.environ['VELA_SOURCE_COMMIT'];model=os.environ['VELA_MODEL_PROFILE_ID'];retrieval=os.environ['VELA_RETRIEVAL_MODE']
 if len(source)!=40:raise SystemExit('exact source SHA required')
 pack=safe_rel(os.environ['VELA_PROBLEM_PACK_REL']);cfg=safe_rel(os.environ['VELA_EXECUTION_CONFIG_REL']);ev=Path(os.environ.get('VELA_EVIDENCE_DIR','/tmp/vela-statebank'));ev.mkdir(parents=True,exist_ok=True)
 plan={'source_commit':source,'model_profile_id':model,'retrieval_mode':retrieval,'problem_pack_rel':pack,'execution_config_rel':cfg,'max_pod_count':1,'model_selection_cardinality':1,'provider':'RUNPOD','gpu':'NVIDIA A40','image':PINNED_IMAGE};(ev/'execution-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
 if args.dry_run:print(json.dumps(plan,sort_keys=True));return
 if os.environ.get('VELA_PAID_EXECUTION_AUTHORIZED')!='true':raise SystemExit('paid execution not authorized')
 if not os.environ.get('RUNPOD_API_KEY'):raise SystemExit('RUNPOD_API_KEY required')
 max_runtime=int(os.environ.get('VELA_MAX_RUNTIME_SECONDS','900'));port=8000;sh(['runpodctl','user'],stdout=subprocess.DEVNULL,check=True);gpu=ev/'gpu-list.json';gpu.write_text(sh(['runpodctl','gpu','list','--include-unavailable'],capture_output=True,check=True).stdout);selp=ev/'selection.json';sh([sys.executable,str(Path(__file__).with_name('runpod_select_a40_capacity_v1.py')),str(gpu),str(selp)],check=True);sel=json.loads(selp.read_text());dc=sel['selectedDataCenter']['dataCenterId'];price=float(sel['securePricePerHr']);emit('VELA_PROVIDER_SELECTION',data_center=dc,stock=sel['selectedDataCenter'].get('stockStatus'),price_per_hr=price)
 sup=base64.b64encode(Path(__file__).with_name('runpod_statebank_supervisor_v1.py').read_bytes()).decode();token=secrets.token_hex(32);(ev/'result-token-sha256.txt').write_text(hashlib.sha256(token.encode()).hexdigest()+'\n');env={'VELA_SUPERVISOR_B64':sup,'VELA_RESULT_TOKEN':token,'VELA_SOURCE_COMMIT':source,'VELA_BASE_RAW':f'https://raw.githubusercontent.com/jeonghun917/opened-arm/{source}/experiments/vela-active-state/r2-configurable-statebank-benchmark-v1','VELA_MODEL_PROFILE_ID':model,'VELA_RETRIEVAL_MODE':retrieval,'VELA_PROBLEM_PACK_REL':pack,'VELA_EXECUTION_CONFIG_REL':cfg,'VELA_RESULT_PORT':str(port),'VELA_WORKLOAD_TIMEOUT_SECONDS':str(max_runtime-60),'PYTHONUNBUFFERED':'1'};scale=model.split('-')[-2];name=f'vela-statebank-{scale}-{source[:10]}';image_ref(ev);cmd=['runpodctl','pod','create','--name',name,'--image',PINNED_IMAGE,'--gpu-id','NVIDIA A40','--gpu-count','1','--cloud-type','SECURE','--data-center-ids',dc,'--container-disk-in-gb','50','--ports',f'{port}/http','--ssh=false','--min-cuda-version','12.6','--terminate-after','15m','--env',json.dumps(env,separators=(',',':')),'--docker-args',"sh -lc 'echo \"$VELA_SUPERVISOR_B64\" | base64 -d > /tmp/vela_supervisor.py && exec python /tmp/vela_supervisor.py'"]
 emit('VELA_POD_CREATE_START',name=name,image=PINNED_IMAGE);create=sh(cmd,capture_output=True);save_obj(create.stdout or '',ev/'pod-create.json',token,sup);(ev/'pod-create.stderr').write_text(redact(create.stderr or '',token,sup))
 if create.returncode:raise SystemExit(create.returncode)
 created=json.loads(create.stdout);pod=str(created.get('id') or created.get('podId') or '')
 if not pod:raise SystemExit('pod id missing')
 (ev/'pod-id.txt').write_text(pod+'\n');proxy=f'https://{pod}-{port}.proxy.runpod.net';start=time.time();found=False;terminal='';deleted=False;first_running=None;last_u=None;last_report=0.;last_phase=None;last_diag=0.;sig=None;old={}
 def on_signal(n,_frame):raise ControllerSignal(n)
 for s in (signal.SIGINT,signal.SIGTERM):old[s]=signal.getsignal(s);signal.signal(s,on_signal)
 emit('VELA_POD_CREATED',pod_id=pod,proxy_host=f'{pod}-{port}.proxy.runpod.net')
 try:
  while time.time()-start<max_runtime:
   now=time.time();runtime,u,ctl=pod_get(pod,ev,token,sup)
   if ctl:terminal=ctl;logs(pod,ev,token,sup,'pod-control-failure');break
   if runtime=='running' and first_running is None:first_running=now;emit('VELA_POD_RUNNING',pod_id=pod,elapsed_seconds=int(now-start))
   if runtime in {'failed','exited','stopped','terminated'}:terminal=f'POD_RUNTIME_TERMINAL_{runtime.upper()}';logs(pod,ev,token,sup,'pod-runtime-terminal');break
   if u is not None:
    if last_u is not None and last_u>=20 and u+10<last_u:terminal='CONTAINER_RESTART_DETECTED_BY_CONTROL_PLANE';logs(pod,ev,token,sup,'container-restart');break
    last_u=u
   code,st=request_json(f'{proxy}/v1/status/{token}');phase=st.get('phase') if isinstance(st,dict) else None;state=st.get('state') if isinstance(st,dict) else None
   if code==200:
    bad=status_error(st,source,model,retrieval)
    if bad:terminal='PROXY_STATUS_IDENTITY_MISMATCH';(ev/'terminal-status.json').write_text(json.dumps(st,indent=2,sort_keys=True)+'\n');emit('VELA_STATUS_IDENTITY_FAILURE',detail=bad);break
    (ev/'last-status.json').write_text(json.dumps(st,indent=2,sort_keys=True)+'\n')
    if phase!=last_phase or now-last_report>=15:emit('VELA_POLL',elapsed_seconds=int(now-start),http=code,state=state,phase=phase,runtime_status=runtime,uptime_seconds=u);last_report=now;last_phase=phase
    if state=='FAILED':terminal='SUPERVISOR_FAILED';logs(pod,ev,token,sup,'failure');break
    if state=='COMPLETE':
     rc,raw=fetch_result(f'{proxy}/v1/result/{token}',ev/'benchmark_result.json');bad=integrity_error(st,raw) if rc==200 else f'http_{rc}'
     if bad:terminal='RESULT_INTEGRITY_OR_FETCH_FAILURE';emit('VELA_RESULT_INTEGRITY_FAILURE',detail=bad);logs(pod,ev,token,sup,'result-failure');break
     obj=json.loads(raw)
     if obj.get('schema')!='vela-statebank-benchmark-result:v1' or obj.get('source_commit')!=source or obj.get('model_profile',{}).get('id')!=model or obj.get('retrieval_mode')!=retrieval:terminal='RESULT_IDENTITY_MISMATCH';break
     found=True;terminal='RESULT_FETCHED';break
   else:
    if now-last_report>=15:emit('VELA_POLL',elapsed_seconds=int(now-start),http=code,state='UNREACHABLE',phase='proxy_wait',runtime_status=runtime,uptime_seconds=u);last_report=now
    if first_running is not None and now-first_running>=30 and now-last_diag>=60:logs(pod,ev,token,sup,f'proxy-debug-{int(now-start)}s');last_diag=now
    if first_running is not None and now-first_running>=PROXY_STARTUP_GRACE_SECONDS:terminal='PROXY_UNREACHABLE_AFTER_RUNNING_GRACE';logs(pod,ev,token,sup,'proxy-start-timeout');break
   time.sleep(POLL_INTERVAL_SECONDS)
  if not terminal:terminal='MAX_RUNTIME_EXPIRED';logs(pod,ev,token,sup,'max-runtime')
 except ControllerSignal as e:sig=e.signum;terminal='CONTROLLER_CANCELLED';logs(pod,ev,token,sup,'controller-cancelled')
 finally:
  for s,h in old.items():signal.signal(s,h)
  emit('VELA_POD_DELETE_START',pod_id=pod,terminal_reason=terminal);deleted=delete_pod(pod,ev,token,sup)
 elapsed=min(max_runtime,max(0,int(time.time()-start)));out={'status':'PASS' if found and deleted else 'FAIL','scientific_evidence':bool(found),'source_commit':source,'model_profile_id':model,'retrieval_mode':retrieval,'pod_id':pod,'pod_deleted':deleted,'terminal_reason':terminal,'elapsed_seconds':elapsed,'max_runtime_seconds':max_runtime,'secure_price_per_hr_usd':price,'estimated_gpu_cost_upper_bound_usd':round(elapsed*price/3600,6),'data_center_id':dc,'max_pod_count':1,'model_selection_cardinality':1,'controller_signal':sig};(ev/'provider-envelope.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,sort_keys=True),flush=True)
 if sig is not None:raise SystemExit(128+sig)
 raise SystemExit(0 if out['status']=='PASS' else 1)
if __name__=='__main__':main()
