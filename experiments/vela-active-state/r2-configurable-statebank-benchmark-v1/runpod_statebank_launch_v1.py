#!/usr/bin/env python3
from __future__ import annotations
import argparse,base64,hashlib,json,os,secrets,subprocess,sys,time,urllib.error,urllib.request
from pathlib import Path
PINNED_IMAGE='pytorch/pytorch@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3'
def sh(cmd,**kw):return subprocess.run(cmd,text=True,**kw)
def emit(tag,**data):print(json.dumps({'tag':tag,'unix':int(time.time()),**data},sort_keys=True),flush=True)
def safe_rel(v):
 p=Path(v)
 if not v or p.is_absolute() or '..' in p.parts or '\\' in v:raise ValueError(f'unsafe relative path {v}')
 return p.as_posix()
def request_json(url,timeout=6):
 try:
  with urllib.request.urlopen(url,timeout=timeout) as r:return r.status,json.loads(r.read())
 except urllib.error.HTTPError as e:
  try:b=json.loads(e.read())
  except Exception:b={}
  return e.code,b
 except Exception:return 0,{}
def fetch_result(url,path):
 try:
  with urllib.request.urlopen(url,timeout=15) as r:raw=r.read();path.write_bytes(raw);return r.status,raw
 except Exception:return 0,b''
def image_ref(evidence):
 (evidence/'image-ref.txt').write_text(PINNED_IMAGE+'\n');return PINNED_IMAGE
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');a=ap.parse_args();keys=('VELA_SOURCE_COMMIT','VELA_MODEL_PROFILE_ID','VELA_RETRIEVAL_MODE','VELA_PROBLEM_PACK_REL','VELA_EXECUTION_CONFIG_REL')
 for k in keys:
  if not os.environ.get(k):raise SystemExit(f'{k} required')
 if len(os.environ['VELA_SOURCE_COMMIT'])!=40:raise SystemExit('exact source SHA required')
 pack=safe_rel(os.environ['VELA_PROBLEM_PACK_REL']);cfg=safe_rel(os.environ['VELA_EXECUTION_CONFIG_REL']);evidence=Path(os.environ.get('VELA_EVIDENCE_DIR','/tmp/vela-statebank'));evidence.mkdir(parents=True,exist_ok=True);plan={'source_commit':os.environ['VELA_SOURCE_COMMIT'],'model_profile_id':os.environ['VELA_MODEL_PROFILE_ID'],'retrieval_mode':os.environ['VELA_RETRIEVAL_MODE'],'problem_pack_rel':pack,'execution_config_rel':cfg,'max_pod_count':1,'model_selection_cardinality':1,'provider':'RUNPOD','gpu':'NVIDIA A40','image':PINNED_IMAGE};(evidence/'execution-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
 if a.dry_run:print(json.dumps(plan,sort_keys=True));return
 if os.environ.get('VELA_PAID_EXECUTION_AUTHORIZED')!='true':raise SystemExit('paid execution not authorized')
 if not os.environ.get('RUNPOD_API_KEY'):raise SystemExit('RUNPOD_API_KEY required')
 max_runtime=int(os.environ.get('VELA_MAX_RUNTIME_SECONDS','900'));result_port=8000;sh(['runpodctl','user'],stdout=subprocess.DEVNULL,check=True);gpu=evidence/'gpu-list.json';gpu.write_text(sh(['runpodctl','gpu','list','--include-unavailable'],capture_output=True,check=True).stdout);selection=evidence/'selection.json';sh([sys.executable,str(Path(__file__).with_name('runpod_select_a40_capacity_v1.py')),str(gpu),str(selection)],check=True);sel=json.loads(selection.read_text());dc=sel['selectedDataCenter']['dataCenterId'];price=float(sel['securePricePerHr']);emit('VELA_PROVIDER_SELECTION',data_center=dc,stock=sel['selectedDataCenter'].get('stockStatus'),price_per_hr=price)
 sup=base64.b64encode(Path(__file__).with_name('runpod_statebank_supervisor_v1.py').read_bytes()).decode();token=secrets.token_hex(32);(evidence/'result-token-sha256.txt').write_text(hashlib.sha256(token.encode()).hexdigest()+'\n');env={'VELA_SUPERVISOR_B64':sup,'VELA_RESULT_TOKEN':token,'VELA_SOURCE_COMMIT':os.environ['VELA_SOURCE_COMMIT'],'VELA_BASE_RAW':f"https://raw.githubusercontent.com/jeonghun917/opened-arm/{os.environ['VELA_SOURCE_COMMIT']}/experiments/vela-active-state/r2-configurable-statebank-benchmark-v1",'VELA_MODEL_PROFILE_ID':os.environ['VELA_MODEL_PROFILE_ID'],'VELA_RETRIEVAL_MODE':os.environ['VELA_RETRIEVAL_MODE'],'VELA_PROBLEM_PACK_REL':pack,'VELA_EXECUTION_CONFIG_REL':cfg,'VELA_RESULT_PORT':str(result_port),'VELA_WORKLOAD_TIMEOUT_SECONDS':str(max_runtime-60),'PYTHONUNBUFFERED':'1'};scale=os.environ['VELA_MODEL_PROFILE_ID'].split('-')[-2];name=f"vela-statebank-{scale}-{os.environ['VELA_SOURCE_COMMIT'][:10]}";cmd=['runpodctl','pod','create','--name',name,'--image',image_ref(evidence),'--gpu-id','NVIDIA A40','--gpu-count','1','--cloud-type','SECURE','--data-center-ids',dc,'--container-disk-in-gb','50','--ports',f'{result_port}/http','--ssh=false','--min-cuda-version','12.6','--terminate-after','15m','--env',json.dumps(env,separators=(',',':')),'--docker-args',"sh -lc 'echo \"$VELA_SUPERVISOR_B64\" | base64 -d > /tmp/vela_supervisor.py && exec python /tmp/vela_supervisor.py'"];emit('VELA_POD_CREATE_START',name=name,image=PINNED_IMAGE);create=sh(cmd,capture_output=True);(evidence/'pod-create.stdout').write_text(create.stdout);(evidence/'pod-create.stderr').write_text(create.stderr)
 if create.returncode!=0:emit('VELA_POD_CREATE_FAILED',exit_code=create.returncode,stderr=create.stderr[-1000:]);raise SystemExit(create.returncode)
 try:created=json.loads(create.stdout)
 except Exception:raise SystemExit('pod create output invalid json')
 pod=str(created.get('id') or created.get('podId') or '')
 if not pod:raise SystemExit('pod id missing')
 (evidence/'pod-id.txt').write_text(pod+'\n');proxy=f'https://{pod}-{result_port}.proxy.runpod.net';start=time.time();found=False;terminal='';deleted=False;proxy_seen=False;last_report=0.0;last_phase=None
 emit('VELA_POD_CREATED',pod_id=pod,proxy_host=f'{pod}-{result_port}.proxy.runpod.net')
 try:
  while time.time()-start<max_runtime:
   code,st=request_json(f'{proxy}/v1/status/{token}')
   now=time.time();phase=st.get('phase') if isinstance(st,dict) else None;state=st.get('state') if isinstance(st,dict) else None
   if code==200:
    proxy_seen=True;(evidence/'last-status.json').write_text(json.dumps(st,indent=2,sort_keys=True)+'\n')
    if phase!=last_phase or now-last_report>=15:emit('VELA_POLL',elapsed_seconds=int(now-start),http=code,state=state,phase=phase);last_report=now;last_phase=phase
    if state=='FAILED':terminal='SUPERVISOR_FAILED';break
    if state=='COMPLETE':
     rc,raw=fetch_result(f'{proxy}/v1/result/{token}',evidence/'benchmark_result.json')
     if rc==200:
      obj=json.loads(raw);assert obj['schema']=='vela-statebank-benchmark-result:v1' and obj['source_commit']==os.environ['VELA_SOURCE_COMMIT'] and obj['model_profile']['id']==os.environ['VELA_MODEL_PROFILE_ID'] and obj['retrieval_mode']==os.environ['VELA_RETRIEVAL_MODE'];found=True;terminal='RESULT_FETCHED';break
     terminal=f'RESULT_FETCH_FAILED_{rc}';break
   else:
    if now-last_report>=15:emit('VELA_POLL',elapsed_seconds=int(now-start),http=code,state='UNREACHABLE',phase='proxy_wait');last_report=now
    if not proxy_seen and now-start>=120:terminal='PROXY_START_TIMEOUT';break
   time.sleep(3)
  if not terminal:terminal='MAX_RUNTIME_EXPIRED'
 finally:
  emit('VELA_POD_DELETE_START',pod_id=pod,terminal_reason=terminal);d=sh(['runpodctl','pod','delete',pod],capture_output=True);(evidence/'pod-delete.stdout').write_text(d.stdout);(evidence/'pod-delete.stderr').write_text(d.stderr);deleted=(d.returncode==0 or 'not_found' in d.stderr);emit('VELA_POD_DELETE_DONE',pod_id=pod,deleted=deleted,exit_code=d.returncode)
 elapsed=min(max_runtime,max(0,int(time.time()-start)));out={'status':'PASS' if found and deleted else 'FAIL','scientific_evidence':bool(found),'source_commit':os.environ['VELA_SOURCE_COMMIT'],'model_profile_id':os.environ['VELA_MODEL_PROFILE_ID'],'retrieval_mode':os.environ['VELA_RETRIEVAL_MODE'],'pod_id':pod,'pod_deleted':deleted,'terminal_reason':terminal,'elapsed_seconds':elapsed,'max_runtime_seconds':max_runtime,'secure_price_per_hr_usd':price,'estimated_gpu_cost_upper_bound_usd':round(elapsed*price/3600,6),'data_center_id':dc,'max_pod_count':1,'model_selection_cardinality':1};(evidence/'provider-envelope.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,sort_keys=True),flush=True);raise SystemExit(0 if out['status']=='PASS' else 1)
if __name__=='__main__':main()
