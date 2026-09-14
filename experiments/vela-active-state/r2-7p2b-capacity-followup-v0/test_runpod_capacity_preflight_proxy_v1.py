#!/usr/bin/env python3
import hashlib,importlib.util,json,sys,tempfile,threading,urllib.error,urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

def main():
 p=Path(__file__).with_name('runpod_capacity_preflight_supervisor_v1.py'); spec=importlib.util.spec_from_file_location('_vela_preflight_proxy',p); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
 root=Path(tempfile.mkdtemp()); token='a'*64; source='d'*40; ctx=mod.Ctx(root,token,source); mod.update(ctx,'RUNNING','RUN_PREFLIGHT','x'); server=ThreadingHTTPServer(('127.0.0.1',0),mod.handler(ctx)); threading.Thread(target=server.serve_forever,daemon=True).start(); base=f'http://127.0.0.1:{server.server_port}'
 s=json.load(urllib.request.urlopen(f'{base}/v1/status/{token}')); assert s['state']=='RUNNING'
 try: urllib.request.urlopen(f'{base}/v1/status/wrong')
 except urllib.error.HTTPError as e: assert e.code==404
 else: raise AssertionError('wrong token accepted')
 result={'status':'PASS','scientific_evidence':False,'purpose':'P-R2-03_RESULT_BLIND_CAPACITY_PREFLIGHT','source_commit':source,'matched_profile':'g1i-20260805','runtime':{'device':'NVIDIA A40','device_capability':[8,6]},'models':[]}; raw=(json.dumps(result,separators=(',',':'),sort_keys=True)+'\n').encode(); ctx.result.write_bytes(raw); mod.update(ctx,'COMPLETE','RESULT_READY','x',result_bytes=len(raw),result_sha256=hashlib.sha256(raw).hexdigest(),scientific_evidence=False); assert urllib.request.urlopen(f'{base}/v1/result/{token}').read()==raw; server.shutdown(); print('RUNPOD_CAPACITY_PREFLIGHT_PROXY_SYNTHETIC_PASS')
if __name__=='__main__':main()
