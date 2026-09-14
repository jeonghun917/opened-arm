#!/usr/bin/env python3
import argparse, json
from pathlib import Path

GPU_ID='NVIDIA A40'
PRICE_CAP=0.49
MIN_MEMORY_GB=48
STOCK_PRIORITY=('High','Medium','Low')
ALLOWED_STOCK=set(STOCK_PRIORITY)

def load_rows(path):
 data=json.loads(Path(path).read_text())
 if isinstance(data,list): return data
 if isinstance(data,dict):
  for k in ('gpus','items'):
   if isinstance(data.get(k),list): return data[k]
 raise SystemExit('GPU_INVENTORY_SHAPE_UNSUPPORTED')

def select_capacity(rows,required_dc=None):
 gpu=next((x for x in rows if isinstance(x,dict) and x.get('gpuId')==GPU_ID),None)
 if gpu is None: raise SystemExit('A40_NOT_LISTED')
 if gpu.get('secureCloud') is not True or gpu.get('available') is not True:
  raise SystemExit('A40_NOT_AVAILABLE_SECURE')
 price=gpu.get('securePricePerHr')
 if not isinstance(price,(int,float)) or float(price)>PRICE_CAP:
  raise SystemExit(f'A40_PRICE_CAP_EXCEEDED:{price}')
 if int(gpu.get('memoryInGb') or 0)<MIN_MEMORY_GB:
  raise SystemExit(f'A40_MEMORY_DRIFT:{gpu.get("memoryInGb")}')
 dcs=gpu.get('dataCenterAvailability')
 if not isinstance(dcs,list):
  raise SystemExit('A40_DATACENTER_LIST_MISSING')
 norm=[]
 for dc in dcs:
  if not isinstance(dc,dict): continue
  dcid=dc.get('dataCenterId'); stock=dc.get('stockStatus')
  if not isinstance(dcid,str) or not dcid:
   raise SystemExit('A40_DATACENTER_SCHEMA_DRIFT:dataCenterId')
  norm.append({'dataCenterId':dcid,'stockStatus':stock})
 if required_dc:
  chosen=next((x for x in norm if x['dataCenterId']==required_dc),None)
  if chosen is None:
   raise SystemExit(f'DATACENTER_NOT_LISTED:{required_dc}')
  if chosen['stockStatus'] not in ALLOWED_STOCK:
   raise SystemExit(f'DATACENTER_STOCK_NOT_SUFFICIENT:{required_dc}={chosen["stockStatus"]}')
 else:
  chosen=None
  for stock in STOCK_PRIORITY:
   cand=sorted((x for x in norm if x['stockStatus']==stock),key=lambda x:x['dataCenterId'])
   if cand:
    chosen=cand[0]
    break
  if chosen is None:
   raise SystemExit('NO_A40_AVAILABLE_DATACENTER')
 return {
  'status':'PASS',
  'paid_mutation':False,
  'gpuId':GPU_ID,
  'displayName':gpu.get('displayName'),
  'memoryInGb':gpu.get('memoryInGb'),
  'secureCloud':gpu.get('secureCloud'),
  'securePricePerHr':price,
  'selectedDataCenter':chosen,
  'inventoryStockStatus':gpu.get('stockStatus')
 }

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('inventory')
 ap.add_argument('output')
 ap.add_argument('--require-data-center')
 a=ap.parse_args()
 r=select_capacity(load_rows(a.inventory),a.require_data_center)
 Path(a.output).write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 print(json.dumps(r,sort_keys=True))

if __name__=='__main__':
 main()
