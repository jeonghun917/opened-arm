from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import random
import traceback
from pathlib import Path

import torch

BASE=Path(__file__).resolve().parents[1]
V3_PATH=BASE/"learned-upgrade-v3"/"mamba_upgrade_migration.py"
spec=importlib.util.spec_from_file_location("v3",V3_PATH)
if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {V3_PATH}")
v3=importlib.util.module_from_spec(spec);spec.loader.exec_module(v3)

SEED=917
LR=1e-4
EPOCHS=3
HISTORY=v3.MIGRATION_HISTORY
FUTURE=(" Additional telemetry arrives without resolving verification. Project Orion remains active and the codeword remains BETA. "
        "External action stays blocked. More telemetry arrives, but hypothesis two remains unresolved and verification stays incomplete.")
PROBES=[
 {"id":"codeword","suffix":"\nCurrent codeword:","candidates":[" BETA"," ALPHA"],"expected":" BETA"},
 {"id":"verification","suffix":"\nVerification status:","candidates":[" incomplete"," complete"],"expected":" incomplete"},
 {"id":"project","suffix":"\nProject Orion status:","candidates":[" active"," paused"],"expected":" active"},
 {"id":"action","suffix":"\nExternal action is:","candidates":[" blocked"," allowed"],"expected":" blocked"},
]


def write_report(x):
 pth=os.environ.get("VELA_RESULT_PATH")
 if pth:
  p=Path(pth);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps(x,ensure_ascii=False,indent=2))

def clone_cache(x):
 if x is None:return None
 b=io.BytesIO();torch.save(x,b);b.seek(0);return torch.load(b,map_location="cpu",weights_only=False)

def tensor_refs(obj):
 out=[];seen=set()
 def walk(x):
  if torch.is_tensor(x):
   if id(x) not in seen:seen.add(id(x));out.append(x)
  elif isinstance(x,dict):
   for v in x.values():walk(v)
  elif isinstance(x,(list,tuple)):
   for v in x:walk(v)
  elif hasattr(x,"__dict__"):
   for v in vars(x).values():walk(v)
 walk(obj);return out

def dist(a,b):
 aa,bb=tensor_refs(a),tensor_refs(b);sq=0.;mx=0.;n=0
 for x,y in zip(aa,bb):
  d=x.detach().float().cpu()-y.detach().float().cpu();sq+=float((d*d).sum());mx=max(mx,float(d.abs().max()));n+=d.numel()
 return {"rms":math.sqrt(sq/max(n,1)),"l2":math.sqrt(sq),"max_abs":mx,"numel":n}

def run_ids(model,ids,cache,pos):
 out=None
 for j in range(ids.shape[1]):
  out=model(ids[:,j:j+1],cache_params=cache,cache_position=torch.tensor([pos+j]),use_cache=True,return_dict=True);cache=out.cache_params
 return clone_cache(cache)

def score_candidate(model,tok,state,pos,suffix,cand):
 cache=clone_cache(state);s=tok(suffix,return_tensors="pt",add_special_tokens=False).input_ids;c=tok(cand,return_tensors="pt",add_special_tokens=False).input_ids;logits=None
 with torch.no_grad():
  for j in range(s.shape[1]):
   o=model(s[:,j:j+1],cache_params=cache,cache_position=torch.tensor([pos]),use_cache=True,return_dict=True);cache=o.cache_params;logits=o.logits[:,-1].float();pos+=1
  total=0.
  for j in range(c.shape[1]):
   lp=torch.log_softmax(logits,dim=-1);tid=int(c[0,j]);total+=float(lp[0,tid])
   if j+1<c.shape[1]:
    o=model(c[:,j:j+1],cache_params=cache,cache_position=torch.tensor([pos]),use_cache=True,return_dict=True);cache=o.cache_params;logits=o.logits[:,-1].float();pos+=1
 return total

def probe(model,tok,state,pos):
 rows=[]
 for fx in PROBES:
  scores={c:score_candidate(model,tok,state,pos,fx["suffix"],c) for c in fx["candidates"]};chosen=max(scores,key=scores.get)
  rows.append({"id":fx["id"],"chosen":chosen,"expected":fx["expected"],"correct":chosen==fx["expected"],"scores":scores})
 return rows

def compare(rows,native):
 nb={r["id"]:r for r in native};agree=sum(r["chosen"]==nb[r["id"]]["chosen"] for r in rows);correct=sum(r["correct"] for r in rows)
 return {"decision_agreement":agree/len(rows),"expected_accuracy":correct/len(rows)}

def set_interp(trainable,w1,w2,alpha):
 with torch.no_grad():
  for name,p in trainable:p.copy_(w1[name]*(1-alpha)+w2[name]*alpha)

def train_w2(model,tok):
 train=[]
 for p in model.parameters():p.requires_grad_(False)
 for name,p in model.named_parameters():
  if ".mixer.x_proj.weight" in name:p.requires_grad_(True);train.append((name,p))
 w1={n:p.detach().clone() for n,p in train};opt=torch.optim.AdamW([p for _,p in train],lr=LR,weight_decay=0.0);order=list(range(len(v3.TRAIN)));losses=[]
 for e in range(EPOCHS):
  random.Random(SEED+e).shuffle(order);tot=0.;model.train()
  for idx in order:
   prompt,gold=v3.TRAIN[idx];opt.zero_grad(set_to_none=True);loss=v3.train_loss(model,tok,prompt,gold);loss.backward();torch.nn.utils.clip_grad_norm_([p for _,p in train],1.0);opt.step();tot+=float(loss.detach())
  losses.append(tot/len(order))
 model.eval();w2={n:p.detach().clone() for n,p in train};return train,w1,w2,losses

def gradual_path(model,ids_future,start_state,start_pos,train,w1,w2,stages):
 state=clone_cache(start_state);N=int(ids_future.shape[1]);bounds=[round(i*N/stages) for i in range(stages+1)];trace=[]
 for i in range(stages):
  alpha=(i+1)/stages;set_interp(train,w1,w2,alpha);a,b=bounds[i],bounds[i+1]
  if b>a:
   with torch.no_grad():state=run_ids(model,ids_future[:,a:b],state,start_pos+a)
  trace.append({"stage":i+1,"alpha":alpha,"future_tokens_seen":b})
 set_interp(train,w1,w2,1.0);return state,trace

def run():
 tv=None
 try:
  from transformers import AutoModelForCausalLM,AutoTokenizer,__version__ as tv
  torch.manual_seed(SEED);random.seed(SEED);tok=AutoTokenizer.from_pretrained(v3.MODEL_ID);model=AutoModelForCausalLM.from_pretrained(v3.MODEL_ID,torch_dtype=torch.float32).cpu().eval()
  hids=tok(HISTORY,return_tensors="pt",add_special_tokens=False).input_ids;T=int(hids.shape[1]);fids=tok(FUTURE,return_tensors="pt",add_special_tokens=False).input_ids;F=int(fids.shape[1])
  prefix_before="Project Orion remains active. The old codeword was ALPHA. "
  prefix_after=prefix_before+"Correction: the current codeword is BETA, not ALPHA. "
  before_idx=int(tok(prefix_before,return_tensors="pt",add_special_tokens=False).input_ids.shape[1]);after_idx=int(tok(prefix_after,return_tensors="pt",add_special_tokens=False).input_ids.shape[1])
  with torch.no_grad():
   s1=clone_cache(model(hids,use_cache=True,return_dict=True).cache_params)
   anchor_before=clone_cache(model(hids[:,:before_idx],use_cache=True,return_dict=True).cache_params)
   anchor_after=clone_cache(model(hids[:,:after_idx],use_cache=True,return_dict=True).cache_params)
  baseline=v3.evaluate(model,tok);train,w1,w2,losses=train_w2(model,tok);after=v3.evaluate(model,tok)
  set_interp(train,w1,w2,1.0)
  with torch.no_grad():
   w2_native=clone_cache(model(hids,use_cache=True,return_dict=True).cache_params)
   w2_native_future=run_ids(model,fids,clone_cache(w2_native),T)
  native_probe=probe(model,tok,w2_native_future,T+F)

  # E9: abrupt versus gradual online weight morph, same real future tokens.
  modes=[]
  set_interp(train,w1,w2,1.0)
  with torch.no_grad():abrupt=run_ids(model,fids,clone_cache(s1),T)
  modes.append({"mode":"abrupt_W2","state_error":dist(abrupt,w2_native_future),"probe":compare(probe(model,tok,abrupt,T+F),native_probe)})
  for stages in [2,4,8]:
   set_interp(train,w1,w2,0.0);state,trace=gradual_path(model,fids,s1,T,train,w1,w2,stages)
   modes.append({"mode":f"gradual_{stages}_stages","trace":trace,"state_error":dist(state,w2_native_future),"probe":compare(probe(model,tok,state,T+F),native_probe)})

  # E10: exact causal-anchor replay around the correction event.
  set_interp(train,w1,w2,1.0);anchors=[]
  for name,idx,state in [("before_correction",before_idx,anchor_before),("after_correction",after_idx,anchor_after),("current_no_replay",T,s1)]:
   with torch.no_grad():m=clone_cache(state) if idx==T else run_ids(model,hids[:,idx:],clone_cache(state),idx)
   anchors.append({"anchor":name,"anchor_token":idx,"replayed_tokens":T-idx,"state_error":dist(m,w2_native),"probe":compare(probe(model,tok,m,T),probe(model,tok,w2_native,T)),"probe_rows":probe(model,tok,m,T)})
  anchors.append({"anchor":"full_W2_replay_control","anchor_token":0,"replayed_tokens":T,"state_error":dist(w2_native,w2_native),"probe":compare(probe(model,tok,w2_native,T),probe(model,tok,w2_native,T))})

  write_report({"status":"VELA_GRADUAL_AND_CAUSAL_ANCHOR_UPGRADE_PROBE","model":v3.MODEL_ID,"torch_version":torch.__version__,"transformers_version":tv,"capability":{"baseline_accuracy":baseline["accuracy"],"after_accuracy":after["accuracy"],"baseline_correction":baseline["correction_accuracy"],"after_correction":after["correction_accuracy"],"control_before":baseline["control_accuracy"],"control_after":after["control_accuracy"],"epoch_loss":losses},"E9_gradual_weight_morph":{"history_tokens":T,"future_tokens":F,"native_probe":native_probe,"modes":modes},"E10_causal_anchor_replay":{"before_correction_token":before_idx,"after_correction_token":after_idx,"rows":anchors},"claim_boundary":"Narrow same-architecture Mamba learned-upgrade experiment. Gradual morph preserves the actual recurrent state while weights change during real future computation; anchor replay re-executes exact causal tokens under W2 and is not declarative record reinjection. Neither establishes personal identity or general reasoning."})
 except BaseException as e:
  write_report({"status":"VELA_GRADUAL_UPGRADE_ERROR","model":v3.MODEL_ID,"error_type":type(e).__name__,"error":str(e),"traceback_tail":traceback.format_exc().splitlines()[-40:]});raise

if __name__=="__main__":run()
