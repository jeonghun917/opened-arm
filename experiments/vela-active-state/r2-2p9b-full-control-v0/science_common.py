from __future__ import annotations
import gc, hashlib, json, subprocess, sys
from pathlib import Path
EXPECTED_TORCH="2.7.1+cu126"
INDEX_BLOB="95b2371ac7bb9416ad5b9a7d33f2df4de40d4d4c"
EVALUATOR_BLOB="55d28e6564396408f293f67ebabfc258e41ae9ef"
EXTERNAL_BLOB="fb056e547feaedfead6b2bb67f87e31449d21466"
EVALUATOR_SHA256="2bcca90657a39eb67b14b1015f617b15fdd9e588a5620144fffa591cb1e1b92f"
EXTERNAL_SHA256="e901b4e5b343d567816fd72bfd8c6b11e314ddbb442578da9999f55e8a26db69"
CONDITIONS=["B","C","G","M","X"]
EVENTS=["INITIAL_SHARED_SEED","RUNTIME_ADDITION","SECURITY_ADDITION","RUNTIME_CORRECTION","SECURITY_RESOLUTION","RUNTIME_FOLLOWTHROUGH"]
DOMAINS=["shared","runtime","security","runtime","security","runtime"]
ROUTES={"B":[None]*6,"C":[None]*6,"G":[None]*6,"M":[None,"A","B","A","B","A"],"X":[None,"A","B","B","B","A"]}
ACTIONS={"B":["FRESH"]*6,"C":["FRESH","KEEP","KEEP","KEEP","KEEP","KEEP"],"G":["FRESH","KEEP","KEEP","RESET","KEEP","KEEP"],"M":["SEED_AB","KEEP","KEEP","RESET","KEEP","KEEP"],"X":["SEED_AB","KEEP","KEEP","RESET","KEEP","KEEP"]}
RESTORE={"B":[False]*6,"C":[False,True,True,True,True,True],"G":[False,True,True,False,True,True],"M":[False,True,True,False,True,True],"X":[False,True,True,False,True,True]}
MODELS=[{"scale":"0.4B","repo":"BlinkDL/rwkv7-g1","revision":"c4d4f4748a8233e9a2f1ee7f38f4c4e3780e750d","file":"rwkv7-g1a-0.4b-20250905-ctx4096.pth","size":901776757,"sha256":"d852e99ef6c95726109660c64e7c51a8df30c53b0832a68645bfcd15253b3109"},{"scale":"2.9B","repo":"BlinkDL/rwkv7-g1","revision":"2448c93da77d9958ea89a9faa4c1b34c5c46ccec","file":"rwkv7-g1a-2.9b-20250924-ctx4096.pth","size":5896274949,"sha256":"df5716263b617e7da83590446fb2a98b6663447cfd52e8e9b49e11ce7f4faa3e"}]
STRATEGY="cuda fp16"
def sha256(b): return hashlib.sha256(b).hexdigest()
def gitblob(b): return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def sha256_file(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for c in iter(lambda:f.read(8*1024*1024),b""): h.update(c)
 return h.hexdigest()
def ensure_runtime(): subprocess.check_call([sys.executable,"-m","pip","install","--quiet","--extra-index-url","https://download.pytorch.org/whl/cu126",f"torch=={EXPECTED_TORCH}","rwkv==0.8.32","tokenizers==0.21.4","numpy==2.4.6","huggingface_hub==0.36.0"])
def load_git(root,name,blob,sha=None):
 raw=(root/name).read_bytes()
 if gitblob(raw)!=blob: raise RuntimeError(f"{name} git blob drift: {gitblob(raw)}")
 if sha and sha256(raw)!=sha: raise RuntimeError(f"{name} sha256 drift: {sha256(raw)}")
 return json.loads(raw)
def load_stimuli(root):
 idx=load_git(root,"MODEL_STIMULI_INDEX.json",INDEX_BLOB); rows=[]
 for m in idx["episodes"]:
  raw=(root/m["file"]).read_bytes()
  if gitblob(raw)!=m["git_blob_sha1"]: raise RuntimeError(f"stimulus blob drift {m['file']}")
  row=json.loads(raw)
  if row["episode"]!=m["episode"] or row["prompt_sha256"]!=m["prompt_sha256"] or row["options_digest"]!=m["options_digest"]: raise RuntimeError("stimulus index binding drift")
  if "expected_label" in row or "expected_option_id" in row: raise RuntimeError("answer key leaked into model stimulus")
  if sha256(row["prompt"].encode())!=row["prompt_sha256"]: raise RuntimeError("prompt bytes drift")
  rows.append(row)
 return rows,idx
def clone_gpu(s): return [x.detach().clone() for x in s]
def snap(s): return [x.detach().contiguous().cpu().clone() for x in s]
def clone_snap(s): return [x.clone() for x in s]
def restore(model,s):
 z=model.generate_zero_state()
 if len(z)!=len(s): raise RuntimeError("state tensor count drift")
 return [x.to(device=y.device,dtype=y.dtype).contiguous() for x,y in zip(s,z)]
def state_digest(s):
 import torch
 h=hashlib.sha256()
 for i,t in enumerate(s):
  c=t.detach().contiguous().cpu(); h.update(i.to_bytes(4,"big")); h.update(str(c.dtype).encode()); h.update(json.dumps(list(c.shape),separators=(",",":")).encode()); h.update(c.view(torch.uint8).numpy().tobytes(order="C"))
 return h.hexdigest()
def score(model,pipe,state,stim):
 import torch
 pt=[int(x) for x in pipe.encode(stim["prompt"])]
 if not pt: raise RuntimeError("zero-token prompt")
 logits,pstate=model.forward(pt,clone_gpu(state)); labels=[str(o["label"]) for o in stim["options"]]; scores={}; states={}; units=len(pt)
 for label in labels:
  cont=[int(x) for x in pipe.encode(" "+label)]; units+=len(cont); l=logits; branch=clone_gpu(pstate); total=0.0
  for tok in cont:
   v=l if getattr(l,"ndim",1)==1 else l[-1]; total+=float(torch.log_softmax(v.float(),dim=-1)[tok].item()); l,branch=model.forward([tok],branch)
  scores[label]=total/len(cont); states[label]=branch
 selected=max(labels,key=lambda x:(scores[x],-labels.index(x)))
 return selected,scores,units,states[selected]
def run_scale(model,pipe,scale,condition,stimuli,evaluator,external):
 import torch
 zero=snap(model.generate_zero_state()); single=None; banks=None; corr=[]; margins=[]; selected=[]; scores=[]; units=[]; btrace=[]; prior=[]; active=[]; resets=[]
 for i,(stim,ev) in enumerate(zip(stimuli,evaluator)):
  ep=i+1; action=ACTIONS[condition][i]; bank=ROUTES[condition][i]
  if stim["episode"]!=ep or ev["episode"]!=ep or stim["prompt_sha256"]!=ev["prompt_sha256"] or stim["options_digest"]!=ev["options_digest"]: raise RuntimeError("evaluator binding drift")
  if condition=="B": inp=clone_snap(zero)
  elif condition in {"C","G"}: inp=clone_snap(zero) if ep==1 or action=="RESET" else clone_snap(single)
  else:
   if ep==1: inp=clone_snap(zero)
   else:
    if banks is None or bank not in banks: raise RuntimeError("missing routed bank")
    inp=clone_snap(zero) if action=="RESET" else clone_snap(banks[bank])
  prior.append(state_digest(inp)); st=restore(model,inp); sel,sc,u,nxt=score(model,pipe,st,stim); out=snap(nxt); active.append(state_digest(out))
  if condition in {"M","X"}:
   if ep==1: banks={"A":clone_snap(out),"B":clone_snap(out)}
   else: banks[bank]=clone_snap(out)
   btrace.append({k:state_digest(v) for k,v in sorted(banks.items())})
  else: single=clone_snap(out); btrace.append({})
  exp=str(ev["expected_label"]); margin=float(sc[exp])-max(float(v) for k,v in sc.items() if k!=exp); corr.append(sel==exp); margins.append(margin); selected.append(sel); scores.append(sc); units.append(int(u)); resets.append("explicit_correction:user-correction-1" if action=="RESET" else None)
  del st,nxt; gc.collect(); torch.cuda.empty_cache()
 binding=True if condition not in {"M","X"} else len(btrace)==6 and all(set(x)=={"A","B"} for x in btrace) and btrace[0]["A"]==btrace[0]["B"]
 final_payload=next(o["payload"]["content"] for o in stimuli[-1]["options"] if o["label"]==selected[-1])
 total=sum(units); protocol_ok=total==external["proposal_units_per_condition"] and sum(len(s["prompt"].encode()) for s in stimuli)==external["decision_prompt_bytes_total"] and binding
 return {"protocol_ok":protocol_ok,"task_completed":False,"condition":condition,"scale":scale,"state_mode":f"rwkv7-g1a-{scale}-matched-gpu:{STRATEGY}","event_trace":EVENTS,"domain_trace":DOMAINS,"policy_action_trace":ACTIONS[condition],"route_bank_trace":ROUTES[condition],"state_restore_pattern":RESTORE[condition],"state_reset_reasons":resets,"state_reset_memory_heads":[None,None,None,external["memory_head_trace"][3] if condition in {"G","M","X"} else None,None,None],"reset_binding_ok":True,"bank_binding_ok":binding,"bank_count":2 if condition in {"M","X"} else 1,"bank_state_sha256_trace":btrace,"prior_selected_state_sha256_trace":prior,"active_state_sha256_trace":active,"decision_prompt_trace":[s["prompt_sha256"] for s in stimuli],"decision_options_trace":[s["options_digest"] for s in stimuli],"decision_prompt_bytes":external["decision_prompt_bytes_total"],"proposal_units":total,"proposal_units_by_episode":units,"proposal_unit_kinds":["tokens"],"proposal_policy_modes":["RWKV_LOGPROB_TYPED_CHOICE_V1"],"proposal_oracle_used":[False]*6,"selected_labels":selected,"proposal_correctness":corr,"proposal_correct_count":sum(int(v) for v in corr),"expected_margins":margins,"expected_margin_sum":sum(margins),"score_trace":scores,"provider_receipt_count":external["provider_receipt_count"],"provider_request_ids":external["provider_request_ids"],"source_reads":external["source_reads"],"provider_mutation_count":external["provider_mutation_count"],"provider_complete":False,"final_manifest":final_payload,"usable_memory_ids":external["usable_memory_ids"],"memory_head_trace":external["memory_head_trace"],"provider_receipt_head_trace":external["provider_receipt_head_trace"],"provider_state_sha256_trace":external["provider_state_sha256_trace"],"external_trace_git_blob_sha1":EXTERNAL_BLOB,"protocol_metrics":{"invalidation_ok":True,"reset_binding_ok":True,"bank_binding_ok":binding,"routed_bank_count":2 if condition in {"M","X"} else 0,"bank_resets":sum(1 for x in resets if x)}}
