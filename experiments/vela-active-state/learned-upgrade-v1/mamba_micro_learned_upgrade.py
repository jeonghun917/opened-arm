from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_ID = "state-spaces/mamba-130m-hf"
LR = 1e-4
EPOCHS = 1

TRAIN = [
    ("A record first said RED. The correction replaces it with BLUE.\nCorrected color:", " BLUE"),
    ("A record first said BLUE. The correction replaces it with RED.\nCorrected color:", " RED"),
    ("Old value: RED. New authoritative value: BLUE.\nCurrent color:", " BLUE"),
    ("Old value: BLUE. New authoritative value: RED.\nCurrent color:", " RED"),
    ("Initial answer RED is obsolete; use BLUE instead.\nFinal color:", " BLUE"),
    ("Initial answer BLUE is obsolete; use RED instead.\nFinal color:", " RED"),
]

HELDOUT = [
    {"pair":"correction","variant":"A","prompt":"The color was RED. Correction: the final color is BLUE, not RED.\nFinal color:","candidates":[" BLUE"," RED"],"expected":" BLUE"},
    {"pair":"correction","variant":"B","prompt":"The color was BLUE. Correction: the final color is RED, not BLUE.\nFinal color:","candidates":[" BLUE"," RED"],"expected":" RED"},
    {"pair":"codeword-control","variant":"A","prompt":"The active codeword is ALPHA. BETA is not active.\nActive codeword:","candidates":[" ALPHA"," BETA"],"expected":" ALPHA"},
    {"pair":"codeword-control","variant":"B","prompt":"The active codeword is BETA. ALPHA is not active.\nActive codeword:","candidates":[" ALPHA"," BETA"],"expected":" BETA"},
    {"pair":"plan-control","variant":"A","prompt":"Plan: collect evidence, then commit. Evidence collection is complete.\nNext step:","candidates":[" commit"," collect"],"expected":" commit"},
    {"pair":"plan-control","variant":"B","prompt":"Plan: collect evidence, then commit. Evidence is still missing.\nNext step:","candidates":[" commit"," collect"],"expected":" collect"},
]

MIGRATION_PREFIX = (
    "The system is tracking a correction from RED to BLUE while preserving an unrelated plan to verify evidence before commit."
)
MIGRATION_CONT = " Therefore"


def write_report(report: dict) -> None:
    path = os.environ.get("VELA_RESULT_PATH")
    if path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def candidate_logprob(model, tok, prompt: str, candidate: str) -> float:
    pids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    cids = tok(candidate, return_tensors="pt", add_special_tokens=False).input_ids
    if cids.shape[1] < 1: raise RuntimeError("empty candidate")
    feed = torch.cat([pids, cids[:,:-1]], dim=1) if cids.shape[1] > 1 else pids
    with torch.no_grad():
        logits = model(feed, use_cache=False, return_dict=True).logits.detach().float()
        lp = torch.log_softmax(logits, dim=-1)
    total=0.0; start=pids.shape[1]-1
    for j in range(cids.shape[1]):
        total += float(lp[0,start+j,int(cids[0,j])])
    return total


def evaluate(model,tok):
    rows=[]; correct=0
    for fx in HELDOUT:
        scores={c:candidate_logprob(model,tok,fx["prompt"],c) for c in fx["candidates"]}
        chosen=max(scores,key=scores.get); other=next(c for c in fx["candidates"] if c!=fx["expected"])
        ok=chosen==fx["expected"]; correct+=int(ok)
        rows.append({"pair":fx["pair"],"variant":fx["variant"],"expected":fx["expected"],"chosen":chosen,"correct":ok,"margin":scores[fx["expected"]]-scores[other],"scores":scores})
    pair_summary={}
    for pair in sorted({r["pair"] for r in rows}):
        rr=sorted([r for r in rows if r["pair"]==pair],key=lambda x:x["variant"])
        pair_summary[pair]={"both_correct":all(r["correct"] for r in rr),"choices":[r["chosen"] for r in rr],"choice_flips":rr[0]["chosen"]!=rr[1]["chosen"]}
    return {"accuracy":correct/len(rows),"pair_summary":pair_summary,"rows":rows}


def train_loss(model,tok,prompt,target):
    pids=tok(prompt,return_tensors="pt",add_special_tokens=False).input_ids
    tids=tok(target,return_tensors="pt",add_special_tokens=False).input_ids
    full=torch.cat([pids,tids],dim=1)
    logits=model(full,use_cache=False,return_dict=True).logits
    start=pids.shape[1]-1
    pred=logits[:,start:start+tids.shape[1],:].reshape(-1,logits.shape[-1])
    gold=tids.reshape(-1)
    return F.cross_entropy(pred,gold)


def tensor_refs(obj):
    refs=[]; seen=set()
    def walk(x):
        if torch.is_tensor(x):
            if id(x) not in seen: seen.add(id(x)); refs.append(x)
        elif isinstance(x,dict):
            for v in x.values(): walk(v)
        elif isinstance(x,(list,tuple)):
            for v in x: walk(v)
        elif hasattr(x,"__dict__"):
            for v in vars(x).values(): walk(v)
    walk(obj); return refs


def cache_distance(a,b):
    aa,bb=tensor_refs(a),tensor_refs(b)
    if len(aa)!=len(bb): raise RuntimeError("cache tensor count mismatch")
    sq=0.0;mx=0.0;n=0
    for x,y in zip(aa,bb):
        d=x.detach().float().cpu()-y.detach().float().cpu();sq+=float((d*d).sum());mx=max(mx,float(d.abs().max()));n+=d.numel()
    return {"rms":math.sqrt(sq/max(n,1)),"l2":math.sqrt(sq),"max_abs":mx,"numel":int(n)}


def continue_one(model,token,cache,pos):
    cp=torch.tensor([pos],dtype=torch.long)
    out=model(token,cache_params=cache,cache_position=cp,use_cache=True,return_dict=True)
    return out.logits[:,-1].detach().float().cpu()


def run():
    transformers_version=None
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer,__version__ as transformers_version
        torch.manual_seed(917)
        tok=AutoTokenizer.from_pretrained(MODEL_ID)
        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32).cpu()
        target=model.backbone.layers[0].mixer.x_proj.weight
        before_weight=target.detach().clone()
        for p in model.parameters(): p.requires_grad_(False)
        target.requires_grad_(True)

        # Capture W1 active state before any learning.
        mig_ids=tok(MIGRATION_PREFIX,return_tensors="pt",add_special_tokens=False).input_ids
        cont=tok(MIGRATION_CONT,return_tensors="pt",add_special_tokens=False).input_ids[:,:1]
        model.eval()
        with torch.no_grad():
            old_cache=model(mig_ids,use_cache=True,return_dict=True).cache_params
        with tempfile.TemporaryDirectory() as td:
            old_path=Path(td)/"w1_state.pt";torch.save(old_cache,old_path)

            baseline=evaluate(model,tok)

            opt=torch.optim.AdamW([target],lr=LR,weight_decay=0.0)
            losses=[]
            model.train()
            for _ in range(EPOCHS):
                for prompt,gold in TRAIN:
                    opt.zero_grad(set_to_none=True)
                    loss=train_loss(model,tok,prompt,gold)
                    loss.backward()
                    grad_norm=float(target.grad.detach().float().norm()) if target.grad is not None else 0.0
                    opt.step()
                    losses.append({"loss":float(loss.detach()),"grad_norm":grad_norm})

            model.eval()
            after=evaluate(model,tok)
            after_weight=target.detach().clone()
            wd=(after_weight-before_weight).float()
            weight_delta={"l2":float(wd.norm()),"rms":float(torch.sqrt(torch.mean(wd*wd))),"max_abs":float(wd.abs().max()),"relative_l2":float(wd.norm()/before_weight.float().norm())}

            # W2-native state vs direct W1-state import under the actually learned W2.
            with torch.no_grad():
                new_cache=model(mig_ids,use_cache=True,return_dict=True).cache_params
                imported=torch.load(old_path,map_location="cpu",weights_only=False)
                state_err=cache_distance(imported,new_cache)
                native_logits=continue_one(model,cont,copy.deepcopy(new_cache),int(mig_ids.shape[1]))
                imported_logits=continue_one(model,cont,imported,int(mig_ids.shape[1]))
                ld=imported_logits-native_logits
                migration={"old_vs_new_native_state":state_err,"continuation_logit_error":{"rms":float(torch.sqrt(torch.mean(ld*ld))),"max_abs":float(ld.abs().max()),"argmax_same":bool(torch.argmax(imported_logits,dim=-1).item()==torch.argmax(native_logits,dim=-1).item())}}

        report={
            "status":"MICRO_LEARNED_ENGINE_UPGRADE_PROBE",
            "model":MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,
            "trainable_parameter":"backbone.layers[0].mixer.x_proj.weight","lr":LR,"epochs":EPOCHS,"train_examples":len(TRAIN),
            "baseline":baseline,"after":after,"training_trace":losses,"weight_delta":weight_delta,"migration":migration,
            "claim_boundary":"This is an actual gradient-based same-architecture weight update on a tiny synthetic correction task. It may demonstrate narrow task learning and state incompatibility after learning, but it is not evidence of broad reasoning improvement."
        }
        write_report(report)
    except BaseException as exc:
        write_report({"status":"MICRO_LEARNED_ENGINE_UPGRADE_ERROR","model":MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-30:],"claim_boundary":"Runtime/training failure only; no engine-upgrade verdict."})
        raise

if __name__=="__main__": run()
