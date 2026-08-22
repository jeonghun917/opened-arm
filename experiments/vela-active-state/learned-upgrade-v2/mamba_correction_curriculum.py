from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_ID = "state-spaces/mamba-130m-hf"
LR = 2e-4
EPOCHS = 3
SEED = 917

PAIRS = [(" CAT"," DOG"),(" EAST"," WEST"),(" one"," two"),(" Mars"," Venus"),(" HOT"," COLD"),(" UP"," DOWN")]
TEMPLATES = [
    "Initial value:{old}. Correction: replace it with{new}.\nCurrent value:",
    "The old answer was{old}, but that answer is obsolete. The corrected answer is{new}.\nFinal answer:",
    "Before correction:{old}. After correction:{new}.\nUse the corrected value:",
]
TRAIN=[]
for i,(a,b) in enumerate(PAIRS):
    t=TEMPLATES[i%len(TEMPLATES)]
    TRAIN.append((t.format(old=a,new=b),b))
    TRAIN.append((t.format(old=b,new=a),a))

HELDOUT = [
    {"pair":"color-correction","variant":"A","prompt":"The color was RED. Correction: the final color is BLUE, not RED.\nFinal color:","candidates":[" BLUE"," RED"],"expected":" BLUE"},
    {"pair":"color-correction","variant":"B","prompt":"The color was BLUE. Correction: the final color is RED, not BLUE.\nFinal color:","candidates":[" BLUE"," RED"],"expected":" RED"},
    {"pair":"codeword-correction","variant":"A","prompt":"The codeword was ALPHA. Correction: the current codeword is BETA, not ALPHA.\nCurrent codeword:","candidates":[" BETA"," ALPHA"],"expected":" BETA"},
    {"pair":"codeword-correction","variant":"B","prompt":"The codeword was BETA. Correction: the current codeword is ALPHA, not BETA.\nCurrent codeword:","candidates":[" BETA"," ALPHA"],"expected":" ALPHA"},
    {"pair":"plan-control","variant":"A","prompt":"Plan: collect evidence, then commit. Evidence collection is complete.\nNext step:","candidates":[" commit"," collect"],"expected":" commit"},
    {"pair":"plan-control","variant":"B","prompt":"Plan: collect evidence, then commit. Evidence is still missing.\nNext step:","candidates":[" commit"," collect"],"expected":" collect"},
]

MIGRATION_PREFIX = "The old answer was RED, the corrected answer is BLUE, and an unrelated plan still requires verification before commit."
MIGRATION_CONT = " Therefore"


def write_report(report):
    pth=os.environ.get("VELA_RESULT_PATH")
    if pth:
        p=Path(pth);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


def candidate_logprob(model,tok,prompt,candidate):
    pids=tok(prompt,return_tensors="pt",add_special_tokens=False).input_ids
    cids=tok(candidate,return_tensors="pt",add_special_tokens=False).input_ids
    feed=torch.cat([pids,cids[:,:-1]],dim=1) if cids.shape[1]>1 else pids
    with torch.no_grad():
        logits=model(feed,use_cache=False,return_dict=True).logits.detach().float();lp=torch.log_softmax(logits,dim=-1)
    total=0.0;start=pids.shape[1]-1
    for j in range(cids.shape[1]): total+=float(lp[0,start+j,int(cids[0,j])])
    return total


def evaluate(model,tok):
    rows=[];correct=0
    for fx in HELDOUT:
        scores={c:candidate_logprob(model,tok,fx["prompt"],c) for c in fx["candidates"]};chosen=max(scores,key=scores.get)
        other=next(c for c in fx["candidates"] if c!=fx["expected"]);ok=chosen==fx["expected"];correct+=int(ok)
        rows.append({"pair":fx["pair"],"variant":fx["variant"],"expected":fx["expected"],"chosen":chosen,"correct":ok,"margin":scores[fx["expected"]]-scores[other],"scores":scores})
    summary={}
    for pair in sorted({r["pair"] for r in rows}):
        rr=sorted([r for r in rows if r["pair"]==pair],key=lambda x:x["variant"])
        summary[pair]={"both_correct":all(r["correct"] for r in rr),"choices":[r["chosen"] for r in rr],"choice_flips":rr[0]["chosen"]!=rr[1]["chosen"]}
    return {"accuracy":correct/len(rows),"pair_summary":summary,"rows":rows}


def train_loss(model,tok,prompt,target):
    pids=tok(prompt,return_tensors="pt",add_special_tokens=False).input_ids;tids=tok(target,return_tensors="pt",add_special_tokens=False).input_ids
    full=torch.cat([pids,tids],dim=1);logits=model(full,use_cache=False,return_dict=True).logits
    start=pids.shape[1]-1;pred=logits[:,start:start+tids.shape[1],:].reshape(-1,logits.shape[-1]);gold=tids.reshape(-1)
    return F.cross_entropy(pred,gold)


def tensor_refs(obj):
    refs=[];seen=set()
    def walk(x):
        if torch.is_tensor(x):
            if id(x) not in seen:seen.add(id(x));refs.append(x)
        elif isinstance(x,dict):
            for v in x.values():walk(v)
        elif isinstance(x,(list,tuple)):
            for v in x:walk(v)
        elif hasattr(x,"__dict__"):
            for v in vars(x).values():walk(v)
    walk(obj);return refs


def cache_distance(a,b):
    aa,bb=tensor_refs(a),tensor_refs(b);sq=0.0;mx=0.0;n=0
    if len(aa)!=len(bb):raise RuntimeError("cache tensor count mismatch")
    for x,y in zip(aa,bb):
        d=x.detach().float().cpu()-y.detach().float().cpu();sq+=float((d*d).sum());mx=max(mx,float(d.abs().max()));n+=d.numel()
    return {"rms":math.sqrt(sq/max(n,1)),"l2":math.sqrt(sq),"max_abs":mx,"numel":int(n)}


def continue_one(model,token,cache,pos):
    out=model(token,cache_params=cache,cache_position=torch.tensor([pos],dtype=torch.long),use_cache=True,return_dict=True)
    return out.logits[:,-1].detach().float().cpu()


def run():
    transformers_version=None
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer,__version__ as transformers_version
        torch.manual_seed(SEED);random.seed(SEED)
        tok=AutoTokenizer.from_pretrained(MODEL_ID);model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32).cpu()
        target=model.backbone.layers[0].mixer.x_proj.weight;before=target.detach().clone()
        for p in model.parameters():p.requires_grad_(False)
        target.requires_grad_(True)
        mig_ids=tok(MIGRATION_PREFIX,return_tensors="pt",add_special_tokens=False).input_ids;cont=tok(MIGRATION_CONT,return_tensors="pt",add_special_tokens=False).input_ids[:,:1]
        model.eval()
        with torch.no_grad():old_cache=model(mig_ids,use_cache=True,return_dict=True).cache_params
        with tempfile.TemporaryDirectory() as td:
            old_path=Path(td)/"w1.pt";torch.save(old_cache,old_path)
            baseline=evaluate(model,tok)
            opt=torch.optim.AdamW([target],lr=LR,weight_decay=0.0);trace=[];epoch_eval=[]
            order=list(range(len(TRAIN)))
            for epoch in range(EPOCHS):
                random.Random(SEED+epoch).shuffle(order);model.train();loss_sum=0.0
                for idx in order:
                    prompt,gold=TRAIN[idx];opt.zero_grad(set_to_none=True);loss=train_loss(model,tok,prompt,gold);loss.backward();opt.step();loss_sum+=float(loss.detach())
                model.eval();ev=evaluate(model,tok);epoch_eval.append({"epoch":epoch+1,"mean_train_loss":loss_sum/len(TRAIN),"heldout":ev})
            after=epoch_eval[-1]["heldout"];after_w=target.detach().clone();d=(after_w-before).float()
            weight_delta={"l2":float(d.norm()),"rms":float(torch.sqrt(torch.mean(d*d))),"max_abs":float(d.abs().max()),"relative_l2":float(d.norm()/before.float().norm())}
            with torch.no_grad():
                new_cache=model(mig_ids,use_cache=True,return_dict=True).cache_params;imported=torch.load(old_path,map_location="cpu",weights_only=False)
                state_err=cache_distance(imported,new_cache);native_logits=continue_one(model,cont,copy.deepcopy(new_cache),int(mig_ids.shape[1]));imported_logits=continue_one(model,cont,imported,int(mig_ids.shape[1]));ld=imported_logits-native_logits
            migration={"old_vs_new_native_state":state_err,"continuation_logit_error":{"rms":float(torch.sqrt(torch.mean(ld*ld))),"max_abs":float(ld.abs().max()),"argmax_same":bool(torch.argmax(imported_logits,dim=-1).item()==torch.argmax(native_logits,dim=-1).item())}}
        report={"status":"CORRECTION_CURRICULUM_LEARNED_UPGRADE_PROBE","model":MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,"trainable_parameter":"backbone.layers[0].mixer.x_proj.weight","lr":LR,"epochs":EPOCHS,"train_examples":len(TRAIN),"train_value_pairs":[list(x) for x in PAIRS],"baseline":baseline,"epoch_eval":epoch_eval,"after":after,"weight_delta":weight_delta,"migration":migration,"claim_boundary":"Actual gradient-based update trained on correction patterns with held-out value pairs. Improvement, if any, is narrow transfer evidence only; controls and state migration are reported separately, and this is not broad reasoning evidence."}
        write_report(report)
    except BaseException as exc:
        write_report({"status":"CORRECTION_CURRICULUM_UPGRADE_ERROR","model":MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-30:],"claim_boundary":"Runtime/training failure only; no capability or migration verdict."});raise

if __name__=="__main__":run()
