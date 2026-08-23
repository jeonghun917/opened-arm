from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1] / "learned-upgrade-v3" / "mamba_upgrade_migration.py"
spec = importlib.util.spec_from_file_location("vela_v3", BASE)
if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {BASE}")
v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)

ORIG_DEEPCOPY = copy.deepcopy

def clone_cache(obj):
    if obj is None: return None
    buf = io.BytesIO(); torch.save(obj, buf); buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)

def patched_deepcopy(obj, memo=None):
    if obj is not None and (hasattr(obj,"conv_states") or hasattr(obj,"ssm_states")):
        return clone_cache(obj)
    return ORIG_DEEPCOPY(obj, memo)
copy.deepcopy = patched_deepcopy
v3.copy.deepcopy = patched_deepcopy

PROBES = [
    {"id":"codeword","suffix":"\nCurrent codeword:","candidates":[" BETA"," ALPHA"],"expected":" BETA"},
    {"id":"verification","suffix":"\nVerification status:","candidates":[" incomplete"," complete"],"expected":" incomplete"},
    {"id":"project","suffix":"\nProject Orion status:","candidates":[" active"," paused"],"expected":" active"},
    {"id":"hypothesis","suffix":"\nHypothesis one is:","candidates":[" weakened"," strengthened"],"expected":" weakened"},
]
HORIZONS=[0,2,4,8,16,32]


def write_report(report):
    pth=os.environ.get("VELA_RESULT_PATH")
    if pth:
        p=Path(pth); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


def score_candidate_from_state(model,tok,state,start_pos,suffix,candidate):
    cache=clone_cache(state); pos=start_pos; logits=None
    sids=tok(suffix,return_tensors="pt",add_special_tokens=False).input_ids
    cids=tok(candidate,return_tensors="pt",add_special_tokens=False).input_ids
    with torch.no_grad():
        for j in range(sids.shape[1]):
            token=sids[:,j:j+1]
            out=model(token,cache_params=cache,cache_position=torch.tensor([pos],dtype=torch.long),use_cache=True,return_dict=True)
            cache=out.cache_params; logits=out.logits[:,-1].detach().float(); pos+=1
        if logits is None: raise RuntimeError("probe suffix tokenized empty")
        total=0.0
        for j in range(cids.shape[1]):
            lp=torch.log_softmax(logits,dim=-1); tid=int(cids[0,j]); total+=float(lp[0,tid])
            if j+1<cids.shape[1]:
                token=cids[:,j:j+1]
                out=model(token,cache_params=cache,cache_position=torch.tensor([pos],dtype=torch.long),use_cache=True,return_dict=True)
                cache=out.cache_params; logits=out.logits[:,-1].detach().float(); pos+=1
    return total


def probe_state(model,tok,state,T):
    rows=[]
    for fx in PROBES:
        scores={c:score_candidate_from_state(model,tok,state,T,fx["suffix"],c) for c in fx["candidates"]}
        chosen=max(scores,key=scores.get); other=next(c for c in fx["candidates"] if c!=fx["expected"])
        rows.append({"id":fx["id"],"expected":fx["expected"],"chosen":chosen,"correct":chosen==fx["expected"],"margin":scores[fx["expected"]]-scores[other],"scores":scores})
    return rows


def compare_to_native(rows,native):
    agreements=0; sq=0.0; n=0; margin_abs=[]
    native_by={r["id"]:r for r in native}
    for r in rows:
        nr=native_by[r["id"]]; agreements+=int(r["chosen"]==nr["chosen"]); margin_abs.append(abs(r["margin"]-nr["margin"]))
        for c,v in r["scores"].items(): sq+=(v-nr["scores"][c])**2; n+=1
    return {"decision_agreement":agreements/len(rows),"score_rms_error":math.sqrt(sq/max(n,1)),"mean_margin_abs_error":sum(margin_abs)/len(margin_abs),"expected_accuracy":sum(int(r["correct"]) for r in rows)/len(rows)}


def run():
    transformers_version=None
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer,__version__ as transformers_version
        torch.manual_seed(v3.SEED); random.seed(v3.SEED)
        tok=AutoTokenizer.from_pretrained(v3.MODEL_ID); model=AutoModelForCausalLM.from_pretrained(v3.MODEL_ID,torch_dtype=torch.float32).cpu().eval()
        hist=tok(v3.MIGRATION_HISTORY,return_tensors="pt",add_special_tokens=False).input_ids; T=int(hist.shape[1]); horizons=sorted(set([h for h in HORIZONS if h<T]+[T]))
        old_caches={}
        with torch.no_grad():
            for h in horizons:
                cut=T-h; old_caches[h]=None if cut==0 else clone_cache(model(hist[:,:cut],use_cache=True,return_dict=True).cache_params)
            old_full=clone_cache(model(hist,use_cache=True,return_dict=True).cache_params)
        baseline_cap=v3.evaluate(model,tok); old_probe=probe_state(model,tok,old_full,T)

        trainable=[]
        for p in model.parameters(): p.requires_grad_(False)
        for name,p in model.named_parameters():
            if ".mixer.x_proj.weight" in name: p.requires_grad_(True); trainable.append((name,p))
        if not trainable: raise RuntimeError("no x_proj weights found")
        opt=torch.optim.AdamW([p for _,p in trainable],lr=v3.LR,weight_decay=0.0); order=list(range(len(v3.TRAIN))); epoch_loss=[]
        for epoch in range(v3.EPOCHS):
            random.Random(v3.SEED+epoch).shuffle(order); model.train(); total=0.0
            for idx in order:
                prompt,gold=v3.TRAIN[idx]; opt.zero_grad(set_to_none=True); loss=v3.train_loss(model,tok,prompt,gold); loss.backward(); torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0); opt.step(); total+=float(loss.detach())
            epoch_loss.append(total/len(v3.TRAIN))
        model.eval(); after_cap=v3.evaluate(model,tok)

        with torch.no_grad(): w2_native=clone_cache(model(hist,use_cache=True,return_dict=True).cache_params)
        native_probe=probe_state(model,tok,w2_native,T); migration=[]
        for h in horizons:
            cut=T-h
            if h==T: migrated=clone_cache(w2_native)
            elif h==0: migrated=clone_cache(old_full)
            else:
                migrated,_=v3.run_tokens_with_cache(model,hist[:,cut:],clone_cache(old_caches[h]),cut); migrated=clone_cache(migrated)
            rows=probe_state(model,tok,migrated,T)
            migration.append({"w2_replayed_recent_tokens":h,"w1_history_tokens_kept_without_recompute":T-h,"state_error_vs_w2_native":v3.cache_distance(migrated,w2_native),"probe_comparison_vs_w2_native":compare_to_native(rows,native_probe),"probe_rows":rows})

        report={"status":"LEARNED_UPGRADE_DECISION_FIDELITY","model":v3.MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,"training":{"trainable_pattern":"*.mixer.x_proj.weight","lr":v3.LR,"epochs":v3.EPOCHS,"train_examples":len(v3.TRAIN),"epoch_mean_loss":epoch_loss},"capability":{"baseline_accuracy":baseline_cap["accuracy"],"after_accuracy":after_cap["accuracy"],"baseline_correction_accuracy":baseline_cap["correction_accuracy"],"after_correction_accuracy":after_cap["correction_accuracy"],"baseline_control_accuracy":baseline_cap["control_accuracy"],"after_control_accuracy":after_cap["control_accuracy"]},"migration":{"history_tokens":T,"target":"W2-native after same causal history","w1_probe_before_upgrade":old_probe,"w2_native_probe":native_probe,"rows":migration},"success_definition":"W2 must improve held-out capability, while migrated W2 is compared to W2-native decisions/margins rather than to W1 outputs.","claim_boundary":"Narrow synthetic learned upgrade and task-relevant migration probe. This is stronger than raw-logit matching but still not broad reasoning, identity, or consciousness evidence."}
        write_report(report)
    except BaseException as exc:
        write_report({"status":"LEARNED_UPGRADE_V4_ERROR","model":getattr(v3,"MODEL_ID",None),"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-35:],"claim_boundary":"Runtime/training failure only; no migration verdict."}); raise

if __name__=="__main__": run()
