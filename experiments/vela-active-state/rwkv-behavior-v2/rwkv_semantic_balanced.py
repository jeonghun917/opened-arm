from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import torch

MODEL_ID="RWKV/rwkv-4-169m-pile"
FIXTURES=[
{"pair":"codeword","variant":"A","prefix":"Codeword: ALPHA. BETA is inactive.\n","suffix":"Codeword:","candidates":[" ALPHA"," BETA"],"expected":" ALPHA"},
{"pair":"codeword","variant":"B","prefix":"Codeword: BETA. ALPHA is inactive.\n","suffix":"Codeword:","candidates":[" ALPHA"," BETA"],"expected":" BETA"},
{"pair":"correction","variant":"A","prefix":"Initial codeword: RED.\nCorrection: codeword is BLUE.\n","suffix":"Current codeword:","candidates":[" BLUE"," RED"],"expected":" BLUE"},
{"pair":"correction","variant":"B","prefix":"Initial codeword: BLUE.\nCorrection: codeword is RED.\n","suffix":"Current codeword:","candidates":[" BLUE"," RED"],"expected":" RED"},
{"pair":"scope","variant":"A","prefix":"Mars is in scope. Venus is out of scope.\n","suffix":"Record in scope:","candidates":[" MARS"," VENUS"],"expected":" MARS"},
{"pair":"scope","variant":"B","prefix":"Venus is in scope. Mars is out of scope.\n","suffix":"Record in scope:","candidates":[" MARS"," VENUS"],"expected":" VENUS"},
{"pair":"hypothesis","variant":"A","prefix":"Option one status: ACTIVE.\nOption two status: REJECTED.\n","suffix":"Active option:","candidates":[" one"," two"],"expected":" one"},
{"pair":"hypothesis","variant":"B","prefix":"Option one status: REJECTED.\nOption two status: ACTIVE.\n","suffix":"Active option:","candidates":[" one"," two"],"expected":" two"},
{"pair":"plan","variant":"A","prefix":"Step one: collect.\nStep two: verify.\nStep three: commit.\nCompleted: collect, verify.\n","suffix":"Next step:","candidates":[" commit"," collect"],"expected":" commit"},
{"pair":"plan","variant":"B","prefix":"Step one: collect.\nStep two: verify.\nStep three: commit.\nCompleted: none.\n","suffix":"Next step:","candidates":[" commit"," collect"],"expected":" collect"},
]

def clone_state(state,device): return None if state is None else [x.detach().to(device).clone() for x in state]
def state_cpu(state): return [x.detach().cpu().clone() for x in state]

def score_candidate(model,tok,device,state,suffix,candidate):
    s=tok(suffix,return_tensors="pt",add_special_tokens=False).input_ids.to(device)
    c=tok(candidate,return_tensors="pt",add_special_tokens=False).input_ids.to(device)
    feed=torch.cat([s,c[:,:-1]],dim=1) if c.shape[1]>1 else s
    with torch.no_grad():
        out=model(feed,state=clone_state(state,device),use_cache=True,return_dict=True)
        lp=torch.log_softmax(out.logits.detach().float(),dim=-1)
    total=0.0; n=s.shape[1]
    for j in range(c.shape[1]): total+=float(lp[0,n-1+j,int(c[0,j])])
    return total

def score_cond(model,tok,device,state,fx):
    scores={c:score_candidate(model,tok,device,state,fx["suffix"],c) for c in fx["candidates"]}
    chosen=max(scores,key=scores.get); other=next(c for c in fx["candidates"] if c!=fx["expected"])
    return {"scores":scores,"chosen":chosen,"correct":chosen==fx["expected"],"expected_margin":scores[fx["expected"]]-scores[other]}

def run():
    from transformers import AutoModelForCausalLM,AutoTokenizer
    device="cuda" if torch.cuda.is_available() else "cpu"; dtype=torch.float16 if device=="cuda" else torch.float32
    tok=AutoTokenizer.from_pretrained(MODEL_ID); model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=dtype).to(device).eval()
    rows=[]; max_restore=0.0; max_replay=0.0
    for fx in FIXTURES:
        ids=tok(fx["prefix"],return_tensors="pt",add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            a=model(ids,use_cache=True,return_dict=True); base=state_cpu(a.state)
            b=model(ids,use_cache=True,return_dict=True); replay=state_cpu(b.state)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"state.pt"; torch.save(base,p); restored=torch.load(p,map_location="cpu")
            conds={"native":score_cond(model,tok,device,base,fx),"restored":score_cond(model,tok,device,restored,fx),"replay":score_cond(model,tok,device,replay,fx),"fresh":score_cond(model,tok,device,None,fx)}
        for c in fx["candidates"]:
            max_restore=max(max_restore,abs(conds["native"]["scores"][c]-conds["restored"]["scores"][c]))
            max_replay=max(max_replay,abs(conds["native"]["scores"][c]-conds["replay"]["scores"][c]))
        rows.append({"pair":fx["pair"],"variant":fx["variant"],"expected":fx["expected"],"conditions":conds})
    pair_summary={}
    for pair in sorted({r["pair"] for r in rows}):
        rr=sorted([r for r in rows if r["pair"]==pair],key=lambda r:r["variant"]); pair_summary[pair]={}
        for cond in ("native","restored","replay","fresh"):
            choices=[r["conditions"][cond]["chosen"] for r in rr]
            pair_summary[pair][cond]={"choices":choices,"choice_flips_with_state":choices[0]!=choices[1],"both_variants_correct":all(r["conditions"][cond]["correct"] for r in rr)}
    n=len(rows); acc={c:sum(int(r["conditions"][c]["correct"]) for r in rows)/n for c in ("native","restored","replay","fresh")}
    report={"status":"M1_SEMANTIC_BALANCED_PROBE_ONLY","model":MODEL_ID,"device":device,"dtype":str(dtype),"fixture_count":n,"accuracy":acc,"max_native_vs_restored_score_diff":max_restore,"max_native_vs_replay_score_diff":max_replay,"native_pairs_correct_and_flipping":sum(1 for v in pair_summary.values() if v["native"]["both_variants_correct"] and v["native"]["choice_flips_with_state"]),"fresh_pairs_flipping":sum(1 for v in pair_summary.values() if v["fresh"]["choice_flips_with_state"]),"pair_summary":pair_summary,"rows":rows,"claim_boundary":"Balanced semantic paired probe on RWKV-4 169M. Behavioral/state evidence only, not VELA identity or engine-selection evidence."}
    pth=os.environ.get("VELA_RESULT_PATH")
    if pth:
        p=Path(pth);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if max_restore>1e-5 or max_replay>1e-5: raise SystemExit(1)
if __name__=="__main__": run()
