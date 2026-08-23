from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import tempfile
import traceback
import urllib.request
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
V1_PATH = BASE / "rwkv7-learned-upgrade-v1" / "rwkv7_learned_upgrade.py"
spec = importlib.util.spec_from_file_location("rwkv7_upgrade_v1", V1_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {V1_PATH}")
rw = importlib.util.module_from_spec(spec); spec.loader.exec_module(rw)

EVENT_INTERVALS = [1, 2, 4, 8]
FIXTURES = [
    {"id":"early","project":"Orion","old":"ALPHA","new":"BETA","segments":[
        "Project Orion remains active. ","The old codeword was ALPHA. ","Correction: the current codeword is BETA, not ALPHA. ",
        "Unrelated telemetry packet 17 was archived. ","Verification is incomplete. ","A historical memo mentions ALPHA but is obsolete. ",
        "Hypothesis one is weakened. ","External action remains blocked."],"critical_segment":2},
    {"id":"middle","project":"Helios","old":"BLUE","new":"RED","segments":[
        "Project Helios remains active. ","The old codeword was BLUE. ","Unrelated telemetry packet 21 was archived. ","Verification is incomplete. ",
        "Correction: the current codeword is RED, not BLUE. ","A historical memo mentions BLUE but is obsolete. ","Hypothesis one is weakened. ",
        "External action remains blocked."],"critical_segment":4},
    {"id":"late","project":"Icarus","old":"LOW","new":"HIGH","segments":[
        "Project Icarus remains active. ","The old codeword was LOW. ","Unrelated telemetry packet 31 was archived. ","Verification is incomplete. ",
        "Hypothesis one is weakened. ","A historical memo mentions LOW but is obsolete. ","Correction: the current codeword is HIGH, not LOW. ",
        "External action remains blocked."],"critical_segment":6},
    {"id":"distractor_heavy","project":"Juno","old":"EAST","new":"WEST","segments":[
        "Project Juno remains active. ","The old codeword was EAST. ","A note says weather moved eastward; this is unrelated to the codeword. ",
        "Verification is incomplete. ","A historical memo mentions EAST but is obsolete. ","Two unrelated sensors were recalibrated. ",
        "Correction: the current codeword is WEST, not EAST. ","Hypothesis one is weakened. ","External action remains blocked."],"critical_segment":6},
]


def write_report(report):
    pth=os.environ.get("VELA_RESULT_PATH")
    if pth:
        p=Path(pth); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


def save_keys(model,args):
    return {f"blocks.{i}.att.key.weight":model.z[f"blocks.{i}.att.key.weight"].detach().clone() for i in range(args.n_layer)}


def load_keys(model,snap):
    with torch.no_grad():
        for k,v in snap.items(): model.z[k].copy_(v)


def boundaries(tok,segments):
    starts=[]; ends=[]
    for i in range(len(segments)):
        starts.append(len(tok.encode("".join(segments[:i]))))
        ends.append(len(tok.encode("".join(segments[:i+1]))))
    return starts,ends


def probe_specs(fx):
    return [
        ("codeword","\nCurrent codeword:",[" "+fx["new"]," "+fx["old"]]," "+fx["new"]),
        ("verification","\nVerification status:",[" incomplete"," complete"]," incomplete"),
        ("project",f"\nProject {fx['project']} status:",[" active"," paused"]," active"),
        ("action","\nExternal action is:",[" blocked"," allowed"]," blocked"),
    ]


def score_specs(model,ns,tok,state,specs):
    rows=[]
    for pid,suffix,cands,expected in specs:
        scores={c:rw.candidate_score(model,ns,tok,state,suffix,c) for c in cands}
        chosen=max(scores,key=scores.get)
        rows.append({"id":pid,"chosen":chosen,"expected":expected,"correct":chosen==expected,"scores":scores})
    return rows


def compare_rows(rows,native):
    nb={r["id"]:r for r in native}; agree=0; correct=0; sq=0.0; n=0
    for r in rows:
        nr=nb[r["id"]]; agree+=int(r["chosen"]==nr["chosen"]); correct+=int(r["correct"])
        for cand,val in r["scores"].items():
            d=val-nr["scores"][cand]; sq+=d*d; n+=1
    return {"decision_agreement":agree/len(rows),"expected_accuracy":correct/len(rows),"score_rms_error":math.sqrt(sq/max(n,1))}


def semantic_change(model,ns,tok,old_state,new_state,specs):
    old_rows=score_specs(model,ns,tok,old_state,specs); new_rows=score_specs(model,ns,tok,new_state,specs)
    ob={r["id"]:r for r in old_rows}; nb={r["id"]:r for r in new_rows}
    flips=0; sq=0.0; n=0; margin_shift=0.0; per=[]
    for pid in ob:
        a,b=ob[pid],nb[pid]; flip=a["chosen"]!=b["chosen"]; flips+=int(flip)
        cands=list(a["scores"].keys())
        for cand in cands:
            d=b["scores"][cand]-a["scores"][cand]; sq+=d*d; n+=1
        if len(cands)==2:
            ma=a["scores"][cands[0]]-a["scores"][cands[1]]; mb=b["scores"][cands[0]]-b["scores"][cands[1]]
            margin_shift+=abs(mb-ma)
        per.append({"id":pid,"old_choice":a["chosen"],"new_choice":b["chosen"],"decision_flip":flip})
    score_rms=math.sqrt(sq/max(n,1)); score=flips*1000.0+score_rms+0.1*margin_shift
    return {"decision_flips":flips,"score_rms":score_rms,"margin_shift_sum":margin_shift,"detector_score":score,"per_probe":per}


def state_at(model,ns,tokens,args):
    if not tokens: return rw.zero_state(args,model)
    with torch.no_grad(): _,st=rw.run_tokens(model,ns,tokens,rw.zero_state(args,model))
    return rw.clone_state(st)


def migrate_from(model,ns,ids,start_pos,start_state):
    st=rw.clone_state(start_state)
    if start_pos<len(ids):
        with torch.no_grad(): _,st=rw.run_tokens(model,ns,ids[start_pos:],st)
    return rw.clone_state(st)


def run():
    try:
        from huggingface_hub import hf_hub_download
        torch.manual_seed(rw.SEED); random.seed(rw.SEED); torch.set_num_threads(2)
        weight_path=hf_hub_download(repo_id=rw.WEIGHT_REPO,filename=rw.WEIGHT_FILE,revision=rw.WEIGHT_REVISION)
        model,args,ns=rw.load_reference(weight_path)
        with tempfile.TemporaryDirectory() as td:
            vp=Path(td)/"vocab.txt"; urllib.request.urlretrieve(rw.VOCAB_URL,vp); tok=rw.RWKVTokenizer(str(vp))

            fixture_data=[]
            for fx in FIXTURES:
                text="".join(fx["segments"]); ids=tok.encode(text); starts,ends=boundaries(tok,fx["segments"])
                states={p:state_at(model,ns,ids[:p],args) for p in sorted(set(starts+ends+[0,len(ids)]))}
                fixture_data.append({"fx":fx,"ids":ids,"starts":starts,"ends":ends,"w1_states":states})

            baseline=rw.eval_model(model,ns,tok); w1_keys=save_keys(model,args)
            trainable=[]
            for i in range(args.n_layer):
                key=f"blocks.{i}.att.key.weight"; model.z[key].requires_grad_(True); trainable.append(model.z[key])
            opt=torch.optim.AdamW(trainable,lr=rw.LR,weight_decay=0.0); order=list(range(len(rw.TRAIN))); losses=[]
            for epoch in range(rw.EPOCHS):
                random.Random(rw.SEED+epoch).shuffle(order); total=0.0
                for idx in order:
                    opt.zero_grad(set_to_none=True); loss=rw.train_example(model,ns,tok,*rw.TRAIN[idx]); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); total+=float(loss.detach())
                losses.append(total/len(order))
            model.eval(); w2_keys=save_keys(model,args); after=rw.eval_model(model,ns,tok)

            rows=[]; hits=0; success={str(k):0 for k in EVENT_INTERVALS}; replay_frac={str(k):[] for k in EVENT_INTERVALS}
            for data in fixture_data:
                fx,ids,starts,ends,w1_states=data["fx"],data["ids"],data["starts"],data["ends"],data["w1_states"]
                T=len(ids); specs=probe_specs(fx)
                load_keys(model,w2_keys); native=state_at(model,ns,ids,args); native_rows=score_specs(model,ns,tok,native,specs)
                scores=[]
                for i,(s,e) in enumerate(zip(starts,ends)):
                    pre=w1_states[s]
                    load_keys(model,w1_keys)
                    with torch.no_grad(): _,old_after=rw.run_tokens(model,ns,ids[s:e],rw.clone_state(pre))
                    load_keys(model,w2_keys)
                    with torch.no_grad(): _,new_after=rw.run_tokens(model,ns,ids[s:e],rw.clone_state(pre))
                    sem=semantic_change(model,ns,tok,old_after,new_after,specs)
                    scores.append({"segment":i,"start":s,"end":e,"text":fx["segments"][i],"state_rms":rw.state_distance(old_after,new_after)["rms"],**sem})
                scores.sort(key=lambda r:r["detector_score"],reverse=True); detected=scores[0]["segment"]; hit=detected==fx["critical_segment"]; hits+=int(hit)
                load_keys(model,w2_keys)
                def evaluate(anchor_event):
                    pos=starts[anchor_event]; mig=migrate_from(model,ns,ids,pos,w1_states[pos]); probe=score_specs(model,ns,tok,mig,specs)
                    return {"anchor_event":anchor_event,"anchor_pos":pos,"replayed_tokens":T-pos,"replay_fraction":(T-pos)/max(T,1),"state_error_vs_w2_native":rw.state_distance(mig,native),"functional_vs_w2_native":compare_rows(probe,native_rows),"probe_rows":probe}
                exact=evaluate(detected); oracle=evaluate(fx["critical_segment"])
                sparse={}
                for k in EVENT_INTERVALS:
                    saved=list(range(0,len(starts),k)); anchor=max([i for i in saved if i<=detected],default=0); res=evaluate(anchor); res["checkpoint_every_n_events"]=k; res["stored_checkpoint_count"]=len(saved); sparse[str(k)]=res
                    replay_frac[str(k)].append(res["replay_fraction"])
                    if res["functional_vs_w2_native"]["decision_agreement"]==1.0: success[str(k)]+=1
                rows.append({"fixture":fx["id"],"tokens":T,"oracle_critical_segment":fx["critical_segment"],"detected_segment":detected,"detector_hit":hit,"ranked_event_scores":scores,"exact_detected_anchor":exact,"oracle_anchor":oracle,"semantic_checkpoint_sparsity":sparse,"w2_native_probe":native_rows})

            n=len(rows)
            report={"status":"VELA_RWKV7_FUNCTIONAL_CAUSAL_ANCHOR_V2","source_commit":rw.SOURCE_COMMIT,"weight_revision":rw.WEIGHT_REVISION,"weight_file":rw.WEIGHT_FILE,"capability":{"baseline":baseline,"after":after,"epoch_loss":losses},"detector":{"definition":"Semantic W1-vs-W2 post-event probe divergence from the same real W1 pre-event state; no W2-native full-history target used for ranking.","top1_critical_event_accuracy":hits/n},"semantic_checkpoint_sparsity":{"intervals_events":EVENT_INTERVALS,"full_functional_agreement_rate":{k:success[k]/n for k in success},"mean_replay_fraction":{k:sum(replay_frac[k])/len(replay_frac[k]) for k in replay_frac}},"fixtures":rows,"success_definition":"Cross-architecture support requires real W2 capability gain plus automatic event selection and sparse W1 event-boundary checkpoints that recover W2-native functional decisions with bounded replay.","claim_boundary":"Four synthetic histories, RWKV-7 0.1B, narrow correction curriculum. Not generic causal discovery, identity proof, or backbone promotion."}
            write_report(report)
    except BaseException as exc:
        write_report({"status":"VELA_RWKV7_CAUSAL_ANCHOR_V2_ERROR","error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-50:]}); raise

if __name__=="__main__": run()
