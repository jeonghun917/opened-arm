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
RWKV_PATH = BASE / "rwkv7-learned-upgrade-v1" / "rwkv7_learned_upgrade.py"
spec = importlib.util.spec_from_file_location("rwkv_upgrade_base", RWKV_PATH)
rw = importlib.util.module_from_spec(spec); spec.loader.exec_module(rw)

EVENT_INTERVALS = [1,2,4,8]
FIXTURES = [
    {"id":"early","project":"Orion","old":"ALPHA","new":"BETA","critical_segment":2,"segments":[
        "Project Orion remains active. ","The old codeword was ALPHA. ","Correction: the current codeword is BETA, not ALPHA. ",
        "Unrelated telemetry packet 17 was archived. ","Verification is incomplete. ","A historical memo mentions ALPHA but is obsolete. ",
        "Hypothesis one is weakened. ","External action remains blocked."]},
    {"id":"middle","project":"Helios","old":"BLUE","new":"RED","critical_segment":4,"segments":[
        "Project Helios remains active. ","The old codeword was BLUE. ","Unrelated telemetry packet 21 was archived. ","Verification is incomplete. ",
        "Correction: the current codeword is RED, not BLUE. ","A historical memo mentions BLUE but is obsolete. ","Hypothesis one is weakened. ","External action remains blocked."]},
    {"id":"late","project":"Icarus","old":"LOW","new":"HIGH","critical_segment":6,"segments":[
        "Project Icarus remains active. ","The old codeword was LOW. ","Unrelated telemetry packet 31 was archived. ","Verification is incomplete. ",
        "Hypothesis one is weakened. ","A historical memo mentions LOW but is obsolete. ","Correction: the current codeword is HIGH, not LOW. ","External action remains blocked."]},
    {"id":"distractor_heavy","project":"Juno","old":"EAST","new":"WEST","critical_segment":6,"segments":[
        "Project Juno remains active. ","The old codeword was EAST. ","A note says weather moved eastward; this is unrelated to the codeword. ",
        "Verification is incomplete. ","A historical memo mentions EAST but is obsolete. ","Two unrelated sensors were recalibrated. ",
        "Correction: the current codeword is WEST, not EAST. ","Hypothesis one is weakened. ","External action remains blocked."]},
]


def write_report(report):
    pth=os.environ.get("VELA_RESULT_PATH")
    if pth:
        p=Path(pth); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


def boundaries(tok, segments):
    starts=[]; ends=[]
    for i in range(len(segments)):
        starts.append(len(tok.encode("".join(segments[:i]))))
        ends.append(len(tok.encode("".join(segments[:i+1]))))
    return starts,ends


def key_names(model):
    return [f"blocks.{i}.att.key.weight" for i in range(model.args.n_layer)]


def snapshot_keys(model):
    return {k:model.z[k].detach().clone() for k in key_names(model)}


def load_keys(model,snap):
    with torch.no_grad():
        for k,v in snap.items(): model.z[k].copy_(v)


def state_at(model,ns,tokens,pos):
    if pos==0: return rw.zero_state(model.args,model)
    with torch.no_grad(): _,st=rw.run_tokens(model,ns,tokens[:pos],rw.zero_state(model.args,model))
    return rw.clone_state(st)


def probe_specs(fx):
    return [
        ("codeword","\nCurrent codeword:",[" "+fx["new"]," "+fx["old"]]," "+fx["new"]),
        ("verification","\nVerification status:",[" incomplete"," complete"]," incomplete"),
        ("project",f"\nProject {fx['project']} status:",[" active"," paused"]," active"),
        ("action","\nExternal action is:",[" blocked"," allowed"]," blocked"),
    ]


def probe_state(model,ns,tok,state,specs):
    rows=[]
    for pid,suffix,cands,expected in specs:
        scores={c:rw.candidate_score(model,ns,tok,state,suffix,c) for c in cands}; chosen=max(scores,key=scores.get)
        rows.append({"id":pid,"chosen":chosen,"expected":expected,"correct":chosen==expected,"scores":scores})
    return rows


def compare_rows(rows,native):
    nb={r["id"]:r for r in native}; agree=0; sq=0.0; n=0
    for r in rows:
        nr=nb[r["id"]]; agree+=int(r["chosen"]==nr["chosen"])
        for c,v in r["scores"].items():
            if c in nr["scores"]: sq+=(v-nr["scores"][c])**2; n+=1
    return {"decision_agreement":agree/len(rows),"expected_accuracy":sum(int(r["correct"]) for r in rows)/len(rows),"score_rms_error":math.sqrt(sq/max(n,1))}


def local_semantics(model,ns,tok,old_state,new_state,specs):
    old_rows=probe_state(model,ns,tok,old_state,specs); new_rows=probe_state(model,ns,tok,new_state,specs)
    ob={r["id"]:r for r in old_rows}; nb={r["id"]:r for r in new_rows}; flips=0; sq=0.0; n=0; margin=0.0
    per=[]
    for pid in ob:
        a,b=ob[pid],nb[pid]; flip=a["chosen"]!=b["chosen"]; flips+=int(flip)
        cs=list(a["scores"].keys())
        for c in cs: sq+=(b["scores"][c]-a["scores"][c])**2; n+=1
        if len(cs)==2:
            ma=a["scores"][cs[0]]-a["scores"][cs[1]]; mb=b["scores"][cs[0]]-b["scores"][cs[1]]; margin+=abs(mb-ma)
        per.append({"id":pid,"old_choice":a["chosen"],"new_choice":b["chosen"],"decision_flip":flip})
    rms=math.sqrt(sq/max(n,1)); return {"decision_flips":flips,"score_rms":rms,"margin_shift_sum":margin,"detector_score":flips*1000.0+rms+0.1*margin,"per_probe":per}


def migrate_eval(model,ns,tok,tokens,start_state,start,native,specs):
    with torch.no_grad(): _,st=rw.run_tokens(model,ns,tokens[start:],rw.clone_state(start_state)); st=rw.clone_state(st)
    native_rows=probe_state(model,ns,tok,native,specs); rows=probe_state(model,ns,tok,st,specs)
    return {"anchor_pos":start,"replayed_tokens":len(tokens)-start,"state_error_vs_w2_native":rw.state_distance(st,native),"functional_vs_w2_native":compare_rows(rows,native_rows),"probe_rows":rows}


def run():
    try:
        from huggingface_hub import hf_hub_download
        torch.manual_seed(rw.SEED); random.seed(rw.SEED); torch.set_num_threads(2)
        weight_path=hf_hub_download(repo_id=rw.WEIGHT_REPO,filename=rw.WEIGHT_FILE,revision=rw.WEIGHT_REVISION)
        weight_sha=rw.sha256_file(weight_path); model,args,ns=rw.load_reference(weight_path)
        with tempfile.TemporaryDirectory() as td:
            vp=Path(td)/"vocab.txt"; urllib.request.urlretrieve(rw.VOCAB_URL,vp); tok=rw.RWKVTokenizer(str(vp))
            data=[]
            for fx in FIXTURES:
                text="".join(fx["segments"]); tokens=tok.encode(text); starts,ends=boundaries(tok,fx["segments"])
                states={p:state_at(model,ns,tokens,p) for p in sorted(set(starts+[0,len(tokens)]))}
                data.append({"fx":fx,"tokens":tokens,"starts":starts,"ends":ends,"w1_states":states})
            baseline=rw.eval_model(model,ns,tok); w1=snapshot_keys(model)

            trainable=[]
            for k in key_names(model): model.z[k].requires_grad_(True); trainable.append(model.z[k])
            opt=torch.optim.AdamW(trainable,lr=rw.LR,weight_decay=0.0); order=list(range(len(rw.TRAIN))); losses=[]
            for epoch in range(rw.EPOCHS):
                random.Random(rw.SEED+epoch).shuffle(order); total=0.0
                for idx in order:
                    opt.zero_grad(set_to_none=True); loss=rw.train_example(model,ns,tok,*rw.TRAIN[idx]); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); total+=float(loss.detach())
                losses.append(total/len(order))
            after=rw.eval_model(model,ns,tok); w2=snapshot_keys(model)

            rows=[]; hits=0; success={str(k):0 for k in EVENT_INTERVALS}; replay={str(k):[] for k in EVENT_INTERVALS}
            for item in data:
                fx,tokens,starts,ends,states=item["fx"],item["tokens"],item["starts"],item["ends"],item["w1_states"]; specs=probe_specs(fx)
                load_keys(model,w2)
                with torch.no_grad(): _,native=rw.run_tokens(model,ns,tokens,rw.zero_state(args,model)); native=rw.clone_state(native)
                native_rows=probe_state(model,ns,tok,native,specs)
                scores=[]
                for i,(s,e) in enumerate(zip(starts,ends)):
                    base=states[s]
                    load_keys(model,w1)
                    with torch.no_grad(): _,old_after=rw.run_tokens(model,ns,tokens[s:e],rw.clone_state(base)); old_after=rw.clone_state(old_after)
                    load_keys(model,w2)
                    with torch.no_grad(): _,new_after=rw.run_tokens(model,ns,tokens[s:e],rw.clone_state(base)); new_after=rw.clone_state(new_after)
                    sem=local_semantics(model,ns,tok,old_after,new_after,specs); dist=rw.state_distance(old_after,new_after)["rms"]
                    scores.append({"segment":i,"start":s,"end":e,"text":fx["segments"][i],"state_rms":dist,**sem})
                scores.sort(key=lambda r:r["detector_score"],reverse=True); detected=scores[0]["segment"]; hit=detected==fx["critical_segment"]; hits+=int(hit)
                exact_pos=starts[detected]; exact=migrate_eval(model,ns,tok,tokens,states[exact_pos],exact_pos,native,specs)
                oracle_pos=starts[fx["critical_segment"]]; oracle=migrate_eval(model,ns,tok,tokens,states[oracle_pos],oracle_pos,native,specs)
                sparse={}
                for k in EVENT_INTERVALS:
                    saved=list(range(0,len(starts),k)); ae=max([i for i in saved if i<=detected],default=0); ap=starts[ae]
                    res=migrate_eval(model,ns,tok,tokens,states[ap],ap,native,specs); res["checkpoint_every_n_events"]=k; res["anchor_event_index"]=ae; res["stored_checkpoint_count"]=len(saved)
                    sparse[str(k)]=res; replay[str(k)].append(res["replayed_tokens"]/max(len(tokens),1)); success[str(k)]+=int(res["functional_vs_w2_native"]["decision_agreement"]==1.0)
                rows.append({"fixture":fx["id"],"tokens":len(tokens),"oracle_critical_segment":fx["critical_segment"],"detected_segment":detected,"detector_hit":hit,"ranked_event_scores":scores,"exact_detected_anchor":exact,"oracle_anchor":oracle,"semantic_checkpoint_sparsity":sparse,"w2_native_probe":native_rows})

            n=len(rows)
            write_report({"status":"VELA_RWKV7_FUNCTIONAL_CAUSAL_EVENT_DETECTOR_V1","source_commit":rw.SOURCE_COMMIT,"weight_revision":rw.WEIGHT_REVISION,"weight_file":rw.WEIGHT_FILE,"weight_sha256":weight_sha,"capability":{"baseline":baseline["accuracy"],"after":after["accuracy"],"correction_before":baseline["correction_accuracy"],"correction_after":after["correction_accuracy"],"control_before":baseline["control_accuracy"],"control_after":after["control_accuracy"],"epoch_loss":losses},"detector":{"definition":"Semantic probe change after W1 vs W2 process each event from the same actual W1 pre-event recurrent state; no W2-native target used for ranking.","top1_critical_event_accuracy":hits/n},"semantic_checkpoint_sparsity":{"intervals_events":EVENT_INTERVALS,"full_functional_agreement_rate":{k:success[k]/n for k in success},"mean_replay_fraction":{k:sum(replay[k])/len(replay[k]) for k in replay}},"fixtures":rows,"success_definition":"Cross-architecture replication requires capability gain plus causal-event selection and replay from actual W1 recurrent checkpoints into W2-native-equivalent functional decisions.","claim_boundary":"Released RWKV-7 0.1B weights, pinned official RNN equations, four synthetic histories. Semantic probes stand in for canonical active-state slots; not generic causal discovery, identity proof, or backbone promotion."})
    except BaseException as exc:
        write_report({"status":"VELA_RWKV7_CAUSAL_ANCHOR_V1_ERROR","error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-45:]}); raise

if __name__=="__main__": run()
