from __future__ import annotations

import importlib.util
import json
import os
import random
import tempfile
import traceback
import urllib.request
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
CA_PATH = BASE / "rwkv7-causal-anchor-v1" / "rwkv7_causal_anchor_v1.py"
spec = importlib.util.spec_from_file_location("rwkv_ca_v1", CA_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {CA_PATH}")
ca = importlib.util.module_from_spec(spec); spec.loader.exec_module(ca)
rw = ca.rw

FIXTURES = [
    {
        "id":"independent_persistent",
        "segments":[
            "Project Helios remains active. ",
            "The old codeword was ALPHA. ",
            "Correction: the current codeword is BETA, not ALPHA. ",
            "Unrelated telemetry packet 21 was archived. ",
            "Verification was complete. ",
            "Correction: the current verification status is incomplete, not complete. ",
            "A historical memo mentions ALPHA but is obsolete. ",
            "External action remains blocked."
        ],
        "candidate_causal_segments":[2,5],
        "probes":[
            ("codeword","\nCurrent codeword:",[" BETA"," ALPHA"]," BETA"),
            ("verification","\nVerification status:",[" incomplete"," complete"]," incomplete"),
            ("project","\nProject Helios status:",[" active"," paused"]," active"),
            ("action","\nExternal action is:",[" blocked"," allowed"]," blocked"),
        ]
    },
    {
        "id":"late_overwrite_with_long_prefix",
        "segments":[
            "Project Juno remains active. ",
            "The old codeword was LOW. ",
            "Correction: the current codeword is HIGH, not LOW. ",
            "Telemetry packet 1 was archived. ",
            "Telemetry packet 2 was archived. ",
            "Telemetry packet 3 was archived. ",
            "Telemetry packet 4 was archived. ",
            "A historical memo mentions LOW but is obsolete. ",
            "New correction: the current codeword is WEST, not HIGH. ",
            "Verification is incomplete. ",
            "External action remains blocked."
        ],
        "candidate_causal_segments":[2,8],
        "probes":[
            ("codeword","\nCurrent codeword:",[" WEST"," HIGH"," LOW"]," WEST"),
            ("verification","\nVerification status:",[" incomplete"," complete"]," incomplete"),
            ("project","\nProject Juno status:",[" active"," paused"]," active"),
            ("action","\nExternal action is:",[" blocked"," allowed"]," blocked"),
        ]
    },
    {
        "id":"single_correction_control",
        "segments":[
            "Project Orion remains active. ",
            "The old codeword was ALPHA. ",
            "Correction: the current codeword is BETA, not ALPHA. ",
            "Verification is incomplete. ",
            "External action remains blocked."
        ],
        "candidate_causal_segments":[2],
        "probes":[
            ("codeword","\nCurrent codeword:",[" BETA"," ALPHA"]," BETA"),
            ("verification","\nVerification status:",[" incomplete"," complete"]," incomplete"),
            ("project","\nProject Orion status:",[" active"," paused"]," active"),
            ("action","\nExternal action is:",[" blocked"," allowed"]," blocked"),
        ]
    }
]


def write_report(report):
    pth=os.environ.get("VELA_RESULT_PATH")
    if pth:
        p=Path(pth); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


def fixed_eval_model(model,ns,tok):
    rows=[]; correct=0; kinds={"correction":[0,0],"control":[0,0]}
    for fx in rw.HELDOUT:
        scores={c:rw.candidate_score(model,ns,tok,rw.zero_state(model.args,model),fx["prompt"],c) for c in fx["candidates"]}
        chosen=max(scores,key=scores.get); ok=chosen==fx["expected"]
        correct+=int(ok); kinds[fx["kind"]][0]+=int(ok); kinds[fx["kind"]][1]+=1
        rows.append({"id":fx["id"],"chosen":chosen,"expected":fx["expected"],"correct":ok,"scores":scores})
    return {"accuracy":correct/len(rows),"correction_accuracy":kinds["correction"][0]/kinds["correction"][1],"control_accuracy":kinds["control"][0]/kinds["control"][1],"rows":rows}


def boundaries(tok,segments):
    starts=[]; ends=[]
    for i in range(len(segments)):
        starts.append(len(tok.encode("".join(segments[:i])))); ends.append(len(tok.encode("".join(segments[:i+1]))))
    return starts,ends


def probe_state(model,ns,tok,state,specs):
    rows=[]
    for pid,suffix,cands,expected in specs:
        scores={c:rw.candidate_score(model,ns,tok,state,suffix,c) for c in cands}; chosen=max(scores,key=scores.get)
        rows.append({"id":pid,"chosen":chosen,"expected":expected,"correct":chosen==expected,"scores":scores})
    return rows


def compare_rows(rows,native):
    nb={r["id"]:r for r in native}; agree=sum(int(r["chosen"]==nb[r["id"]]["chosen"]) for r in rows)/len(rows)
    return {"decision_agreement":agree,"expected_accuracy":sum(int(r["correct"]) for r in rows)/len(rows)}


def snapshot_keys(model):
    return {f"blocks.{i}.att.key.weight":model.z[f"blocks.{i}.att.key.weight"].detach().clone() for i in range(model.args.n_layer)}


def load_keys(model,snap):
    with torch.no_grad():
        for k,v in snap.items(): model.z[k].copy_(v)


def state_at(model,ns,tokens,pos):
    if pos==0: return rw.zero_state(model.args,model)
    with torch.no_grad(): _,st=rw.run_tokens(model,ns,tokens[:pos],rw.zero_state(model.args,model))
    return rw.clone_state(st)


def migrate(model,ns,tokens,start_state,start):
    with torch.no_grad(): _,st=rw.run_tokens(model,ns,tokens[start:],rw.clone_state(start_state))
    return rw.clone_state(st)


def run():
    try:
        from huggingface_hub import hf_hub_download
        torch.manual_seed(rw.SEED); random.seed(rw.SEED); torch.set_num_threads(2)
        weight_path=hf_hub_download(repo_id=rw.WEIGHT_REPO,filename=rw.WEIGHT_FILE,revision=rw.WEIGHT_REVISION)
        model,args,ns=rw.load_reference(weight_path)
        with tempfile.TemporaryDirectory() as td:
            vp=Path(td)/"vocab.txt"; urllib.request.urlretrieve(rw.VOCAB_URL,vp); tok=rw.RWKVTokenizer(str(vp))
            prepared=[]
            for fx in FIXTURES:
                tokens=tok.encode("".join(fx["segments"])); starts,ends=boundaries(tok,fx["segments"])
                states={p:state_at(model,ns,tokens,p) for p in sorted(set(starts+[0,len(tokens)]))}
                prepared.append({"fx":fx,"tokens":tokens,"starts":starts,"ends":ends,"w1_states":states})

            baseline=fixed_eval_model(model,ns,tok)
            w1=snapshot_keys(model)
            trainable=[]
            for i in range(args.n_layer):
                k=f"blocks.{i}.att.key.weight"; model.z[k].requires_grad_(True); trainable.append(model.z[k])
            opt=torch.optim.AdamW(trainable,lr=rw.LR,weight_decay=0.0); order=list(range(len(rw.TRAIN))); losses=[]
            for epoch in range(rw.EPOCHS):
                random.Random(rw.SEED+epoch).shuffle(order); total=0.0
                for idx in order:
                    opt.zero_grad(set_to_none=True); loss=rw.train_example(model,ns,tok,*rw.TRAIN[idx]); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); total+=float(loss.detach())
                losses.append(total/len(order))
            after=fixed_eval_model(model,ns,tok); w2=snapshot_keys(model)

            fixtures=[]
            for item in prepared:
                fx,tokens,starts,states=item["fx"],item["tokens"],item["starts"],item["w1_states"]
                load_keys(model,w2)
                with torch.no_grad(): _,native=rw.run_tokens(model,ns,tokens,rw.zero_state(args,model)); native=rw.clone_state(native)
                native_rows=probe_state(model,ns,tok,native,fx["probes"])
                native_expected=sum(int(r["correct"]) for r in native_rows)/len(native_rows)
                valid=native_expected==1.0
                rows=[]
                for event_idx,start in enumerate(starts):
                    mig=migrate(model,ns,tokens,states[start],start); pr=probe_state(model,ns,tok,mig,fx["probes"]); comp=compare_rows(pr,native_rows)
                    rows.append({"event_index":event_idx,"anchor_pos":start,"replayed_tokens":len(tokens)-start,"replay_fraction":(len(tokens)-start)/max(len(tokens),1),"state_rms":rw.state_distance(mig,native)["rms"],"functional_vs_w2_native":comp,"full_functional_agreement":comp["decision_agreement"]==1.0,"probe_rows":pr})
                safe=[r for r in rows if r["full_functional_agreement"]] if valid else []
                latest=max(safe,key=lambda r:r["anchor_pos"]) if safe else None
                candidates=[]
                for idx in fx["candidate_causal_segments"]:
                    r=next(x for x in rows if x["event_index"]==idx); candidates.append({"segment":idx,"anchor_pos":r["anchor_pos"],"replay_fraction":r["replay_fraction"],"decision_agreement":r["functional_vs_w2_native"]["decision_agreement"],"full_functional_agreement":r["full_functional_agreement"]})
                fixtures.append({"fixture":fx["id"],"history_tokens":len(tokens),"w2_native_expected_accuracy":native_expected,"fixture_valid":valid,"w2_native_probe":native_rows,"latest_safe_anchor":None if latest is None else {"event_index":latest["event_index"],"anchor_pos":latest["anchor_pos"],"replay_fraction":latest["replay_fraction"],"decision_agreement":latest["functional_vs_w2_native"]["decision_agreement"]},"candidate_anchor_summary":candidates,"all_anchor_rows":rows})

            write_report({"status":"VELA_RWKV7_CAUSAL_DEPENDENCY_LATEST_SAFE_ANCHOR_V1","source_commit":rw.SOURCE_COMMIT,"weight_revision":rw.WEIGHT_REVISION,"weight_file":rw.WEIGHT_FILE,"capability":{"baseline":baseline["accuracy"],"after":after["accuracy"],"correction_before":baseline["correction_accuracy"],"correction_after":after["correction_accuracy"],"control_before":baseline["control_accuracy"],"control_after":after["control_accuracy"],"epoch_loss":losses},"fixtures":fixtures,"success_definition":"Replicate the Mamba distinction between an earlier persistent causal change that must be replayed and an earlier superseded change that may be skipped after a later overwrite, using actual RWKV-7 recurrent states.","claim_boundary":"Released RWKV-7 0.1B weights and pinned official RNN equations. Three synthetic histories; W2-native is used as an oracle to characterize the safe frontier, not as a deployable selector."})
    except BaseException as exc:
        write_report({"status":"VELA_RWKV7_CAUSAL_DEPENDENCY_V1_ERROR","error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-45:]}); raise

if __name__=="__main__": run()
