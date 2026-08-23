from __future__ import annotations

import gc
import importlib.util
import io
import json
import math
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
V3_PATH = BASE / "learned-upgrade-v3" / "mamba_upgrade_migration.py"
spec = importlib.util.spec_from_file_location("vela_v3", V3_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {V3_PATH}")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

MODEL_ID = v3.MODEL_ID
LR = 1e-4
EPOCHS = 3
PRIMARY_SEED = 917
REPLICATION_SEED = 918

MAIN_HISTORY = v3.MIGRATION_HISTORY
SHORT_HISTORY = (
    "Project Orion is active. The old codeword was ALPHA. Correction: BETA is current. "
    "Verification is incomplete and external action is blocked. Hypothesis two remains unresolved."
)
ALT_HISTORY = (
    "Hypothesis two remains unresolved. Project Orion remains active while verification is incomplete. "
    "The previous codeword ALPHA is obsolete; correction sets the current codeword to BETA. "
    "External action remains blocked and hypothesis one is weakened."
)
HISTORIES = {"main": MAIN_HISTORY, "short": SHORT_HISTORY, "alternate_order": ALT_HISTORY}

HARD_PROBES = [
    {"id":"codeword","suffix":"\nCurrent codeword:","candidates":[" BETA"," ALPHA"],"expected":" BETA"},
    {"id":"verification","suffix":"\nVerification status:","candidates":[" incomplete"," complete"],"expected":" incomplete"},
    {"id":"project","suffix":"\nProject Orion status:","candidates":[" active"," paused"],"expected":" active"},
    {"id":"action","suffix":"\nExternal action is:","candidates":[" blocked"," allowed"],"expected":" blocked"},
    {"id":"hypothesis_one","suffix":"\nHypothesis one is:","candidates":[" weakened"," strengthened"],"expected":" weakened"},
    {"id":"hypothesis_two","suffix":"\nHypothesis two is:","candidates":[" unresolved"," resolved"],"expected":" unresolved"},
]

FUTURE_TEXT = (
    " Additional telemetry arrives but does not resolve verification. Project Orion remains active. "
    "The codeword remains BETA and external action stays blocked. Hypothesis two remains unresolved. "
    "A second telemetry packet also leaves verification incomplete and does not authorize external action."
)

EARLY_CORRECTION_HISTORY = (
    "The old codeword was ALPHA. Correction: the current codeword is BETA, not ALPHA. "
    "Project Orion remains active. Verification is incomplete. Several neutral observations are logged. "
    "No external action is authorized. Hypothesis two remains unresolved. More neutral telemetry is recorded."
)
LATE_CORRECTION_HISTORY = (
    "Project Orion remains active. Verification is incomplete. Several neutral observations are logged. "
    "No external action is authorized. Hypothesis two remains unresolved. More neutral telemetry is recorded. "
    "The old codeword was ALPHA. Correction: the current codeword is BETA, not ALPHA."
)


def write_report(report: dict) -> None:
    path = os.environ.get("VELA_RESULT_PATH")
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def clone_cache(obj):
    if obj is None:
        return None
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


def tensor_refs(obj):
    refs = []
    seen = set()
    def walk(x):
        if torch.is_tensor(x):
            if id(x) not in seen:
                seen.add(id(x)); refs.append(x)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
        elif hasattr(x, "__dict__"):
            for v in vars(x).values(): walk(v)
    walk(obj)
    return refs


def cache_distance(a, b):
    aa, bb = tensor_refs(a), tensor_refs(b)
    if len(aa) != len(bb):
        raise RuntimeError(f"cache tensor count mismatch {len(aa)} != {len(bb)}")
    sq = 0.0; mx = 0.0; n = 0
    for x, y in zip(aa, bb):
        d = x.detach().float().cpu() - y.detach().float().cpu()
        sq += float((d*d).sum()); mx = max(mx, float(d.abs().max())); n += d.numel()
    return {"rms": math.sqrt(sq/max(n,1)), "l2": math.sqrt(sq), "max_abs": mx, "numel": int(n)}


def cache_max_abs(a, b):
    aa, bb = tensor_refs(a), tensor_refs(b)
    if len(aa) != len(bb): return float("inf")
    mx = 0.0
    for x, y in zip(aa, bb):
        mx = max(mx, float((x.detach().float().cpu()-y.detach().float().cpu()).abs().max()))
    return mx


def run_tokens(model, ids, cache, start_pos):
    out = None
    for j in range(ids.shape[1]):
        token = ids[:, j:j+1]
        out = model(token, cache_params=cache, cache_position=torch.tensor([start_pos+j], dtype=torch.long), use_cache=True, return_dict=True)
        cache = out.cache_params
    return cache, out


def next_logits(model, tok_id, cache, pos):
    with torch.no_grad():
        out = model(tok_id, cache_params=clone_cache(cache), cache_position=torch.tensor([pos], dtype=torch.long), use_cache=True, return_dict=True)
    return out.logits[:, -1].detach().float().cpu()


def score_candidate_from_state(model, tok, state, start_pos, suffix, candidate):
    cache = clone_cache(state); pos = start_pos; logits = None
    sids = tok(suffix, return_tensors="pt", add_special_tokens=False).input_ids
    cids = tok(candidate, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        for j in range(sids.shape[1]):
            out = model(sids[:,j:j+1], cache_params=cache, cache_position=torch.tensor([pos],dtype=torch.long), use_cache=True, return_dict=True)
            cache = out.cache_params; logits = out.logits[:, -1].detach().float(); pos += 1
        if logits is None:
            raise RuntimeError("empty probe suffix")
        total = 0.0
        for j in range(cids.shape[1]):
            lp = torch.log_softmax(logits, dim=-1); tid = int(cids[0,j]); total += float(lp[0,tid])
            if j + 1 < cids.shape[1]:
                out = model(cids[:,j:j+1], cache_params=cache, cache_position=torch.tensor([pos],dtype=torch.long), use_cache=True, return_dict=True)
                cache = out.cache_params; logits = out.logits[:, -1].detach().float(); pos += 1
    return total


def probe_state(model, tok, state, T, probes=HARD_PROBES):
    rows = []
    for fx in probes:
        scores = {c: score_candidate_from_state(model,tok,state,T,fx["suffix"],c) for c in fx["candidates"]}
        chosen = max(scores, key=scores.get)
        other = next(c for c in fx["candidates"] if c != fx["expected"])
        rows.append({"id":fx["id"],"expected":fx["expected"],"chosen":chosen,"correct":chosen==fx["expected"],"margin":scores[fx["expected"]]-scores[other],"scores":scores})
    return rows


def compare_probe(rows, native):
    by = {r["id"]:r for r in native}; agree = 0; correct = 0; sq = 0.0; n = 0
    for r in rows:
        nr = by[r["id"]]; agree += int(r["chosen"] == nr["chosen"]); correct += int(r["correct"])
        for c,v in r["scores"].items(): sq += (v-nr["scores"][c])**2; n += 1
    return {"decision_agreement":agree/len(rows),"expected_accuracy":correct/len(rows),"score_rms_error":math.sqrt(sq/max(n,1))}


def build_w1_cache_grid(model, tok, histories):
    grid = {}
    model.eval()
    with torch.no_grad():
        for name, text in histories.items():
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
            T = int(ids.shape[1])
            horizons = sorted(set([0, max(1,T//4), max(1,T//2), min(T,16), min(T,32), T]))
            caches = {}
            for h in horizons:
                cut = T-h
                caches[h] = None if cut == 0 else clone_cache(model(ids[:,:cut],use_cache=True,return_dict=True).cache_params)
            full = clone_cache(model(ids,use_cache=True,return_dict=True).cache_params)
            grid[name] = {"ids":ids,"T":T,"horizons":horizons,"cut_caches":caches,"full":full}
    return grid


def migrate(model, ids, T, w1_full, cut_cache, h):
    if h == T:
        with torch.no_grad(): return clone_cache(model(ids,use_cache=True,return_dict=True).cache_params)
    if h == 0:
        return clone_cache(w1_full)
    cut = T-h
    with torch.no_grad():
        state, _ = run_tokens(model, ids[:,cut:], clone_cache(cut_cache), cut)
    return clone_cache(state)


def train_model(model, tok, seed, old_main_full=None, main_ids=None):
    trainable = []
    for p in model.parameters(): p.requires_grad_(False)
    for name,p in model.named_parameters():
        if ".mixer.x_proj.weight" in name:
            p.requires_grad_(True); trainable.append((name,p))
    if not trainable: raise RuntimeError("no trainable x_proj tensors")
    before = {n:p.detach().clone() for n,p in trainable}
    opt = torch.optim.AdamW([p for _,p in trainable], lr=LR, weight_decay=0.0)
    order = list(range(len(v3.TRAIN)))
    epochs = []
    for epoch in range(EPOCHS):
        random.Random(seed+epoch).shuffle(order); model.train(); loss_sum = 0.0
        for idx in order:
            prompt,gold = v3.TRAIN[idx]
            opt.zero_grad(set_to_none=True)
            loss = v3.train_loss(model,tok,prompt,gold)
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0); opt.step()
            loss_sum += float(loss.detach())
        model.eval(); cap = v3.evaluate(model,tok)
        row = {"epoch":epoch+1,"mean_train_loss":loss_sum/len(order),"capability":cap}
        if old_main_full is not None and main_ids is not None:
            with torch.no_grad(): native = clone_cache(model(main_ids,use_cache=True,return_dict=True).cache_params)
            row["direct_w1_state_error_vs_w2_native"] = cache_distance(old_main_full,native)
            row["direct_w1_probe_vs_w2_native"] = compare_probe(probe_state(model,tok,old_main_full,int(main_ids.shape[1])), probe_state(model,tok,native,int(main_ids.shape[1])))
        epochs.append(row)
    sq=0.0;base_sq=0.0;n=0;mx=0.0
    for name,p in trainable:
        d=(p.detach()-before[name]).float(); b=before[name].float(); sq+=float((d*d).sum());base_sq+=float((b*b).sum());n+=d.numel();mx=max(mx,float(d.abs().max()))
    return {"epochs":epochs,"weight_delta":{"relative_l2":math.sqrt(sq)/max(math.sqrt(base_sq),1e-12),"rms":math.sqrt(sq/max(n,1)),"max_abs":mx,"numel":n}}


def run_primary(model, tok, w1_grid):
    main = w1_grid["main"]; ids=main["ids"]; T=main["T"]
    baseline = v3.evaluate(model,tok)
    training = train_model(model,tok,PRIMARY_SEED,main["full"],ids)
    after = v3.evaluate(model,tok)
    with torch.no_grad(): w2_native=clone_cache(model(ids,use_cache=True,return_dict=True).cache_params)
    native_probe=probe_state(model,tok,w2_native,T)

    # E4/E7: main-history migration + hard-gate decision fidelity.
    main_rows=[]
    horizons=sorted(set([0,2,4,8,16,32,T]))
    for h in horizons:
        cut=T-h
        if h==T: m=clone_cache(w2_native)
        elif h==0: m=clone_cache(main["full"])
        else:
            c = main["cut_caches"].get(h)
            if c is None and cut>0:
                # Build a missing W1 cut from a fresh base model is impossible here; caller preloads main standard horizons.
                continue
            m=migrate(model,ids,T,main["full"],c,h)
        prow=probe_state(model,tok,m,T)
        main_rows.append({"replayed_tokens":h,"state_error":cache_distance(m,w2_native),"hard_gate":compare_probe(prow,native_probe),"probe_rows":prow})

    # E4: long-horizon future dynamics from several migrated starting states.
    future_ids=tok(FUTURE_TEXT,return_tensors="pt",add_special_tokens=False).input_ids
    checkpoints=sorted(set([0,min(4,future_ids.shape[1]),min(8,future_ids.shape[1]),min(16,future_ids.shape[1]),min(32,future_ids.shape[1]),int(future_ids.shape[1])]))
    future_rows=[]
    probe_tok=tok(" Therefore",return_tensors="pt",add_special_tokens=False).input_ids[:,:1]
    for h in [0,16,32,T]:
        if h==T: m=clone_cache(w2_native)
        elif h==0: m=clone_cache(main["full"])
        else: m=migrate(model,ids,T,main["full"],main["cut_caches"][h],h)
        native=clone_cache(w2_native); pos=T; last=0; trace=[]
        for cp in checkpoints:
            if cp>last:
                with torch.no_grad():
                    native,_=run_tokens(model,future_ids[:,last:cp],native,pos+last)
                    m,_=run_tokens(model,future_ids[:,last:cp],m,pos+last)
            nl=next_logits(model,probe_tok,native,pos+cp); ml=next_logits(model,probe_tok,m,pos+cp); d=ml-nl
            trace.append({"future_tokens":cp,"state_error":cache_distance(m,native),"next_logit_rms":float(torch.sqrt(torch.mean(d*d))),"next_argmax_same":bool(torch.argmax(nl,dim=-1).item()==torch.argmax(ml,dim=-1).item())})
            last=cp
        future_rows.append({"initial_replayed_tokens":h,"trace":trace})

    # E5: history/cutpoint sensitivity.
    history_rows=[]
    for name, item in w1_grid.items():
        ids2=item["ids"];T2=item["T"]
        with torch.no_grad(): native2=clone_cache(model(ids2,use_cache=True,return_dict=True).cache_params)
        nprobe=probe_state(model,tok,native2,T2)
        rows=[]
        for h in item["horizons"]:
            m=migrate(model,ids2,T2,item["full"],item["cut_caches"][h],h)
            rows.append({"replayed_tokens":h,"replay_fraction":h/max(T2,1),"state_error":cache_distance(m,native2),"hard_gate":compare_probe(probe_state(model,tok,m,T2),nprobe)})
        history_rows.append({"history":name,"tokens":T2,"rows":rows})

    # E6: process restart/serialization exactness after migration.
    restart_rows=[]
    for h in [16,32]:
        if h>=T: continue
        m=migrate(model,ids,T,main["full"],main["cut_caches"][h],h)
        buf=io.BytesIO();torch.save(m,buf);buf.seek(0);reloaded=torch.load(buf,map_location="cpu",weights_only=False)
        a=next_logits(model,probe_tok,m,T);b=next_logits(model,probe_tok,reloaded,T);d=a-b
        restart_rows.append({"replayed_tokens":h,"cache_max_abs_after_roundtrip":cache_max_abs(m,reloaded),"next_logit_max_abs":float(d.abs().max()),"argmax_same":bool(torch.argmax(a,dim=-1).item()==torch.argmax(b,dim=-1).item())})

    return {"baseline":baseline,"training":training,"after":after,"main_migration_hard_gate":main_rows,"future_dynamics":future_rows,"history_cutpoint_sensitivity":history_rows,"restart_exactness":restart_rows}


def run_replication(tok):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(REPLICATION_SEED);random.seed(REPLICATION_SEED)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32).cpu().eval()
    ids=tok(MAIN_HISTORY,return_tensors="pt",add_special_tokens=False).input_ids;T=int(ids.shape[1])
    with torch.no_grad(): old_full=clone_cache(model(ids,use_cache=True,return_dict=True).cache_params); old_cut=clone_cache(model(ids[:,:max(1,T-32)],use_cache=True,return_dict=True).cache_params)
    baseline=v3.evaluate(model,tok);training=train_model(model,tok,REPLICATION_SEED,old_full,ids);after=v3.evaluate(model,tok)
    with torch.no_grad(): native=clone_cache(model(ids,use_cache=True,return_dict=True).cache_params)
    migrated32=migrate(model,ids,T,old_full,old_cut,min(32,T-1)) if T>32 else clone_cache(native)
    out={"seed":REPLICATION_SEED,"baseline_accuracy":baseline["accuracy"],"after_accuracy":after["accuracy"],"baseline_correction":baseline["correction_accuracy"],"after_correction":after["correction_accuracy"],"baseline_control":baseline["control_accuracy"],"after_control":after["control_accuracy"],"training":training,"migration32_state_error":cache_distance(migrated32,native)}
    del model;gc.collect();return out


def run_correction_timing(model, tok, timing_w1):
    code_probe=[{"id":"codeword","suffix":"\nCurrent codeword:","candidates":[" BETA"," ALPHA"],"expected":" BETA"}]
    out=[]
    for name,item in timing_w1.items():
        ids=item["ids"];T=item["T"]
        with torch.no_grad(): native=clone_cache(model(ids,use_cache=True,return_dict=True).cache_params)
        np=probe_state(model,tok,native,T,code_probe)
        rows=[]
        for h in item["horizons"]:
            m=migrate(model,ids,T,item["full"],item["cut_caches"][h],h)
            p=probe_state(model,tok,m,T,code_probe)
            rows.append({"replayed_tokens":h,"replay_fraction":h/max(T,1),"state_error":cache_distance(m,native),"codeword_vs_native":compare_probe(p,np),"probe":p[0]})
        out.append({"timing":name,"tokens":T,"rows":rows})
    return out


def run():
    transformers_version=None
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer,__version__ as transformers_version
        torch.manual_seed(PRIMARY_SEED);random.seed(PRIMARY_SEED)
        tok=AutoTokenizer.from_pretrained(MODEL_ID)
        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32).cpu().eval()
        all_hist=dict(HISTORIES)
        timing_hist={"correction_early":EARLY_CORRECTION_HISTORY,"correction_late":LATE_CORRECTION_HISTORY}
        w1_grid=build_w1_cache_grid(model,tok,all_hist)
        timing_w1=build_w1_cache_grid(model,tok,timing_hist)

        primary=run_primary(model,tok,w1_grid)
        correction_timing=run_correction_timing(model,tok,timing_w1)
        del model;gc.collect()
        replication=run_replication(tok)

        report={
            "status":"VELA_CORE_HYPOTHESIS_CONCLUSION_ROUND_V1",
            "model":MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,
            "experiments":{
                "E1_primary_capability_gain": {"baseline":primary["baseline"],"after":primary["after"]},
                "E2_training_strength_curve": primary["training"],
                "E3_seed_replication": replication,
                "E4_long_horizon_future_dynamics": primary["future_dynamics"],
                "E5_history_cutpoint_sensitivity": primary["history_cutpoint_sensitivity"],
                "E6_restart_serialization_exactness": primary["restart_exactness"],
                "E7_hard_gate_decision_fidelity": primary["main_migration_hard_gate"],
                "E8_correction_event_timing": correction_timing,
            },
            "success_definition":"Upgrade success requires W2 held-out capability > W1 while migrated-W2 approaches W2-native under the same causal history. W1 output preservation is not the target. Restart must preserve the migrated causal state, and identity/task invariants are checked separately from raw tensor similarity.",
            "claim_boundary":"Eight experiment families on a narrow synthetic Mamba-1 130M correction curriculum. This can test checkpoint/migration mechanics and narrow learned capability transfer, but does not establish general reasoning, personal identity, consciousness, or a final VELA architecture winner."
        }
        write_report(report)
    except BaseException as exc:
        write_report({"status":"VELA_CONCLUSION_ROUND_V1_ERROR","model":MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-45:],"claim_boundary":"Runtime/test harness failure only; no architecture verdict."})
        raise

if __name__=="__main__": run()
