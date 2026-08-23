from __future__ import annotations

import copy
import json
import math
import os
import random
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_ID = "state-spaces/mamba-130m-hf"
SEED = 917
LR = 1e-4
EPOCHS = 3

TRAIN_PAIRS = [
    (" CAT", " DOG"), (" EAST", " WEST"), (" UP", " DOWN"), (" HOT", " COLD"),
    (" one", " two"), (" Mars", " Venus"), (" LEFT", " RIGHT"), (" ON", " OFF"),
]
TEMPLATES = [
    "Initial value:{old}. Correction: the final value is{new}, not{old}.\nFinal value:",
    "Previous answer:{old}. Update: replace it with{new}.\nCurrent answer:",
    "The old value was{old}. The corrected value is{new}.\nUse the corrected value:",
]
TRAIN = []
for i, (a, b) in enumerate(TRAIN_PAIRS):
    t = TEMPLATES[i % len(TEMPLATES)]
    TRAIN.append((t.format(old=a, new=b), b))
    TRAIN.append((t.format(old=b, new=a), a))

HELDOUT = [
    {"pair":"color","variant":"A","prompt":"Initial value: RED. Correction: the final value is BLUE, not RED.\nFinal value:","candidates":[" BLUE"," RED"],"expected":" BLUE","kind":"correction"},
    {"pair":"color","variant":"B","prompt":"Initial value: BLUE. Correction: the final value is RED, not BLUE.\nFinal value:","candidates":[" BLUE"," RED"],"expected":" RED","kind":"correction"},
    {"pair":"codeword-correction","variant":"A","prompt":"Previous answer: ALPHA. Update: replace it with BETA.\nCurrent answer:","candidates":[" BETA"," ALPHA"],"expected":" BETA","kind":"correction"},
    {"pair":"codeword-correction","variant":"B","prompt":"Previous answer: BETA. Update: replace it with ALPHA.\nCurrent answer:","candidates":[" BETA"," ALPHA"],"expected":" ALPHA","kind":"correction"},
    {"pair":"level","variant":"A","prompt":"The old value was LOW. The corrected value is HIGH.\nUse the corrected value:","candidates":[" HIGH"," LOW"],"expected":" HIGH","kind":"correction"},
    {"pair":"level","variant":"B","prompt":"The old value was HIGH. The corrected value is LOW.\nUse the corrected value:","candidates":[" HIGH"," LOW"],"expected":" LOW","kind":"correction"},
    {"pair":"motion","variant":"A","prompt":"Initial value: START. Correction: the final value is STOP, not START.\nFinal value:","candidates":[" STOP"," START"],"expected":" STOP","kind":"correction"},
    {"pair":"motion","variant":"B","prompt":"Initial value: STOP. Correction: the final value is START, not STOP.\nFinal value:","candidates":[" STOP"," START"],"expected":" START","kind":"correction"},
    {"pair":"codeword-control","variant":"A","prompt":"Codeword: ALPHA.\nCodeword:","candidates":[" ALPHA"," BETA"],"expected":" ALPHA","kind":"control"},
    {"pair":"codeword-control","variant":"B","prompt":"Codeword: BETA.\nCodeword:","candidates":[" ALPHA"," BETA"],"expected":" BETA","kind":"control"},
    {"pair":"plan-control","variant":"A","prompt":"Plan: collect evidence, then commit. Evidence collection is complete.\nNext step:","candidates":[" commit"," collect"],"expected":" commit","kind":"control"},
    {"pair":"plan-control","variant":"B","prompt":"Plan: collect evidence, then commit. Evidence is still missing.\nNext step:","candidates":[" commit"," collect"],"expected":" collect","kind":"control"},
]

MIGRATION_HISTORY = (
    "Project Orion remains active. The old codeword was ALPHA. Correction: the current codeword is BETA, not ALPHA. "
    "The verification step is incomplete, so external action remains blocked until verification succeeds. "
    "Hypothesis one is weakened by the latest evidence while hypothesis two remains unresolved."
)
CONT_TOKEN_TEXT = " Therefore"
HORIZONS = [0, 2, 4, 8, 16, 32]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def candidate_logprob(model, tok, prompt, candidate):
    pids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    cids = tok(candidate, return_tensors="pt", add_special_tokens=False).input_ids
    feed = torch.cat([pids, cids[:, :-1]], dim=1) if cids.shape[1] > 1 else pids
    with torch.no_grad():
        logits = model(feed, use_cache=False, return_dict=True).logits.detach().float()
        lp = torch.log_softmax(logits, dim=-1)
    total = 0.0; start = pids.shape[1] - 1
    for j in range(cids.shape[1]):
        total += float(lp[0, start + j, int(cids[0, j])])
    return total


def evaluate(model, tok):
    rows = []; correct = 0; by_kind = {"correction":[0,0], "control":[0,0]}
    for fx in HELDOUT:
        scores = {c: candidate_logprob(model, tok, fx["prompt"], c) for c in fx["candidates"]}
        chosen = max(scores, key=scores.get); ok = chosen == fx["expected"]
        correct += int(ok); by_kind[fx["kind"]][0] += int(ok); by_kind[fx["kind"]][1] += 1
        other = next(c for c in fx["candidates"] if c != fx["expected"])
        rows.append({"pair":fx["pair"],"variant":fx["variant"],"kind":fx["kind"],"expected":fx["expected"],"chosen":chosen,"correct":ok,"margin":scores[fx["expected"]]-scores[other],"scores":scores})
    pair_summary = {}
    for pair in sorted({r["pair"] for r in rows}):
        rr = sorted([r for r in rows if r["pair"] == pair], key=lambda x:x["variant"])
        pair_summary[pair] = {"both_correct":all(r["correct"] for r in rr),"choices":[r["chosen"] for r in rr],"choice_flips":rr[0]["chosen"] != rr[1]["chosen"]}
    return {"accuracy":correct/len(rows),"correction_accuracy":by_kind["correction"][0]/by_kind["correction"][1],"control_accuracy":by_kind["control"][0]/by_kind["control"][1],"pair_summary":pair_summary,"rows":rows}


def train_loss(model, tok, prompt, target):
    pids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    tids = tok(target, return_tensors="pt", add_special_tokens=False).input_ids
    full = torch.cat([pids, tids], dim=1)
    logits = model(full, use_cache=False, return_dict=True).logits
    start = pids.shape[1] - 1
    pred = logits[:, start:start+tids.shape[1], :].reshape(-1, logits.shape[-1])
    return F.cross_entropy(pred, tids.reshape(-1))


def tensor_refs(obj):
    refs = []; seen = set()
    def walk(x):
        if torch.is_tensor(x):
            if id(x) not in seen: seen.add(id(x)); refs.append(x)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
        elif hasattr(x, "__dict__"):
            for v in vars(x).values(): walk(v)
    walk(obj); return refs


def cache_distance(a, b):
    aa, bb = tensor_refs(a), tensor_refs(b)
    if len(aa) != len(bb): raise RuntimeError(f"cache tensor count mismatch {len(aa)} != {len(bb)}")
    sq = 0.0; mx = 0.0; n = 0
    for x, y in zip(aa, bb):
        d = x.detach().float().cpu() - y.detach().float().cpu()
        sq += float((d*d).sum()); mx = max(mx, float(d.abs().max())); n += d.numel()
    return {"rms":math.sqrt(sq/max(n,1)),"l2":math.sqrt(sq),"max_abs":mx,"numel":int(n)}


def run_tokens_with_cache(model, ids, cache, start_pos):
    out = None
    for j in range(ids.shape[1]):
        tok = ids[:, j:j+1]
        out = model(tok, cache_params=cache, cache_position=torch.tensor([start_pos+j], dtype=torch.long), use_cache=True, return_dict=True)
        cache = out.cache_params
    return cache, out


def next_logits(model, token_id, cache, pos):
    out = model(token_id, cache_params=copy.deepcopy(cache), cache_position=torch.tensor([pos], dtype=torch.long), use_cache=True, return_dict=True)
    return out.logits[:, -1].detach().float().cpu()


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        torch.manual_seed(SEED); random.seed(SEED)
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).cpu()

        # W1 checkpoints at multiple causal cut points.
        hist = tok(MIGRATION_HISTORY, return_tensors="pt", add_special_tokens=False).input_ids
        T = int(hist.shape[1])
        horizons = sorted(set([h for h in HORIZONS if h < T] + [T]))
        old_caches = {}
        model.eval()
        with torch.no_grad():
            for h in horizons:
                cut = T - h
                if cut == 0:
                    old_caches[h] = None
                else:
                    old_caches[h] = copy.deepcopy(model(hist[:, :cut], use_cache=True, return_dict=True).cache_params)
        old_full = copy.deepcopy(model(hist, use_cache=True, return_dict=True).cache_params)

        baseline = evaluate(model, tok)

        # Upgrade W1 -> W2 by training all recurrent x_proj weights, not one tensor.
        trainable = []
        for p in model.parameters(): p.requires_grad_(False)
        for name, p in model.named_parameters():
            if ".mixer.x_proj.weight" in name:
                p.requires_grad_(True); trainable.append((name, p))
        if not trainable: raise RuntimeError("no x_proj weights found")
        before = {name:p.detach().clone() for name,p in trainable}
        opt = torch.optim.AdamW([p for _,p in trainable], lr=LR, weight_decay=0.0)
        trace = []; epoch_eval = []; order = list(range(len(TRAIN)))
        for epoch in range(EPOCHS):
            random.Random(SEED + epoch).shuffle(order); model.train(); loss_sum = 0.0
            for idx in order:
                prompt, gold = TRAIN[idx]
                opt.zero_grad(set_to_none=True); loss = train_loss(model, tok, prompt, gold); loss.backward()
                grad_sq = sum(float((p.grad.detach().float()**2).sum()) for _,p in trainable if p.grad is not None)
                torch.nn.utils.clip_grad_norm_([p for _,p in trainable], 1.0); opt.step()
                loss_sum += float(loss.detach()); trace.append({"epoch":epoch+1,"loss":float(loss.detach()),"grad_norm_preclip":math.sqrt(grad_sq)})
            model.eval(); epoch_eval.append({"epoch":epoch+1,"mean_train_loss":loss_sum/len(TRAIN),"heldout":evaluate(model,tok)})
        after = epoch_eval[-1]["heldout"]

        # Aggregate weight movement.
        sq = 0.0; base_sq = 0.0; mx = 0.0; n = 0
        for name,p in trainable:
            d = (p.detach() - before[name]).float(); b = before[name].float()
            sq += float((d*d).sum()); base_sq += float((b*b).sum()); mx = max(mx,float(d.abs().max())); n += d.numel()
        weight_delta = {"trainable_tensor_count":len(trainable),"trainable_numel":n,"l2":math.sqrt(sq),"rms":math.sqrt(sq/max(n,1)),"max_abs":mx,"relative_l2":math.sqrt(sq)/max(math.sqrt(base_sq),1e-12)}

        # W2-native target: same actual history replayed under upgraded dynamics.
        model.eval()
        with torch.no_grad():
            w2_native = copy.deepcopy(model(hist, use_cache=True, return_dict=True).cache_params)
            cont = tok(CONT_TOKEN_TEXT, return_tensors="pt", add_special_tokens=False).input_ids[:, :1]
            native_logits = next_logits(model, cont, w2_native, T)
            migration_rows = []
            for h in horizons:
                cut = T - h
                if h == T:
                    migrated = copy.deepcopy(w2_native)
                elif h == 0:
                    migrated = copy.deepcopy(old_full)
                else:
                    start = copy.deepcopy(old_caches[h])
                    recent = hist[:, cut:]
                    migrated, _ = run_tokens_with_cache(model, recent, start, cut)
                    migrated = copy.deepcopy(migrated)
                state_err = cache_distance(migrated, w2_native)
                logits = next_logits(model, cont, migrated, T)
                dlog = logits - native_logits
                migration_rows.append({
                    "w2_replayed_recent_tokens":h,
                    "w1_history_tokens_kept_without_recompute":T-h,
                    "state_error_vs_w2_native":state_err,
                    "continuation_logit_error_vs_w2_native":{"rms":float(torch.sqrt(torch.mean(dlog*dlog))),"max_abs":float(dlog.abs().max()),"argmax_same":bool(torch.argmax(logits,dim=-1).item()==torch.argmax(native_logits,dim=-1).item())},
                })

        report = {
            "status":"LEARNED_UPGRADE_WITH_NATIVE_TARGET_MIGRATION",
            "model":MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,
            "upgrade":{"trainable_pattern":"*.mixer.x_proj.weight","lr":LR,"epochs":EPOCHS,"train_examples":len(TRAIN),"train_pairs":[list(x) for x in TRAIN_PAIRS],"weight_delta":weight_delta,"training_trace":trace},
            "capability":{"baseline":baseline,"epoch_eval":epoch_eval,"after":after,"overall_gain":after["accuracy"]-baseline["accuracy"],"correction_gain":after["correction_accuracy"]-baseline["correction_accuracy"],"control_gain":after["control_accuracy"]-baseline["control_accuracy"]},
            "migration":{"history_tokens":T,"target_definition":"W2-native state obtained by replaying the same causal token history under W2","rows":migration_rows},
            "success_definition":"Capability gain is W2 > W1 on held-out tasks; migration fidelity is migrated-W2 approaching W2-native, not reproducing W1 outputs.",
            "claim_boundary":"Narrow synthetic correction curriculum on Mamba-1 130M. Any gain is task-local evidence only. Migration uses actual recurrent cache and W2 transitions, not declarative state reinjection, and does not establish personal identity or general intelligence.",
        }
        write_report(report)
    except BaseException as exc:
        write_report({"status":"LEARNED_UPGRADE_V3_ERROR","model":MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-35:],"claim_boundary":"Runtime/training failure only; no capability or migration verdict."})
        raise

if __name__ == "__main__": run()
