from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
V4_PATH = BASE / "learned-upgrade-v4" / "mamba_upgrade_decision_fidelity.py"
spec = importlib.util.spec_from_file_location("vela_v4", V4_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {V4_PATH}")
v4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4)
v3 = v4.v3

LAMBDA = float(os.environ.get("VELA_COMPAT_LAMBDA", "0.03"))
COMPAT_EVERY = 4
COMPAT_TEXTS = [
    "Project Aster remains active. The old codeword was CAT. Correction: the current codeword is DOG, not CAT. Verification is incomplete.",
    "Project Boreal remains active. The old codeword was EAST. Correction: the current codeword is WEST, not EAST. Verification is incomplete.",
    "Project Cinder remains active. The old codeword was UP. Correction: the current codeword is DOWN, not UP. Verification is incomplete.",
    "Project Delta remains active. The old codeword was HOT. Correction: the current codeword is COLD, not HOT. Verification is incomplete.",
]
SELECTED_LAYERS = [0, 5, 11, 17, 23]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cache_targets(cache):
    return {
        "conv": [x.detach().clone() for x in cache.conv_states],
        "ssm": [x.detach().clone() for x in cache.ssm_states],
    }


def hidden_target(model, ids):
    with torch.no_grad():
        out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
        return out.hidden_states[-1][:, -1].detach().clone()


def state_regularizer(model, tok, text, target_cache, target_hidden):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    # Prefer actual recurrent cache compatibility. Fall back to final hidden-state compatibility only if
    # the installed cache implementation is not differentiable through its in-place storage path.
    try:
        from transformers.models.mamba.modeling_mamba import MambaCache
        cache = MambaCache(config=model.config, max_batch_size=1, device=model.device, dtype=model.dtype)
        pos = torch.arange(ids.shape[1], dtype=torch.long, device=model.device)
        out = model(ids, cache_params=cache, cache_position=pos, use_cache=True, return_dict=True)
        terms = []
        for i in SELECTED_LAYERS:
            for cur, tgt in ((out.cache_params.conv_states[i], target_cache["conv"][i]), (out.cache_params.ssm_states[i], target_cache["ssm"][i])):
                tgt = tgt.to(cur.device, cur.dtype)
                denom = tgt.detach().float().pow(2).mean().clamp_min(1e-4)
                terms.append((cur.float() - tgt.float()).pow(2).mean() / denom)
        loss = torch.stack(terms).mean()
        if loss.requires_grad:
            return loss, "recurrent_cache_selected_layers"
    except Exception:
        pass
    out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
    cur = out.hidden_states[-1][:, -1].float()
    tgt = target_hidden.to(cur.device).float()
    denom = tgt.pow(2).mean().clamp_min(1e-4)
    return (cur - tgt).pow(2).mean() / denom, "fallback_final_hidden_state"


def train(model, tok, cache_tgts, hidden_tgts):
    trainable = []
    for p in model.parameters(): p.requires_grad_(False)
    for name, p in model.named_parameters():
        if ".mixer.x_proj.weight" in name:
            p.requires_grad_(True); trainable.append((name, p))
    if not trainable: raise RuntimeError("no x_proj weights found")
    before = {n:p.detach().clone() for n,p in trainable}
    opt = torch.optim.AdamW([p for _,p in trainable], lr=v3.LR, weight_decay=0.0)
    order = list(range(len(v3.TRAIN))); epoch_rows=[]; regularizer_modes=set()
    for epoch in range(v3.EPOCHS):
        random.Random(v3.SEED+epoch).shuffle(order); model.train(); cap_sum=0.0; compat_sum=0.0; compat_n=0
        for step, idx in enumerate(order):
            prompt, gold = v3.TRAIN[idx]
            opt.zero_grad(set_to_none=True)
            cap_loss = v3.train_loss(model, tok, prompt, gold)
            total = cap_loss
            compat_val = None
            if step % COMPAT_EVERY == 0:
                ci = (step // COMPAT_EVERY + epoch) % len(COMPAT_TEXTS)
                reg, mode = state_regularizer(model, tok, COMPAT_TEXTS[ci], cache_tgts[ci], hidden_tgts[ci])
                regularizer_modes.add(mode); total = total + LAMBDA * reg; compat_val=float(reg.detach()); compat_sum += compat_val; compat_n += 1
            total.backward(); torch.nn.utils.clip_grad_norm_([p for _,p in trainable], 1.0); opt.step(); cap_sum += float(cap_loss.detach())
        model.eval(); epoch_rows.append({"epoch":epoch+1,"mean_capability_loss":cap_sum/len(order),"mean_compat_loss":compat_sum/max(compat_n,1),"heldout":v3.evaluate(model,tok)})
    sq=base=mx=0.0; n=0
    for name,p in trainable:
        d=(p.detach()-before[name]).float(); b=before[name].float(); sq+=float((d*d).sum()); base+=float((b*b).sum()); mx=max(mx,float(d.abs().max())); n+=d.numel()
    wd={"relative_l2":math.sqrt(sq)/max(math.sqrt(base),1e-12),"rms":math.sqrt(sq/max(n,1)),"max_abs":mx,"numel":n}
    return epoch_rows, sorted(regularizer_modes), wd


def run():
    transformers_version=None
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer,__version__ as transformers_version
        torch.manual_seed(v3.SEED); random.seed(v3.SEED)
        tok=AutoTokenizer.from_pretrained(v3.MODEL_ID); model=AutoModelForCausalLM.from_pretrained(v3.MODEL_ID,torch_dtype=torch.float32).cpu().eval()
        baseline=v3.evaluate(model,tok)
        cache_tgts=[]; hidden_tgts=[]
        for text in COMPAT_TEXTS:
            ids=tok(text,return_tensors="pt",add_special_tokens=False).input_ids
            with torch.no_grad(): cache_tgts.append(cache_targets(model(ids,use_cache=True,return_dict=True).cache_params))
            hidden_tgts.append(hidden_target(model,ids))
        hist=tok(v3.MIGRATION_HISTORY,return_tensors="pt",add_special_tokens=False).input_ids; T=int(hist.shape[1])
        with torch.no_grad(): old_full=v4.clone_cache(model(hist,use_cache=True,return_dict=True).cache_params)
        epoch_rows,modes,weight_delta=train(model,tok,cache_tgts,hidden_tgts); after=epoch_rows[-1]["heldout"]
        with torch.no_grad(): w2_native=v4.clone_cache(model(hist,use_cache=True,return_dict=True).cache_params)
        old_probe=v4.probe_state(model,tok,old_full,T); native_probe=v4.probe_state(model,tok,w2_native,T)
        migration={"direct_state_error_vs_w2_native":v3.cache_distance(old_full,w2_native),"direct_probe_vs_w2_native":v4.compare_to_native(old_probe,native_probe),"w2_native_probe":native_probe}
        report={"status":"VELA_COMPATIBILITY_AWARE_UPGRADE_V1","model":v3.MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,"compat_lambda":LAMBDA,"compat_every":COMPAT_EVERY,"regularizer_modes":modes,"selected_layers":SELECTED_LAYERS,"capability":{"baseline":baseline,"epoch_rows":epoch_rows,"after":after,"overall_gain":after['accuracy']-baseline['accuracy'],"correction_gain":after['correction_accuracy']-baseline['correction_accuracy'],"control_gain":after['control_accuracy']-baseline['control_accuracy']},"weight_delta":weight_delta,"migration":migration,"success_definition":"Useful compatibility-aware training should preserve a genuine held-out capability gain while reducing old-state mismatch or improving W2-native decision fidelity relative to the unconstrained learned-upgrade reference.","claim_boundary":"Narrow Mamba-1 130M synthetic probe. The primary regularizer targets selected recurrent cache layers when differentiable; if runtime forces fallback, that is reported explicitly and is weaker evidence."}
        write_report(report)
    except BaseException as exc:
        write_report({"status":"VELA_COMPAT_AWARE_ERROR","compat_lambda":LAMBDA,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-45:]}); raise

if __name__=="__main__": run()
