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
V2_PATH = BASE / "causal-anchor-v2" / "mamba_causal_anchor_v2.py"
spec = importlib.util.spec_from_file_location("vela_anchor_v2", V2_PATH)
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
v4 = v2.v4; fa = v2.fa; v3 = v2.v3

METRICS = ["state_rms", "terminal_logit_rms", "mean_sym_kl", "max_sym_kl", "observed_logprob_abs_mean"]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_trace(model, ids, start, end, cache):
    logits = []
    c = v4.clone_cache(cache)
    with torch.no_grad():
        for j in range(start, end):
            out = model(ids[:, j:j+1], cache_params=c, cache_position=torch.tensor([j], dtype=torch.long), use_cache=True, return_dict=True)
            c = out.cache_params
            logits.append(out.logits[:, -1].detach().float().cpu())
    return v4.clone_cache(c), logits


def trace_metrics(old_logits, new_logits, ids, start, end):
    if len(old_logits) != len(new_logits): raise RuntimeError("trace length mismatch")
    syms = []; obs = []
    for j, (a, b) in enumerate(zip(old_logits, new_logits)):
        la = F.log_softmax(a, dim=-1); lb = F.log_softmax(b, dim=-1)
        pa = la.exp(); pb = lb.exp()
        skl = 0.5 * ((pa * (la-lb)).sum() + (pb * (lb-la)).sum())
        syms.append(float(skl))
        # Logit after current token predicts the next actual token within this event.
        absolute_index = start + j + 1
        if absolute_index < end:
            tid = int(ids[0, absolute_index])
            obs.append(abs(float(la[0, tid] - lb[0, tid])))
    term = 0.0
    if old_logits:
        d = old_logits[-1] - new_logits[-1]
        term = float(torch.sqrt(torch.mean(d*d)))
    return {
        "terminal_logit_rms": term,
        "mean_sym_kl": sum(syms)/max(len(syms),1),
        "max_sym_kl": max(syms) if syms else 0.0,
        "observed_logprob_abs_mean": sum(obs)/max(len(obs),1),
    }


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        torch.manual_seed(v3.SEED); random.seed(v3.SEED)
        tok = AutoTokenizer.from_pretrained(v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(v3.MODEL_ID, torch_dtype=torch.float32).cpu().eval()

        data = []
        for fx in v2.FIXTURES:
            text = "".join(fx["segments"])
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
            T = int(ids.shape[1]); starts, ends = v2.boundaries(tok, fx["segments"])
            caches = {p: v2.prefix_cache(model, ids, p) for p in starts}
            data.append({"fx":fx,"ids":ids,"T":T,"starts":starts,"ends":ends,"w1_caches":caches})

        baseline = v3.evaluate(model, tok)
        w1_weights = v2.save_xproj(model)
        losses = v2.train_upgrade(model, tok)
        w2_weights = v2.save_xproj(model)
        after = v3.evaluate(model, tok)

        metric_hits = {m:0 for m in METRICS}
        metric_functional = {m:0 for m in METRICS}
        fixture_rows = []

        for item in data:
            fx, ids, T = item["fx"], item["ids"], item["T"]
            starts, ends, caches = item["starts"], item["ends"], item["w1_caches"]
            v2.load_xproj(model, w2_weights)
            with torch.no_grad(): native = v4.clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            native_rows = fa.score_probe_specs(model, tok, native, T, v2.probe_specs(fx))

            events = []
            for i,(s,e) in enumerate(zip(starts,ends)):
                base = caches[s]
                v2.load_xproj(model, w1_weights)
                old_after, old_logits = run_trace(model, ids, s, e, base)
                v2.load_xproj(model, w2_weights)
                new_after, new_logits = run_trace(model, ids, s, e, base)
                metrics = trace_metrics(old_logits, new_logits, ids, s, e)
                metrics["state_rms"] = v3.cache_distance(old_after, new_after)["rms"]
                events.append({"segment":i,"start":s,"end":e,"tokens":e-s,"text":fx["segments"][i],**metrics})

            rankings = {}
            migrations = {}
            for metric in METRICS:
                ranked = sorted(events, key=lambda r:r[metric], reverse=True)
                pred = ranked[0]["segment"]
                hit = pred == fx["critical_segment"]
                metric_hits[metric] += int(hit)
                anchor = starts[pred]
                v2.load_xproj(model, w2_weights)
                migrated = v2.evaluate_migration(model, tok, ids, caches[anchor], anchor, native, T, v2.probe_specs(fx))
                full = migrated["functional_vs_w2_native"]["decision_agreement"] == 1.0
                metric_functional[metric] += int(full)
                rankings[metric] = {"predicted_segment":pred,"detector_hit":hit,"top3":[{k:r[k] for k in ["segment","start","end","tokens","text",metric]} for r in ranked[:3]]}
                migrations[metric] = migrated

            fixture_rows.append({
                "fixture":fx["id"],"history_tokens":T,"oracle_critical_segment":fx["critical_segment"],
                "rankings":rankings,"migration_from_predicted_anchor":migrations,
                "w2_native_probe":native_rows,
            })

        n = len(fixture_rows)
        report = {
            "status":"VELA_EVENT_DETECTOR_V3_SEQUENCE_DIVERGENCE",
            "model":v3.MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,
            "capability":{"baseline":baseline["accuracy"],"after":after["accuracy"],"correction_before":baseline["correction_accuracy"],"correction_after":after["correction_accuracy"],"control_before":baseline["control_accuracy"],"control_after":after["control_accuracy"],"epoch_loss":losses},
            "detectors":{
                m:{"top1_critical_event_accuracy":metric_hits[m]/n,"functional_full_agreement_rate_from_selected_anchor":metric_functional[m]/n}
                for m in METRICS
            },
            "fixtures":fixture_rows,
            "selection_constraint":"All detector metrics compare W1 and W2 processing of the same event from the same stored W1 pre-event state. W2-native full-history state is used only after selection for evaluation, never for ranking.",
            "success_definition":"A useful automatic detector should rank the upgrade-relevant event above unrelated events and its selected real W1 checkpoint should replay into W2-native-equivalent functional decisions.",
            "claim_boundary":"Four synthetic histories and one narrow correction upgrade. This compares detector signals, not a general causal discovery theorem or architecture promotion.",
        }
        write_report(report)
    except BaseException as exc:
        write_report({"status":"VELA_EVENT_DETECTOR_V3_ERROR","model":getattr(v3,"MODEL_ID",None),"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-40:]}); raise

if __name__ == "__main__": run()
