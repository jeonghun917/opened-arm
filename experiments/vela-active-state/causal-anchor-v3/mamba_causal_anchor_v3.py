from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
V2_PATH = BASE / "causal-anchor-v2" / "mamba_causal_anchor_v2.py"
spec = importlib.util.spec_from_file_location("vela_anchor_v2", V2_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {V2_PATH}")
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
v3, v4, fa = v2.v3, v2.v4, v2.fa
EVENT_INTERVALS = [1, 2, 4, 8]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def compare_local_semantics(model, tok, old_state, new_state, pos, specs):
    old_rows = fa.score_probe_specs(model, tok, old_state, pos, specs)
    new_rows = fa.score_probe_specs(model, tok, new_state, pos, specs)
    old_by = {r["id"]: r for r in old_rows}; new_by = {r["id"]: r for r in new_rows}
    flips = 0; sq = 0.0; n = 0; margin_shift = 0.0
    per_probe = []
    for pid in old_by:
        a, b = old_by[pid], new_by[pid]
        flip = a["chosen"] != b["chosen"]
        flips += int(flip)
        candidates = list(a["scores"].keys())
        for cand in candidates:
            d = b["scores"][cand] - a["scores"][cand]
            sq += d*d; n += 1
        if len(candidates) == 2:
            ma = a["scores"][candidates[0]] - a["scores"][candidates[1]]
            mb = b["scores"][candidates[0]] - b["scores"][candidates[1]]
            margin_shift += abs(mb-ma)
        per_probe.append({"id": pid, "old_choice": a["chosen"], "new_choice": b["chosen"], "decision_flip": flip})
    score_rms = math.sqrt(sq/max(n,1))
    # Decision flips dominate; continuous score movement breaks ties.
    score = flips * 1000.0 + score_rms + 0.1 * margin_shift
    return {"decision_flips": flips, "score_rms": score_rms, "margin_shift_sum": margin_shift, "detector_score": score, "per_probe": per_probe}


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        torch.manual_seed(v3.SEED); random.seed(v3.SEED)
        tok = AutoTokenizer.from_pretrained(v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(v3.MODEL_ID, torch_dtype=torch.float32).cpu().eval()

        fixture_data = []
        for fx in v2.FIXTURES:
            text = "".join(fx["segments"])
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
            T = int(ids.shape[1]); starts, ends = v2.boundaries(tok, fx["segments"])
            positions = sorted(set(starts + ends + [0, T]))
            caches = {p: v2.prefix_cache(model, ids, p) for p in positions}
            fixture_data.append({"fx":fx,"ids":ids,"T":T,"starts":starts,"ends":ends,"w1_caches":caches})

        baseline = v3.evaluate(model, tok)
        w1_weights = v2.save_xproj(model)
        losses = v2.train_upgrade(model, tok)
        w2_weights = v2.save_xproj(model)
        after = v3.evaluate(model, tok)

        rows = []; hits = 0
        semantic_success = {str(k):0 for k in EVENT_INTERVALS}
        replay_fractions = {str(k):[] for k in EVENT_INTERVALS}
        for data in fixture_data:
            fx, ids, T = data["fx"], data["ids"], data["T"]
            starts, ends, caches = data["starts"], data["ends"], data["w1_caches"]
            specs = v2.probe_specs(fx)

            v2.load_xproj(model, w2_weights)
            with torch.no_grad(): native = v4.clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            native_rows = fa.score_probe_specs(model, tok, native, T, specs)

            event_scores = []
            for i,(s,e) in enumerate(zip(starts,ends)):
                pre = caches[s]
                v2.load_xproj(model, w1_weights); old_after,_ = v2.run_slice(model,ids,s,e,pre)
                v2.load_xproj(model, w2_weights); new_after,_ = v2.run_slice(model,ids,s,e,pre)
                sem = compare_local_semantics(model,tok,old_after,new_after,e,specs)
                state_rms = v3.cache_distance(old_after,new_after)["rms"]
                event_scores.append({"segment":i,"start":s,"end":e,"text":fx["segments"][i],"state_rms":state_rms,**sem})
            event_scores.sort(key=lambda r:r["detector_score"], reverse=True)
            detected = event_scores[0]["segment"]
            hit = detected == fx["critical_segment"]; hits += int(hit)

            v2.load_xproj(model, w2_weights)
            exact_pos = starts[detected]
            exact = v2.evaluate_migration(model,tok,ids,caches[exact_pos],exact_pos,native,T,specs)
            oracle_pos = starts[fx["critical_segment"]]
            oracle = v2.evaluate_migration(model,tok,ids,caches[oracle_pos],oracle_pos,native,T,specs)

            semantic = {}
            for k in EVENT_INTERVALS:
                saved_event_indices = list(range(0,len(starts),k))
                anchor_event = max([i for i in saved_event_indices if i <= detected], default=0)
                anchor_pos = starts[anchor_event]
                res = v2.evaluate_migration(model,tok,ids,caches[anchor_pos],anchor_pos,native,T,specs)
                res["checkpoint_every_n_events"] = k
                res["stored_checkpoint_count"] = len(saved_event_indices)
                res["anchor_event_index"] = anchor_event
                semantic[str(k)] = res
                replay_fractions[str(k)].append(res["replayed_tokens"]/max(T,1))
                if res["functional_vs_w2_native"]["decision_agreement"] == 1.0:
                    semantic_success[str(k)] += 1

            rows.append({
                "fixture":fx["id"],"tokens":T,"oracle_critical_segment":fx["critical_segment"],
                "detected_segment":detected,"detector_hit":hit,"ranked_event_scores":event_scores,
                "exact_detected_anchor":exact,"oracle_anchor":oracle,
                "semantic_checkpoint_sparsity":semantic,"w2_native_probe":native_rows,
            })

        n=len(rows)
        report={
            "status":"VELA_FUNCTIONAL_CAUSAL_EVENT_DETECTOR_V3",
            "model":v3.MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,
            "capability":{"baseline_accuracy":baseline["accuracy"],"after_accuracy":after["accuracy"],"baseline_correction":baseline["correction_accuracy"],"after_correction":after["correction_accuracy"],"baseline_control":baseline["control_accuracy"],"after_control":after["control_accuracy"],"epoch_loss":losses},
            "detector":{"definition":"For each event, compare semantic probe decisions after W1 vs W2 processes that event from the same actual W1 pre-event checkpoint; decision flips dominate the score.","uses_w2_native_for_ranking":False,"top1_critical_event_accuracy":hits/n},
            "semantic_checkpoint_sparsity":{"intervals_events":EVENT_INTERVALS,"full_functional_agreement_rate":{k:semantic_success[k]/n for k in semantic_success},"mean_replay_fraction":{k:sum(replay_fractions[k])/len(replay_fractions[k]) for k in replay_fractions}},
            "fixtures":rows,
            "success_definition":"Prefer an event detector that identifies upgrade-reinterpreted causal events without W2-native target access and an event-boundary checkpoint scheme that preserves W2-native functional decisions with bounded replay.",
            "claim_boundary":"Four synthetic histories and a narrow Mamba-130M correction upgrade. The semantic probe bank stands in for VELA canonical active-state slots; this is not generic causal discovery, identity proof, or architecture promotion."
        }
        write_report(report)
    except BaseException as exc:
        write_report({"status":"VELA_CAUSAL_ANCHOR_V3_ERROR","model":getattr(v3,"MODEL_ID",None),"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-45:]}); raise

if __name__=="__main__": run()
