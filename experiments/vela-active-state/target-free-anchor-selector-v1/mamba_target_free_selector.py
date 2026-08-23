from __future__ import annotations

import importlib.util
import json
import os
import random
import statistics
import traceback
from pathlib import Path

import torch

BASE=Path(__file__).resolve().parents[1]
DEP_PATH=BASE/"causal-dependency-v1"/"mamba_causal_dependency.py"
spec=importlib.util.spec_from_file_location("vela_dep_v1",DEP_PATH)
if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {DEP_PATH}")
dep=importlib.util.module_from_spec(spec); spec.loader.exec_module(dep)
v3a, v2, v3, v4, fa = dep.v3a, dep.v2, dep.v3, dep.v4, dep.fa


def single_fixture(src):
    fx={"id":"single_"+src["id"],"segments":src["segments"],"probes":v2.probe_specs(src)}
    return fx

FIXTURES=[dep.FIXTURES[1],dep.FIXTURES[2]]+[single_fixture(x) for x in v2.FIXTURES]


def write_report(report):
    pth=os.environ.get("VELA_RESULT_PATH")
    if pth:
        p=Path(pth); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


def decision_signature(rows):
    return tuple((r["id"],r["chosen"]) for r in rows)


def choose_candidates(event_scores):
    vals=[r["detector_score"] for r in event_scores]
    med=statistics.median(vals); mad=statistics.median([abs(x-med) for x in vals])
    threshold=med+2.0*mad
    selected=[r for r in event_scores if r["decision_flips"]>0 or r["detector_score"]>=threshold]
    if not selected: selected=[max(event_scores,key=lambda r:r["detector_score"])]
    selected=sorted(selected,key=lambda r:r["segment"])
    return selected,{"median":med,"mad":mad,"threshold":threshold}


def run():
    transformers_version=None
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer,__version__ as transformers_version
        torch.manual_seed(v3.SEED); random.seed(v3.SEED)
        tok=AutoTokenizer.from_pretrained(v3.MODEL_ID)
        model=AutoModelForCausalLM.from_pretrained(v3.MODEL_ID,torch_dtype=torch.float32).cpu().eval()

        data=[]
        for fx in FIXTURES:
            ids=tok("".join(fx["segments"]),return_tensors="pt",add_special_tokens=False).input_ids
            T=int(ids.shape[1]); starts,ends=dep.boundaries(tok,fx["segments"])
            positions=sorted(set(starts+ends+[0,T]))
            caches={p:dep.prefix_cache(model,ids,p) for p in positions}
            data.append({"fx":fx,"ids":ids,"T":T,"starts":starts,"ends":ends,"w1_caches":caches})

        baseline=v3.evaluate(model,tok); losses=v2.train_upgrade(model,tok); after=v3.evaluate(model,tok)
        rows=[]; valid_n=0; success_n=0
        for item in data:
            fx,ids,T=item["fx"],item["ids"],item["T"]; starts,ends,caches=item["starts"],item["ends"],item["w1_caches"]
            specs=fx["probes"]
            with torch.no_grad(): native=v4.clone_cache(model(ids,use_cache=True,return_dict=True).cache_params)
            native_rows=fa.score_probe_specs(model,tok,native,T,specs)
            native_expected=sum(int(r["correct"]) for r in native_rows)/len(native_rows); valid=native_expected==1.0
            valid_n+=int(valid)

            event_scores=[]
            for i,(s,e) in enumerate(zip(starts,ends)):
                old_after=caches[e]
                new_after=dep.run_slice(model,ids,s,e,caches[s])
                sem=v3a.compare_local_semantics(model,tok,old_after,new_after,e,specs)
                event_scores.append({"segment":i,"start":s,"end":e,"text":fx["segments"][i],"state_rms":v3.cache_distance(old_after,new_after)["rms"],**sem})
            event_scores_sorted=sorted(event_scores,key=lambda r:r["detector_score"],reverse=True)
            selected,stats=choose_candidates(event_scores)

            # Target-free pruning: start conservatively at the earliest detected causal event.
            # A later candidate replaces it only when both W2 replays yield the same current
            # functional decision signature. No W2-native target is used in this selection.
            replay_cache={}
            def replay_at(seg):
                if seg not in replay_cache:
                    s=starts[seg]; migrated=dep.run_slice(model,ids,s,T,caches[s]); rr=fa.score_probe_specs(model,tok,migrated,T,specs)
                    replay_cache[seg]={"state":migrated,"rows":rr}
                return replay_cache[seg]
            chosen_seg=selected[0]["segment"]; ref=replay_at(chosen_seg)
            prune_trace=[]
            for cand in selected[1:]:
                seg=cand["segment"]; cr=replay_at(seg); same=decision_signature(cr["rows"])==decision_signature(ref["rows"])
                prune_trace.append({"from_segment":chosen_seg,"candidate_later_segment":seg,"same_functional_signature":same})
                if same:
                    chosen_seg=seg; ref=cr
            chosen_rows=ref["rows"]; comp=fa.compare_rows(chosen_rows,native_rows)
            success=valid and comp["decision_agreement"]==1.0; success_n+=int(success)

            # Oracle latest-safe frontier is evaluation only, never fed into selector.
            oracle=[]
            for i,s in enumerate(starts):
                mig=dep.run_slice(model,ids,s,T,caches[s]); rr=fa.score_probe_specs(model,tok,mig,T,specs); cc=fa.compare_rows(rr,native_rows)
                if cc["decision_agreement"]==1.0: oracle.append((i,s))
            latest_oracle=max(oracle,key=lambda x:x[1]) if oracle and valid else None

            rows.append({
                "fixture":fx["id"],"history_tokens":T,"fixture_valid":valid,"w2_native_expected_accuracy":native_expected,
                "detector_stats":stats,"selected_causal_segments":[x["segment"] for x in selected],
                "selected_causal_event_texts":[x["text"] for x in selected],"prune_trace":prune_trace,
                "target_free_selected_anchor":{"event_index":chosen_seg,"anchor_pos":starts[chosen_seg],"replayed_tokens":T-starts[chosen_seg],"replay_fraction":(T-starts[chosen_seg])/max(T,1)},
                "selected_vs_w2_native":comp,
                "oracle_latest_safe_anchor":None if latest_oracle is None else {"event_index":latest_oracle[0],"anchor_pos":latest_oracle[1],"replay_fraction":(T-latest_oracle[1])/max(T,1)},
                "target_free_matches_oracle_latest":bool(latest_oracle is not None and chosen_seg==latest_oracle[0]),
                "target_free_functional_success":success,
                "semantic_drift_top4":event_scores_sorted[:4],
            })

        write_report({
            "status":"VELA_TARGET_FREE_ANCHOR_SELECTOR_V1",
            "model":v3.MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,
            "capability":{"baseline":baseline["accuracy"],"after":after["accuracy"],"correction_before":baseline["correction_accuracy"],"correction_after":after["correction_accuracy"],"control_before":baseline["control_accuracy"],"control_after":after["control_accuracy"],"epoch_loss":losses},
            "selector":"Robust semantic-drift outliers identify candidate events; begin at earliest candidate, then prune forward when a later-candidate W2 replay has the same current functional decision signature. W2-native is evaluation-only.",
            "valid_fixture_count":valid_n,"valid_functional_success_count":success_n,"valid_functional_success_rate":success_n/max(valid_n,1),
            "fixtures":rows,
            "success_definition":"A deployable-style selector must choose without W2-native full-history access, then be judged afterward against W2-native functional decisions.",
            "claim_boundary":"Six small synthetic Mamba-130M histories. Fixed semantic probes and a fixed robust-outlier rule; this is an engineering proof-of-concept, not generic causal discovery or identity proof."
        })
    except BaseException as exc:
        write_report({"status":"VELA_TARGET_FREE_ANCHOR_SELECTOR_V1_ERROR","model":getattr(v3,"MODEL_ID",None),"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-45:]}); raise

if __name__=="__main__": run()
