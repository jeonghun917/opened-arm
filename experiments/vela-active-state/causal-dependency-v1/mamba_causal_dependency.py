from __future__ import annotations

import importlib.util
import json
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
V3_PATH = BASE / "causal-anchor-v3" / "mamba_causal_anchor_v3.py"
spec = importlib.util.spec_from_file_location("vela_anchor_v3", V3_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {V3_PATH}")
v3a = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3a)
v2, v3, v4, fa = v3a.v2, v3a.v3, v3a.v4, v3a.fa

FIXTURES = [
    {
        "id": "superseded_chain",
        "segments": [
            "Project Orion remains active. ",
            "The old codeword was ALPHA. ",
            "Correction: the current codeword is BETA, not ALPHA. ",
            "Unrelated telemetry packet 17 was archived. ",
            "New correction: the current codeword is GAMMA, not BETA. ",
            "Verification is incomplete. ",
            "A historical memo mentions ALPHA but is obsolete. ",
            "External action remains blocked."
        ],
        "candidate_causal_segments": [2, 4],
        "probes": [
            ("codeword", "\nCurrent codeword:", [" GAMMA", " BETA", " ALPHA"], " GAMMA"),
            ("verification", "\nVerification status:", [" incomplete", " complete"], " incomplete"),
            ("project", "\nProject Orion status:", [" active", " paused"], " active"),
            ("action", "\nExternal action is:", [" blocked", " allowed"], " blocked"),
        ],
        "question": "Can the later correction make the earlier correction replay-unnecessary?"
    },
    {
        "id": "independent_persistent",
        "segments": [
            "Project Helios remains active. ",
            "The old codeword was ALPHA. ",
            "Correction: the current codeword is BETA, not ALPHA. ",
            "Unrelated telemetry packet 21 was archived. ",
            "Verification was complete. ",
            "Correction: the current verification status is incomplete, not complete. ",
            "A historical memo mentions ALPHA but is obsolete. ",
            "External action remains blocked."
        ],
        "candidate_causal_segments": [2, 5],
        "probes": [
            ("codeword", "\nCurrent codeword:", [" BETA", " ALPHA"], " BETA"),
            ("verification", "\nVerification status:", [" incomplete", " complete"], " incomplete"),
            ("project", "\nProject Helios status:", [" active", " paused"], " active"),
            ("action", "\nExternal action is:", [" blocked", " allowed"], " blocked"),
        ],
        "question": "Do two persistent corrected facts force replay from the earlier one?"
    },
    {
        "id": "late_overwrite_with_long_prefix",
        "segments": [
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
        "candidate_causal_segments": [2, 8],
        "probes": [
            ("codeword", "\nCurrent codeword:", [" WEST", " HIGH", " LOW"], " WEST"),
            ("verification", "\nVerification status:", [" incomplete", " complete"], " incomplete"),
            ("project", "\nProject Juno status:", [" active", " paused"], " active"),
            ("action", "\nExternal action is:", [" blocked", " allowed"], " blocked"),
        ],
        "question": "Can a late overwrite bound replay despite a long earlier history?"
    },
]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def boundaries(tok, segments):
    starts=[]; ends=[]
    for i in range(len(segments)):
        starts.append(int(tok("".join(segments[:i]), return_tensors="pt", add_special_tokens=False).input_ids.shape[1]))
        ends.append(int(tok("".join(segments[:i+1]), return_tensors="pt", add_special_tokens=False).input_ids.shape[1]))
    return starts, ends


def prefix_cache(model, ids, pos):
    if pos == 0: return None
    with torch.no_grad(): return v4.clone_cache(model(ids[:, :pos], use_cache=True, return_dict=True).cache_params)


def run_slice(model, ids, start, end, cache):
    if end <= start: return v4.clone_cache(cache)
    if start == 0 and cache is None:
        with torch.no_grad(): return v4.clone_cache(model(ids[:, :end], use_cache=True, return_dict=True).cache_params)
    out, _ = v3.run_tokens_with_cache(model, ids[:, start:end], v4.clone_cache(cache), start)
    return v4.clone_cache(out)


def evaluate_anchor(model, tok, ids, old_cache, start, native, specs):
    T=int(ids.shape[1])
    migrated=run_slice(model, ids, start, T, old_cache)
    native_rows=fa.score_probe_specs(model,tok,native,T,specs)
    rows=fa.score_probe_specs(model,tok,migrated,T,specs)
    comp=fa.compare_rows(rows,native_rows)
    return {
        "anchor_pos": start,
        "replayed_tokens": T-start,
        "replay_fraction": (T-start)/max(T,1),
        "state_error_vs_w2_native": v3.cache_distance(migrated,native),
        "functional_vs_w2_native": comp,
        "probe_rows": rows,
    }


def run():
    transformers_version=None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        torch.manual_seed(v3.SEED); random.seed(v3.SEED)
        tok=AutoTokenizer.from_pretrained(v3.MODEL_ID)
        model=AutoModelForCausalLM.from_pretrained(v3.MODEL_ID,torch_dtype=torch.float32).cpu().eval()

        prepared=[]
        for fx in FIXTURES:
            text="".join(fx["segments"]); ids=tok(text,return_tensors="pt",add_special_tokens=False).input_ids
            T=int(ids.shape[1]); starts,ends=boundaries(tok,fx["segments"])
            caches={p:prefix_cache(model,ids,p) for p in sorted(set(starts+[0,T]))}
            prepared.append({"fx":fx,"ids":ids,"T":T,"starts":starts,"ends":ends,"w1_caches":caches})

        baseline=v3.evaluate(model,tok)
        losses=v2.train_upgrade(model,tok)
        after=v3.evaluate(model,tok)

        fixtures=[]
        latest_safe_success=0
        for item in prepared:
            fx,ids,T=item["fx"],item["ids"],item["T"]
            starts,ends,caches=item["starts"],item["ends"],item["w1_caches"]
            with torch.no_grad(): native=v4.clone_cache(model(ids,use_cache=True,return_dict=True).cache_params)
            native_rows=fa.score_probe_specs(model,tok,native,T,fx["probes"])

            # Characterize every event-boundary checkpoint. The latest boundary that still
            # reaches full W2-native functional agreement is the empirical minimal-replay anchor.
            anchor_rows=[]
            for event_idx,start in enumerate(starts):
                res=evaluate_anchor(model,tok,ids,caches[start],start,native,fx["probes"])
                res["event_index"]=event_idx
                res["event_text"]=fx["segments"][event_idx]
                res["full_functional_agreement"]=res["functional_vs_w2_native"]["decision_agreement"]==1.0
                anchor_rows.append(res)
            safe=[r for r in anchor_rows if r["full_functional_agreement"]]
            latest_safe=max(safe,key=lambda r:r["anchor_pos"]) if safe else None
            latest_safe_success += int(latest_safe is not None)

            # Semantic-drift ranking remains target-free and is compared with the empirical safe frontier.
            event_scores=[]
            for i,(s,e) in enumerate(zip(starts,ends)):
                pre=caches[s]
                # old_after must be generated by W1, but model is now W2. Reconstruct W1 by using
                # the pre-upgrade cached post-event target is unavailable here, so use stored W1 end cache.
                old_after=caches.get(e)
                if old_after is None:
                    continue
                new_after=run_slice(model,ids,s,e,pre)
                sem=v3a.compare_local_semantics(model,tok,old_after,new_after,e,fx["probes"])
                event_scores.append({"segment":i,"start":s,"end":e,"text":fx["segments"][i],"state_rms":v3.cache_distance(old_after,new_after)["rms"],**sem})
            event_scores.sort(key=lambda r:r["detector_score"],reverse=True)

            candidate_rows=[]
            for idx in fx["candidate_causal_segments"]:
                row=next(r for r in anchor_rows if r["event_index"]==idx)
                candidate_rows.append({"segment":idx,"anchor_pos":row["anchor_pos"],"replay_fraction":row["replay_fraction"],"full_functional_agreement":row["full_functional_agreement"],"decision_agreement":row["functional_vs_w2_native"]["decision_agreement"]})

            fixtures.append({
                "fixture":fx["id"],"question":fx["question"],"history_tokens":T,
                "candidate_causal_segments":fx["candidate_causal_segments"],
                "w2_native_probe":native_rows,
                "latest_safe_anchor":None if latest_safe is None else {"event_index":latest_safe["event_index"],"anchor_pos":latest_safe["anchor_pos"],"replayed_tokens":latest_safe["replayed_tokens"],"replay_fraction":latest_safe["replay_fraction"],"state_rms":latest_safe["state_error_vs_w2_native"]["rms"],"decision_agreement":latest_safe["functional_vs_w2_native"]["decision_agreement"]},
                "candidate_anchor_summary":candidate_rows,
                "semantic_drift_top4":event_scores[:4],
                "all_anchor_rows":anchor_rows,
            })

        write_report({
            "status":"VELA_CAUSAL_DEPENDENCY_LATEST_SAFE_ANCHOR_V1",
            "model":v3.MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,
            "capability":{"baseline":baseline["accuracy"],"after":after["accuracy"],"correction_before":baseline["correction_accuracy"],"correction_after":after["correction_accuracy"],"control_before":baseline["control_accuracy"],"control_after":after["control_accuracy"],"epoch_loss":losses},
            "latest_safe_anchor_definition":"Latest stored W1 event-boundary checkpoint from which W2 replay reaches full W2-native probe-decision agreement. Used here as an oracle characterization of the minimum necessary replay, not as a deployable selector.",
            "latest_safe_found_rate":latest_safe_success/len(fixtures),
            "fixtures":fixtures,
            "success_definition":"Determine whether superseded semantic changes can be skipped while independent persistent changes force an earlier anchor, before designing a target-free selector for the latest safe checkpoint.",
            "claim_boundary":"Three synthetic Mamba-130M histories. W2-native is used only to characterize the safe replay frontier; this experiment does not yet solve target-free safe-anchor selection or prove identity continuity."
        })
    except BaseException as exc:
        write_report({"status":"VELA_CAUSAL_DEPENDENCY_V1_ERROR","model":getattr(v3,"MODEL_ID",None),"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-45:]}); raise

if __name__=="__main__": run()
