from __future__ import annotations

import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chain = load_module(
    "vela_rwkv7_chain_v1",
    BASE / "rwkv7-0p4b-chain-v1" / "rwkv7_0p4b_chained_upgrade.py",
)

# Diagnostic/policy variant after v1 showed 4/4 functional agreement but only
# 1/4 hop-2 anchors actually came from W2.  The only policy change here is a
# target-free provenance tiebreak: if a carried W2-origin anchor can replay to
# the same functional signature as the v1-selected result, prefer the latest
# such W2-origin anchor.  W3-native full-history state remains evaluation-only.
_ORIGINS_BY_LINEAGE_ID: dict[int, dict[int, str]] = {}
_orig_rebuild = chain.rebuild_lineage
_orig_write = chain.write_report


def rebuild_lineage(model, ns, ids, starts, ends, prior, prior_origins, anchor_seg, generation):
    out, origins, current = _orig_rebuild(
        model, ns, ids, starts, ends, prior, prior_origins, anchor_seg, generation
    )
    _ORIGINS_BY_LINEAGE_ID[id(out)] = dict(origins)
    return out, origins, current


def target_free_select(model, ns, tok, ids, starts, ends, lineage, specs, segments):
    event_scores = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        old_after = lineage[e]
        new_after = chain.state_after_segment(model, ns, ids, s, e, lineage[s])
        sem = chain.ca.semantic_change(model, ns, tok, old_after, new_after, specs)
        event_scores.append(
            {
                "segment": i,
                "start": s,
                "end": e,
                "text": segments[i],
                "state_rms": chain.rw.state_distance(old_after, new_after)["rms"],
                **sem,
            }
        )

    selected, stats = chain.choose_candidates(event_scores)
    replay_cache = {}

    def replay_at(seg):
        if seg not in replay_cache:
            pos = starts[seg]
            migrated = chain.state_to_end(model, ns, ids, pos, lineage[pos])
            rows = chain.ca.score_specs(model, ns, tok, migrated, specs)
            replay_cache[seg] = {"state": migrated, "rows": rows}
        return replay_cache[seg]

    # Preserve the v1 target-free candidate/pruning result first.
    chosen = selected[0]["segment"]
    ref = replay_at(chosen)
    prune_trace = []
    for cand in selected[1:]:
        seg = cand["segment"]
        cur = replay_at(seg)
        same = chain.decision_signature(cur["rows"]) == chain.decision_signature(ref["rows"])
        prune_trace.append(
            {
                "from_segment": chosen,
                "candidate_later_segment": seg,
                "same_functional_signature": same,
            }
        )
        if same:
            chosen = seg
            ref = cur

    baseline_chosen = chosen
    baseline_signature = chain.decision_signature(ref["rows"])
    provenance_trace = []

    # Only mixed W1/W2 lineages created by hop 1 are registered here.  Hop 1
    # itself and the direct W1->W3 evaluation control therefore keep v1 logic.
    origins = _ORIGINS_BY_LINEAGE_ID.get(id(lineage))
    if origins is not None:
        eligible_w2 = []
        for seg, pos in enumerate(starts):
            origin = origins.get(pos)
            if origin != "W2":
                continue
            cur = replay_at(seg)
            same = chain.decision_signature(cur["rows"]) == baseline_signature
            provenance_trace.append(
                {
                    "segment": seg,
                    "anchor_pos": pos,
                    "origin": origin,
                    "same_functional_signature_as_v1_result": same,
                }
            )
            if same:
                eligible_w2.append(seg)

        if eligible_w2:
            chosen = max(eligible_w2)
            ref = replay_at(chosen)

    selected_segments = sorted({x["segment"] for x in selected} | {chosen})
    return {
        "chosen_seg": chosen,
        "anchor_pos": starts[chosen],
        "state": ref["state"],
        "rows": ref["rows"],
        "selected_segments": selected_segments,
        "detector_stats": stats,
        "prune_trace": prune_trace,
        "semantic_drift_top5": sorted(event_scores, key=lambda x: x["detector_score"], reverse=True)[:5],
        "provenance_tiebreak": {
            "v1_chosen_seg": baseline_chosen,
            "final_chosen_seg": chosen,
            "used": chosen != baseline_chosen,
            "w2_candidates": provenance_trace,
        },
    }


def write_report(report):
    report["status"] = "VELA_RWKV7_0P4B_CHAINED_UPGRADE_PROVENANCE_V2"
    report["policy_variant"] = {
        "name": "latest-carried-generation functional-equivalence tiebreak",
        "oracle_usage": "none for selection; W3-native remains evaluation-only",
        "purpose": "distinguish a continuity failure from v1 provenance-blind anchor preference",
    }
    _orig_write(report)


chain.rebuild_lineage = rebuild_lineage
chain.target_free_select = target_free_select
chain.write_report = write_report

if __name__ == "__main__":
    chain.run()
