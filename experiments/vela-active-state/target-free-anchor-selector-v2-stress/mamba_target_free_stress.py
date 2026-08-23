from __future__ import annotations

import copy
import importlib.util
import json
import os
import random
import statistics
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
SEL_PATH = BASE / "target-free-anchor-selector-v1" / "mamba_target_free_selector.py"
spec = importlib.util.spec_from_file_location("vela_target_free_v1", SEL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SEL_PATH}")
sel = importlib.util.module_from_spec(spec); spec.loader.exec_module(sel)
dep, v3a, v2, v3, v4, fa = sel.dep, sel.v3a, sel.v2, sel.v3, sel.v4, sel.fa

# Migration/replay is inference-only. Force recurrent slices through no_grad so
# generated checkpoint states can be chained safely even while learned parameters
# retain requires_grad=True after training.
_orig_run_slice = dep.run_slice

def safe_run_slice(model, ids, start, end, cache):
    with torch.no_grad():
        return _orig_run_slice(model, ids, start, end, cache)

dep.run_slice = safe_run_slice

NEUTRAL = [
    "Telemetry packet 41 was archived. ",
    "A maintenance checksum completed without incident. ",
    "An unrelated sensor calibration note was filed. ",
    "The backup clock was synchronized. ",
    "A storage inventory entry was closed. ",
    "Routine diagnostics reported no actionable change. ",
]


def clone_fixture(src, new_id, prefix=0, middle=0, suffix=0, adversarial=None):
    out = {"id": new_id, "probes": copy.deepcopy(src["probes"])}
    segs = list(src["segments"])
    pre = [NEUTRAL[i % len(NEUTRAL)] for i in range(prefix)]
    mid = [NEUTRAL[(i + 2) % len(NEUTRAL)] for i in range(middle)]
    post = [NEUTRAL[(i + 4) % len(NEUTRAL)] for i in range(suffix)]
    cut = max(1, len(segs) // 2)
    segs = pre + segs[:cut] + mid + segs[cut:] + post
    if adversarial:
        # Irrelevant/historical mentions intentionally reuse task vocabulary but
        # explicitly mark it obsolete or unrelated.
        insert_at = max(1, len(segs) // 3)
        segs[insert_at:insert_at] = list(adversarial)
    out["segments"] = segs
    return out


BASES = {fx["id"]: fx for fx in sel.FIXTURES}
DEP_BASES = {fx["id"]: fx for fx in dep.FIXTURES}

FIXTURES = [
    copy.deepcopy(BASES["single_early"]),
    copy.deepcopy(BASES["single_middle"]),
    copy.deepcopy(BASES["single_late"]),
    copy.deepcopy(BASES["single_distractor_heavy"]),
    copy.deepcopy(DEP_BASES["independent_persistent"]),
    copy.deepcopy(DEP_BASES["superseded_chain"]),
    copy.deepcopy(DEP_BASES["late_overwrite_with_long_prefix"]),
    clone_fixture(DEP_BASES["independent_persistent"], "independent_interleaved_long", prefix=4, middle=5, suffix=3),
    clone_fixture(DEP_BASES["superseded_chain"], "superseded_long_prefix_suffix", prefix=7, middle=2, suffix=6),
    clone_fixture(BASES["single_early"], "early_with_long_suffix", suffix=12),
    clone_fixture(BASES["single_middle"], "middle_with_long_both_sides", prefix=8, suffix=8),
    clone_fixture(BASES["single_late"], "late_with_dense_middle_noise", middle=12),
    clone_fixture(BASES["single_distractor_heavy"], "distractor_heavy_extended", prefix=5, middle=8, suffix=5),
    clone_fixture(
        DEP_BASES["independent_persistent"],
        "independent_with_obsolete_vocab_noise",
        prefix=2,
        middle=3,
        suffix=2,
        adversarial=[
            "An obsolete training memo repeats ALPHA but is not current state. ",
            "A legacy checklist says complete; it is historical and not the current verification status. ",
            "The word allowed appears in a documentation example unrelated to external action. ",
        ],
    ),
    clone_fixture(
        DEP_BASES["superseded_chain"],
        "superseded_with_old_value_echoes",
        prefix=3,
        middle=4,
        suffix=3,
        adversarial=[
            "A retired report contains ALPHA and BETA as historical codewords only. ",
            "A checksum label contains BETA but does not update the current codeword. ",
        ],
    ),
]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def decision_signature(rows):
    return tuple((r["id"], r["chosen"]) for r in rows)


def choose_candidates(event_scores):
    vals = [r["detector_score"] for r in event_scores]
    med = statistics.median(vals)
    mad = statistics.median([abs(x - med) for x in vals])
    threshold = med + 2.0 * mad
    selected = [r for r in event_scores if r["decision_flips"] > 0 or r["detector_score"] >= threshold]
    if not selected:
        selected = [max(event_scores, key=lambda r: r["detector_score"])]
    return sorted(selected, key=lambda r: r["segment"]), {"median": med, "mad": mad, "threshold": threshold}


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        torch.manual_seed(v3.SEED); random.seed(v3.SEED)
        tok = AutoTokenizer.from_pretrained(v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(v3.MODEL_ID, torch_dtype=torch.float32).cpu().eval()

        prepared = []
        for fx in FIXTURES:
            ids = tok("".join(fx["segments"]), return_tensors="pt", add_special_tokens=False).input_ids
            T = int(ids.shape[1]); starts, ends = dep.boundaries(tok, fx["segments"])
            positions = sorted(set(starts + ends + [0, T]))
            caches = {p: dep.prefix_cache(model, ids, p) for p in positions}
            prepared.append({"fx": fx, "ids": ids, "T": T, "starts": starts, "ends": ends, "w1_caches": caches})

        baseline = v3.evaluate(model, tok)
        losses = v2.train_upgrade(model, tok)
        after = v3.evaluate(model, tok)

        rows = []
        valid_n = 0; functional_success_n = 0; exact_oracle_n = 0
        replay_fracs = []; oracle_fracs = []
        for item in prepared:
            fx, ids, T = item["fx"], item["ids"], item["T"]
            starts, ends, caches = item["starts"], item["ends"], item["w1_caches"]
            specs = fx["probes"]
            with torch.no_grad():
                native = v4.clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            native_rows = fa.score_probe_specs(model, tok, native, T, specs)
            native_expected = sum(int(r["correct"]) for r in native_rows) / len(native_rows)
            valid = native_expected == 1.0
            valid_n += int(valid)

            event_scores = []
            for i, (s, e) in enumerate(zip(starts, ends)):
                old_after = caches[e]
                new_after = dep.run_slice(model, ids, s, e, caches[s])
                sem = v3a.compare_local_semantics(model, tok, old_after, new_after, e, specs)
                event_scores.append({
                    "segment": i, "start": s, "end": e, "text": fx["segments"][i],
                    "state_rms": v3.cache_distance(old_after, new_after)["rms"], **sem,
                })
            selected, stats = choose_candidates(event_scores)

            replay_cache = {}
            def replay_at(seg):
                if seg not in replay_cache:
                    s = starts[seg]
                    migrated = dep.run_slice(model, ids, s, T, caches[s])
                    rr = fa.score_probe_specs(model, tok, migrated, T, specs)
                    replay_cache[seg] = {"state": migrated, "rows": rr}
                return replay_cache[seg]

            chosen_seg = selected[0]["segment"]
            ref = replay_at(chosen_seg)
            prune_trace = []
            for cand in selected[1:]:
                seg = cand["segment"]
                cur = replay_at(seg)
                same = decision_signature(cur["rows"]) == decision_signature(ref["rows"])
                prune_trace.append({"from_segment": chosen_seg, "candidate_later_segment": seg, "same_functional_signature": same})
                if same:
                    chosen_seg = seg; ref = cur
            comp = fa.compare_rows(ref["rows"], native_rows)
            success = valid and comp["decision_agreement"] == 1.0
            functional_success_n += int(success)

            oracle = []
            if valid:
                for i, s in enumerate(starts):
                    mig = dep.run_slice(model, ids, s, T, caches[s])
                    rr = fa.score_probe_specs(model, tok, mig, T, specs)
                    cc = fa.compare_rows(rr, native_rows)
                    if cc["decision_agreement"] == 1.0:
                        oracle.append((i, s))
            latest_oracle = max(oracle, key=lambda x: x[1]) if oracle else None
            exact_match = bool(latest_oracle is not None and chosen_seg == latest_oracle[0])
            exact_oracle_n += int(valid and exact_match)

            replay_fraction = (T - starts[chosen_seg]) / max(T, 1)
            oracle_fraction = None if latest_oracle is None else (T - latest_oracle[1]) / max(T, 1)
            if valid:
                replay_fracs.append(replay_fraction)
                if oracle_fraction is not None:
                    oracle_fracs.append(oracle_fraction)

            rows.append({
                "fixture": fx["id"], "events": len(starts), "history_tokens": T,
                "fixture_valid": valid, "w2_native_expected_accuracy": native_expected,
                "selected_causal_segments": [x["segment"] for x in selected],
                "detector_stats": stats, "prune_trace": prune_trace,
                "selected_anchor": {
                    "event_index": chosen_seg, "anchor_pos": starts[chosen_seg],
                    "replay_fraction": replay_fraction,
                },
                "selected_vs_w2_native": comp,
                "target_free_functional_success": success,
                "oracle_latest_safe_anchor": None if latest_oracle is None else {
                    "event_index": latest_oracle[0], "anchor_pos": latest_oracle[1],
                    "replay_fraction": oracle_fraction,
                },
                "target_free_matches_oracle_latest": exact_match,
                "semantic_drift_top5": sorted(event_scores, key=lambda r: r["detector_score"], reverse=True)[:5],
            })

        report = {
            "status": "VELA_TARGET_FREE_ANCHOR_SELECTOR_V2_STRESS",
            "model": v3.MODEL_ID,
            "torch_version": torch.__version__, "transformers_version": transformers_version,
            "capability": {
                "baseline": baseline["accuracy"], "after": after["accuracy"],
                "correction_before": baseline["correction_accuracy"], "correction_after": after["correction_accuracy"],
                "control_before": baseline["control_accuracy"], "control_after": after["control_accuracy"],
                "epoch_loss": losses,
            },
            "stress_suite": {
                "fixture_count": len(rows), "valid_fixture_count": valid_n,
                "valid_functional_success_count": functional_success_n,
                "valid_functional_success_rate": functional_success_n / max(valid_n, 1),
                "exact_oracle_latest_match_count": exact_oracle_n,
                "exact_oracle_latest_match_rate": exact_oracle_n / max(valid_n, 1),
                "mean_selected_replay_fraction_valid": sum(replay_fracs) / max(len(replay_fracs), 1),
                "mean_oracle_replay_fraction_valid": sum(oracle_fracs) / max(len(oracle_fracs), 1),
            },
            "selector": "Same target-free v1 policy: semantic-drift candidates, earliest conservative start, then prune forward only when later W2 replay preserves the current functional decision signature. W2-native is evaluation-only.",
            "fixtures": rows,
            "success_definition": "Primary safety criterion is zero functional false negatives on valid stress fixtures; oracle-frontier matching is an efficiency criterion, not required for functional correctness.",
            "claim_boundary": "Fifteen deterministic synthetic Mamba-130M histories with longer distractor spans, persistent independent updates, superseded updates, and obsolete-vocabulary noise. Semantic probes remain hand-specified and this is not generic causal discovery or identity proof.",
        }
        write_report(report)
    except BaseException as exc:
        write_report({
            "status": "VELA_TARGET_FREE_ANCHOR_SELECTOR_V2_STRESS_ERROR",
            "model": getattr(v3, "MODEL_ID", None), "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-50:],
        })
        raise


if __name__ == "__main__":
    run()
