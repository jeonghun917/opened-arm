from __future__ import annotations

import gc
import importlib.util
import json
import os
import random
import statistics
import tempfile
import traceback
import urllib.request
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rw = load_module("vela_rwkv7_learned", BASE / "rwkv7-learned-upgrade-v1" / "rwkv7_learned_upgrade.py")
scale = load_module("vela_rwkv7_scale_0p4b", BASE / "rwkv7-scale-0p4b-v1" / "rwkv7_scale_0p4b.py")
resid = load_module("vela_rwkv7_residual", BASE / "rwkv7-0p4b-residual-diagnostic-v1" / "rwkv7_0p4b_residual_diagnostic.py")
sel = load_module("vela_rwkv7_selector", BASE / "rwkv7-target-free-anchor-selector-v1" / "rwkv7_target_free_selector.py")
ca = sel.ca

# Predeclared before seeing this chained run's selector outcomes. These four all
# exercise the same-slot overwrite chain, but they vary suffix distance/noise and
# include the old-value-echo adversarial case. W2 is intentionally the known
# sup-heavy intermediate generation; W3 is the known low-LR status-retention patch.
FIXTURE_IDS = [
    "superseded_base",
    "superseded_suffix4",
    "superseded_suffix8",
    "superseded_old_value_echoes",
]
ROB_BY_ID = {fx["id"]: fx for fx in resid.rob.FIXTURES}
FIXTURES = [ROB_BY_ID[x] for x in FIXTURE_IDS]

W2_ROWS = list(resid.SUPHEAVY)
W2_LR = 1e-5
W3_ROWS = list(resid.STATUS8)
W3_LR = 5e-6


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def kvr_names(args):
    out = []
    for i in range(args.n_layer):
        for suffix in ("key.weight", "value.weight", "receptance.weight"):
            out.append(f"blocks.{i}.att.{suffix}")
    return out


def save_kvr(model, args):
    return {name: model.z[name].detach().clone() for name in kvr_names(args)}


def load_kvr(model, snap):
    with torch.no_grad():
        for name, value in snap.items():
            model.z[name].copy_(value)


def clone_state(x):
    return rw.clone_state(x)


def clone_state_map(states):
    return {k: clone_state(v) for k, v in states.items()}


def state_after_segment(model, ns, ids, start, end, start_state):
    st = clone_state(start_state)
    if end > start:
        with torch.no_grad():
            _, st = rw.run_tokens(model, ns, ids[start:end], st)
    return clone_state(st)


def state_to_end(model, ns, ids, start, start_state):
    st = clone_state(start_state)
    if start < len(ids):
        with torch.no_grad():
            _, st = rw.run_tokens(model, ns, ids[start:], st)
    return clone_state(st)


def decision_signature(rows):
    return tuple((r["id"], r["chosen"]) for r in rows)


def choose_candidates(event_scores):
    vals = [x["detector_score"] for x in event_scores]
    med = statistics.median(vals)
    mad = statistics.median([abs(x - med) for x in vals])
    threshold = med + 2.0 * mad
    selected = [x for x in event_scores if x["decision_flips"] > 0 or x["detector_score"] >= threshold]
    if not selected:
        selected = [max(event_scores, key=lambda x: x["detector_score"])]
    return sorted(selected, key=lambda x: x["segment"]), {"median": med, "mad": mad, "threshold": threshold}


def target_free_select(model, ns, tok, ids, starts, ends, lineage, specs, segments):
    # The lineage contains the actually carried pre-upgrade states. For each event,
    # compare the stored post-event state with the NEW generation reprocessing the
    # same event from the same carried pre-event state. No native full-history target
    # is used in candidate generation or pruning.
    event_scores = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        old_after = lineage[e]
        new_after = state_after_segment(model, ns, ids, s, e, lineage[s])
        sem = ca.semantic_change(model, ns, tok, old_after, new_after, specs)
        event_scores.append({
            "segment": i,
            "start": s,
            "end": e,
            "text": segments[i],
            "state_rms": rw.state_distance(old_after, new_after)["rms"],
            **sem,
        })

    selected, stats = choose_candidates(event_scores)
    replay_cache = {}

    def replay_at(seg):
        if seg not in replay_cache:
            pos = starts[seg]
            migrated = state_to_end(model, ns, ids, pos, lineage[pos])
            rows = ca.score_specs(model, ns, tok, migrated, specs)
            replay_cache[seg] = {"state": migrated, "rows": rows}
        return replay_cache[seg]

    chosen = selected[0]["segment"]
    ref = replay_at(chosen)
    prune_trace = []
    for cand in selected[1:]:
        seg = cand["segment"]
        cur = replay_at(seg)
        same = decision_signature(cur["rows"]) == decision_signature(ref["rows"])
        prune_trace.append({
            "from_segment": chosen,
            "candidate_later_segment": seg,
            "same_functional_signature": same,
        })
        if same:
            chosen = seg
            ref = cur

    return {
        "chosen_seg": chosen,
        "anchor_pos": starts[chosen],
        "state": ref["state"],
        "rows": ref["rows"],
        "selected_segments": [x["segment"] for x in selected],
        "detector_stats": stats,
        "prune_trace": prune_trace,
        "semantic_drift_top5": sorted(event_scores, key=lambda x: x["detector_score"], reverse=True)[:5],
    }


def rebuild_lineage(model, ns, ids, starts, ends, prior, prior_origins, anchor_seg, generation):
    anchor_pos = starts[anchor_seg]
    out = {}
    origins = {}
    for pos, st in prior.items():
        if pos <= anchor_pos:
            out[pos] = clone_state(st)
            origins[pos] = prior_origins[pos]

    current = clone_state(prior[anchor_pos])
    for i in range(anchor_seg, len(starts)):
        s, e = starts[i], ends[i]
        if s not in out:
            out[s] = clone_state(current)
            origins[s] = generation
        current = state_after_segment(model, ns, ids, s, e, current)
        out[e] = clone_state(current)
        origins[e] = generation
    return out, origins, clone_state(current)


def generation_capability(model, ns, tok):
    ordinary = resid.v1.evaluate_heldout(model, ns, tok)
    status = resid.score_status_heldout(model, ns, tok)
    longhist = resid.evaluate_detailed_suite(model, ns, tok)
    ordinary_correct = sum(int(x["correct"]) for x in ordinary["rows"])
    status_correct = int(status["correct"])
    long_correct = int(longhist["full_success_count"])
    total = len(ordinary["rows"]) + int(status["count"]) + int(longhist["fixture_count"])
    correct = ordinary_correct + status_correct + long_correct
    return {
        "fixed_composite_accuracy": correct / max(total, 1),
        "fixed_composite_correct": correct,
        "fixed_composite_count": total,
        "ordinary_accuracy": ordinary["accuracy"],
        "ordinary_correction_accuracy": ordinary["correction_accuracy"],
        "ordinary_control_accuracy": ordinary["control_accuracy"],
        "status_holdout_accuracy": status["accuracy"],
        "long_history_full_success": longhist["full_success_count"],
        "long_history_count": longhist["fixture_count"],
        "superseded_full_success": longhist["by_family"]["superseded"]["full_success"],
    }


def train_stage(model, ns, tok, trainable, rows, lr, seed_offset):
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    order = list(range(len(rows)))
    random.Random(rw.SEED + seed_offset).shuffle(order)
    total = 0.0
    for idx in order:
        opt.zero_grad(set_to_none=True)
        loss = rw.train_example(model, ns, tok, *rows[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        total += float(loss.detach())
    return total / max(len(rows), 1)


def compare(rows, native_rows):
    return ca.compare_rows(rows, native_rows)


def run():
    try:
        from huggingface_hub import hf_hub_download

        torch.manual_seed(rw.SEED)
        random.seed(rw.SEED)
        torch.set_num_threads(2)

        weight_path = hf_hub_download(
            repo_id=scale.WEIGHT_REPO,
            filename=scale.WEIGHT_FILE,
            revision=scale.WEIGHT_REVISION,
        )
        model, args, ns = scale.load_reference_scaled(weight_path)

        with tempfile.TemporaryDirectory() as td:
            vocab_path = Path(td) / "vocab.txt"
            urllib.request.urlretrieve(rw.VOCAB_URL, vocab_path)
            tok = rw.RWKVTokenizer(str(vocab_path))

            # Capture W1 event-boundary lineages before any training.
            prepared = []
            for fx in FIXTURES:
                ids = tok.encode("".join(fx["segments"]))
                starts, ends = ca.boundaries(tok, fx["segments"])
                positions = sorted(set(starts + ends + [0, len(ids)]))
                states = {p: ca.state_at(model, ns, ids[:p], args) for p in positions}
                prepared.append({
                    "fx": fx,
                    "ids": ids,
                    "starts": starts,
                    "ends": ends,
                    "w1_lineage": states,
                    "w1_origins": {p: "W1" for p in positions},
                })

            trainable = scale.configure_kvr(model, args)
            w1_weights = save_kvr(model, args)
            w1_cap = generation_capability(model, ns, tok)

            # W2: the known sup-heavy intermediate. It already solves overwrite itself
            # but leaves the suffix-retention weakness, which gives W3 a real gain target.
            w2_loss = train_stage(model, ns, tok, trainable, W2_ROWS, W2_LR, 101)
            model.eval()
            w2_weights = save_kvr(model, args)
            w2_cap = generation_capability(model, ns, tok)

            # W3: low-LR disjoint status-retention patch. Prior residual diagnostic
            # showed this route can reach 38/38 while preserving correction/control.
            w3_loss = train_stage(model, ns, tok, trainable, W3_ROWS, W3_LR, 202)
            model.eval()
            w3_weights = save_kvr(model, args)
            w3_cap = generation_capability(model, ns, tok)

            strict_gain = (
                w1_cap["fixed_composite_accuracy"] < w2_cap["fixed_composite_accuracy"]
                < w3_cap["fixed_composite_accuracy"]
            )

            fixture_rows = []
            hop1_ok = 0
            hop2_ok = 0
            final_ok = 0
            w3_valid = 0
            hop2_w2_origin = 0

            for item in prepared:
                fx = item["fx"]
                ids = item["ids"]
                starts = item["starts"]
                ends = item["ends"]
                T = len(ids)
                specs = fx["probes"]
                w1_lineage = clone_state_map(item["w1_lineage"])
                w1_origins = dict(item["w1_origins"])

                # Hop 1: actual W1 lineage -> W2.
                load_kvr(model, w2_weights)
                w2_native = ca.state_at(model, ns, ids, args)
                w2_native_rows = ca.score_specs(model, ns, tok, w2_native, specs)
                hop1 = target_free_select(model, ns, tok, ids, starts, ends, w1_lineage, specs, fx["segments"])
                hop1_comp = compare(hop1["rows"], w2_native_rows)
                hop1_ok += int(hop1_comp["decision_agreement"] == 1.0)
                w2_lineage, w2_origins, w2_current = rebuild_lineage(
                    model, ns, ids, starts, ends, w1_lineage, w1_origins, hop1["chosen_seg"], "W2"
                )
                carried_error = rw.state_distance(w2_current, w2_native)

                # Hop 2: the ACTUAL mixed W1/W2 lineage -> W3. No W2-native reset.
                load_kvr(model, w3_weights)
                w3_native = ca.state_at(model, ns, ids, args)
                w3_native_rows = ca.score_specs(model, ns, tok, w3_native, specs)
                native_expected = sum(int(x["correct"]) for x in w3_native_rows) / max(len(w3_native_rows), 1)
                w3_valid += int(native_expected == 1.0)

                hop2 = target_free_select(model, ns, tok, ids, starts, ends, w2_lineage, specs, fx["segments"])
                hop2_comp = compare(hop2["rows"], w3_native_rows)
                hop2_ok += int(hop2_comp["decision_agreement"] == 1.0)
                anchor_origin = w2_origins[starts[hop2["chosen_seg"]]]
                hop2_w2_origin += int(anchor_origin == "W2")

                w3_lineage, w3_origins, w3_current = rebuild_lineage(
                    model, ns, ids, starts, ends, w2_lineage, w2_origins, hop2["chosen_seg"], "W3"
                )
                final_rows = ca.score_specs(model, ns, tok, w3_current, specs)
                final_comp = compare(final_rows, w3_native_rows)
                final_ok += int(final_comp["decision_agreement"] == 1.0)

                # Evaluation-only direct W1->W3 control. It is not used to select the chained path.
                direct = target_free_select(model, ns, tok, ids, starts, ends, w1_lineage, specs, fx["segments"])
                direct_comp = compare(direct["rows"], w3_native_rows)

                fixture_rows.append({
                    "fixture": fx["id"],
                    "history_tokens": T,
                    "w2_native_expected_accuracy": sum(int(x["correct"]) for x in w2_native_rows) / max(len(w2_native_rows), 1),
                    "w3_native_expected_accuracy": native_expected,
                    "hop1_w1_to_w2": {
                        "selected_anchor_event": hop1["chosen_seg"],
                        "anchor_pos": hop1["anchor_pos"],
                        "replay_fraction": (T - hop1["anchor_pos"]) / max(T, 1),
                        "selected_segments": hop1["selected_segments"],
                        "vs_w2_native": hop1_comp,
                    },
                    "carried_w2_lineage_before_hop2": {
                        "state_error_vs_full_w2_native": carried_error,
                        "final_source_generation": w2_origins[T],
                    },
                    "hop2_actual_lineage_to_w3": {
                        "selected_anchor_event": hop2["chosen_seg"],
                        "anchor_pos": hop2["anchor_pos"],
                        "anchor_source_generation": anchor_origin,
                        "replay_fraction": (T - hop2["anchor_pos"]) / max(T, 1),
                        "selected_segments": hop2["selected_segments"],
                        "vs_w3_native": hop2_comp,
                        "semantic_drift_top5": hop2["semantic_drift_top5"],
                    },
                    "final_chain_w3": {
                        "state_error_vs_w3_native": rw.state_distance(w3_current, w3_native),
                        "functional_vs_w3_native": final_comp,
                        "final_source_generation": w3_origins[T],
                    },
                    "direct_w1_to_w3_control": {
                        "selected_anchor_event": direct["chosen_seg"],
                        "anchor_pos": direct["anchor_pos"],
                        "vs_w3_native": direct_comp,
                    },
                })

            n = len(fixture_rows)
            suite_pass = bool(
                strict_gain
                and w3_valid == n
                and hop1_ok == n
                and hop2_ok == n
                and final_ok == n
                and hop2_w2_origin == n
            )

            write_report({
                "status": "VELA_RWKV7_0P4B_CHAINED_UPGRADE_V1",
                "source_commit": rw.SOURCE_COMMIT,
                "weight_revision": scale.WEIGHT_REVISION,
                "weight_file": scale.WEIGHT_FILE,
                "device": "cpu",
                "dtype": "float32",
                "design": {
                    "question": "Does the promoted target-free replay policy preserve functional continuity across a real RWKV-7 0.4B W1->W2->W3 carried-lineage chain, without resetting hop 2 to W2-native?",
                    "W1": "released 0.4B base checkpoint",
                    "W2": "one sup-heavy KVR pass, 16 examples, lr=1e-5",
                    "W3": "one disjoint status-retention KVR patch pass, 8 examples, lr=5e-6",
                    "trainable_scope": "blocks.*.att.{key,value,receptance}.weight",
                    "trainable_numel": int(sum(p.numel() for p in trainable)),
                    "fixtures_predeclared": FIXTURE_IDS,
                    "selector_oracle_usage": "No native full-history target is used for selection or pruning; W2/W3 native states are evaluation-only.",
                    "generation_capability_metric": "Fixed 50-case composite = 8 ordinary heldout decisions + 4 disjoint status holdout decisions + 38 long-history full-fixture successes.",
                },
                "training": {
                    "w2_mean_loss": w2_loss,
                    "w3_mean_loss": w3_loss,
                },
                "generations": {
                    "W1": w1_cap,
                    "W2": w2_cap,
                    "W3": w3_cap,
                },
                "capability_gate": {
                    "strict_w1_lt_w2_lt_w3": strict_gain,
                    "scores": [
                        w1_cap["fixed_composite_accuracy"],
                        w2_cap["fixed_composite_accuracy"],
                        w3_cap["fixed_composite_accuracy"],
                    ],
                },
                "continuity": {
                    "fixture_count": n,
                    "w3_native_valid": w3_valid,
                    "hop1_full_functional_agreement": hop1_ok,
                    "hop2_full_functional_agreement": hop2_ok,
                    "final_chain_full_functional_agreement": final_ok,
                    "hop2_anchor_source_generation_w2": hop2_w2_origin,
                    "suite_pass": suite_pass,
                },
                "fixtures": fixture_rows,
                "success_definition": "PASS requires strict fixed-metric W1<W2<W3 capability gain, W3-native validity on every predeclared fixture, 100% hop1/hop2/final functional agreement, and every hop2 selected anchor to come from the actual W2-produced lineage rather than a W1-origin bypass.",
                "claim_boundary": "One released RWKV-7 0.4B checkpoint, KVR-only synthetic adaptation, four predeclared overwrite/suffix fixtures, and a repeatedly inspected 38-fixture component in the capability metric. This qualifies the carried-lineage mechanism at larger scale; it is not final-backbone, long-horizon, or identity proof.",
            })

        del model
        gc.collect()
    except BaseException as exc:
        write_report({
            "status": "VELA_RWKV7_0P4B_CHAINED_UPGRADE_V1_ERROR",
            "weight_revision": getattr(scale, "WEIGHT_REVISION", None),
            "weight_file": getattr(scale, "WEIGHT_FILE", None),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-140:],
        })
        raise


if __name__ == "__main__":
    run()
