from __future__ import annotations

import importlib.util
import json
import os
import random
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

# Keep this first chained probe small enough for CPU CI while covering:
# one multi-slot persistent history + correction at early/middle/late positions.
FIXTURES = [sel.FIXTURES[0], sel.FIXTURES[2], sel.FIXTURES[3], sel.FIXTURES[4]]
TOTAL_EPOCHS = 3
W2_EPOCH = 1
W3_EPOCH = 3


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def clone_cache(x):
    return None if x is None else v4.clone_cache(x)


def clone_cache_map(m):
    return {k: clone_cache(v) for k, v in m.items()}


def save_xproj(model):
    return v2.save_xproj(model)


def load_xproj(model, snap):
    v2.load_xproj(model, snap)


def train_with_generation_snapshots(model, tok):
    trainable = []
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if ".mixer.x_proj.weight" in name:
            p.requires_grad_(True); trainable.append(p)
    if not trainable:
        raise RuntimeError("no x_proj weights found")
    opt = torch.optim.AdamW(trainable, lr=v3.LR, weight_decay=0.0)
    order = list(range(len(v3.TRAIN)))
    losses = []
    snapshots = {}
    evals = {}
    for epoch in range(1, TOTAL_EPOCHS + 1):
        random.Random(v3.SEED + epoch - 1).shuffle(order)
        model.train(); total = 0.0
        for idx in order:
            prompt, gold = v3.TRAIN[idx]
            opt.zero_grad(set_to_none=True)
            loss = v3.train_loss(model, tok, prompt, gold)
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            total += float(loss.detach())
        losses.append(total / len(order))
        model.eval()
        if epoch in (W2_EPOCH, W3_EPOCH):
            snapshots[epoch] = save_xproj(model)
            evals[epoch] = v3.evaluate(model, tok)
    return losses, snapshots, evals


def score_rows(model, tok, state, T, specs):
    return fa.score_probe_specs(model, tok, state, T, specs)


def compare_rows(rows, native_rows):
    return fa.compare_rows(rows, native_rows)


def target_free_select(model, tok, ids, starts, ends, lineage, specs, segments):
    T = int(ids.shape[1])
    event_scores = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        old_after = lineage[e]
        new_after = dep.run_slice(model, ids, s, e, lineage[s])
        sem = v3a.compare_local_semantics(model, tok, old_after, new_after, e, specs)
        event_scores.append({
            "segment": i, "start": s, "end": e, "text": segments[i],
            "state_rms": v3.cache_distance(old_after, new_after)["rms"], **sem,
        })
    selected, stats = sel.choose_candidates(event_scores)
    replay_cache = {}

    def replay_at(seg):
        if seg not in replay_cache:
            s = starts[seg]
            migrated = dep.run_slice(model, ids, s, T, lineage[s])
            rows = score_rows(model, tok, migrated, T, specs)
            replay_cache[seg] = {"state": migrated, "rows": rows}
        return replay_cache[seg]

    chosen_seg = selected[0]["segment"]
    ref = replay_at(chosen_seg)
    prune_trace = []
    for cand in selected[1:]:
        seg = cand["segment"]
        cur = replay_at(seg)
        same = sel.decision_signature(cur["rows"]) == sel.decision_signature(ref["rows"])
        prune_trace.append({
            "from_segment": chosen_seg,
            "candidate_later_segment": seg,
            "same_functional_signature": same,
        })
        if same:
            chosen_seg = seg; ref = cur
    return {
        "chosen_seg": chosen_seg,
        "anchor_pos": starts[chosen_seg],
        "migrated_state": ref["state"],
        "rows": ref["rows"],
        "selected_segments": [x["segment"] for x in selected],
        "selected_event_texts": [x["text"] for x in selected],
        "detector_stats": stats,
        "prune_trace": prune_trace,
        "semantic_drift_top4": sorted(event_scores, key=lambda r: r["detector_score"], reverse=True)[:4],
    }


def rebuild_lineage(model, ids, starts, ends, prior_lineage, prior_origins, anchor_seg, generation):
    anchor_pos = starts[anchor_seg]
    new_lineage = {}
    new_origins = {}
    for pos, state in prior_lineage.items():
        if pos <= anchor_pos:
            new_lineage[pos] = clone_cache(state)
            new_origins[pos] = prior_origins[pos]
    current = clone_cache(prior_lineage[anchor_pos])
    for i in range(anchor_seg, len(starts)):
        s, e = starts[i], ends[i]
        new_lineage[s] = clone_cache(current)
        if s not in new_origins:
            new_origins[s] = prior_origins.get(s, generation)
        current = dep.run_slice(model, ids, s, e, current)
        new_lineage[e] = clone_cache(current)
        new_origins[e] = generation
    return new_lineage, new_origins, clone_cache(current)


def oracle_latest_safe(model, tok, ids, starts, lineage, specs, native_rows):
    T = int(ids.shape[1])
    safe = []
    for i, s in enumerate(starts):
        mig = dep.run_slice(model, ids, s, T, lineage[s])
        rows = score_rows(model, tok, mig, T, specs)
        if compare_rows(rows, native_rows)["decision_agreement"] == 1.0:
            safe.append((i, s))
    if not safe:
        return None
    i, s = max(safe, key=lambda x: x[1])
    return {"event_index": i, "anchor_pos": s, "replay_fraction": (T-s)/max(T,1)}


def agreement(a, b):
    bb = {r["id"]: r for r in b}
    return sum(int(r["chosen"] == bb[r["id"]]["chosen"]) for r in a) / len(a)


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
            origins = {p: "W1" for p in positions}
            prepared.append({
                "fx": fx, "ids": ids, "T": T, "starts": starts, "ends": ends,
                "w1_lineage": caches, "w1_origins": origins,
            })

        w1_cap = v3.evaluate(model, tok)
        w1_weights = save_xproj(model)
        losses, snapshots, evals = train_with_generation_snapshots(model, tok)
        w2_weights = snapshots[W2_EPOCH]
        w3_weights = snapshots[W3_EPOCH]
        w2_cap = evals[W2_EPOCH]
        w3_cap = evals[W3_EPOCH]

        rows = []
        hop1_success = 0; hop2_success = 0; chain_success = 0
        for item in prepared:
            fx, ids, T = item["fx"], item["ids"], item["T"]
            starts, ends = item["starts"], item["ends"]
            specs = fx["probes"]
            w1_lineage = clone_cache_map(item["w1_lineage"])
            w1_origins = dict(item["w1_origins"])

            # Hop 1: actual W1 lineage -> W2, target-free.
            load_xproj(model, w2_weights)
            with torch.no_grad():
                w2_native = clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            w2_native_rows = score_rows(model, tok, w2_native, T, specs)
            hop1 = target_free_select(model, tok, ids, starts, ends, w1_lineage, specs, fx["segments"])
            hop1_comp = compare_rows(hop1["rows"], w2_native_rows)
            hop1_success += int(hop1_comp["decision_agreement"] == 1.0)
            w2_lineage, w2_origins, w2_current = rebuild_lineage(
                model, ids, starts, ends, w1_lineage, w1_origins, hop1["chosen_seg"], "W2"
            )

            # Hop 2: use the ACTUAL mixed lineage produced by Hop 1, not a fake full W2-native history.
            load_xproj(model, w3_weights)
            with torch.no_grad():
                w3_native = clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            w3_native_rows = score_rows(model, tok, w3_native, T, specs)
            hop2 = target_free_select(model, tok, ids, starts, ends, w2_lineage, specs, fx["segments"])
            hop2_comp = compare_rows(hop2["rows"], w3_native_rows)
            hop2_success += int(hop2_comp["decision_agreement"] == 1.0)
            w3_lineage, w3_origins, w3_current = rebuild_lineage(
                model, ids, starts, ends, w2_lineage, w2_origins, hop2["chosen_seg"], "W3"
            )
            chain_rows = score_rows(model, tok, w3_current, T, specs)
            chain_comp = compare_rows(chain_rows, w3_native_rows)
            chain_success += int(chain_comp["decision_agreement"] == 1.0)

            # Direct W1 -> W3 is a control, not the actual chained path.
            direct = target_free_select(model, tok, ids, starts, ends, w1_lineage, specs, fx["segments"])
            direct_comp = compare_rows(direct["rows"], w3_native_rows)

            # Evaluation-only oracle frontier for the second hop.
            oracle2 = oracle_latest_safe(model, tok, ids, starts, w2_lineage, specs, w3_native_rows)
            hop2_origin = w2_origins[starts[hop2["chosen_seg"]]]

            rows.append({
                "fixture": fx["id"], "history_tokens": T,
                "w2_native_expected_accuracy": sum(int(r["correct"]) for r in w2_native_rows)/len(w2_native_rows),
                "w3_native_expected_accuracy": sum(int(r["correct"]) for r in w3_native_rows)/len(w3_native_rows),
                "hop1_w1_to_w2": {
                    "selected_anchor_event": hop1["chosen_seg"], "anchor_pos": hop1["anchor_pos"],
                    "replay_fraction": (T-hop1["anchor_pos"])/max(T,1),
                    "selected_segments": hop1["selected_segments"], "prune_trace": hop1["prune_trace"],
                    "vs_w2_native": hop1_comp,
                },
                "hop2_actual_lineage_to_w3": {
                    "selected_anchor_event": hop2["chosen_seg"], "anchor_pos": hop2["anchor_pos"],
                    "anchor_source_generation": hop2_origin,
                    "replay_fraction": (T-hop2["anchor_pos"])/max(T,1),
                    "selected_segments": hop2["selected_segments"], "prune_trace": hop2["prune_trace"],
                    "vs_w3_native": hop2_comp,
                    "oracle_latest_safe_anchor": oracle2,
                },
                "final_chain_w3": {
                    "state_error_vs_w3_native": v3.cache_distance(w3_current, w3_native),
                    "functional_vs_w3_native": chain_comp,
                },
                "direct_w1_to_w3_control": {
                    "selected_anchor_event": direct["chosen_seg"], "anchor_pos": direct["anchor_pos"],
                    "replay_fraction": (T-direct["anchor_pos"])/max(T,1),
                    "vs_w3_native": direct_comp,
                    "chain_vs_direct_decision_agreement": agreement(chain_rows, direct["rows"]),
                },
                "lineage_origin_at_final": w3_origins[T],
                "hop2_semantic_drift_top4": hop2["semantic_drift_top4"],
            })

        n = len(rows)
        report = {
            "status": "VELA_SEQUENTIAL_W1_W2_W3_CHAIN_V1",
            "model": v3.MODEL_ID,
            "torch_version": torch.__version__, "transformers_version": transformers_version,
            "generations": {
                "W1": {"training_epoch": 0, "accuracy": w1_cap["accuracy"], "correction_accuracy": w1_cap["correction_accuracy"], "control_accuracy": w1_cap["control_accuracy"]},
                "W2": {"training_epoch": W2_EPOCH, "accuracy": w2_cap["accuracy"], "correction_accuracy": w2_cap["correction_accuracy"], "control_accuracy": w2_cap["control_accuracy"]},
                "W3": {"training_epoch": W3_EPOCH, "accuracy": w3_cap["accuracy"], "correction_accuracy": w3_cap["correction_accuracy"], "control_accuracy": w3_cap["control_accuracy"]},
                "epoch_mean_loss": losses,
            },
            "capability_gate": {
                "w1_to_w2_gain": w2_cap["accuracy"] - w1_cap["accuracy"],
                "w2_to_w3_gain": w3_cap["accuracy"] - w2_cap["accuracy"],
                "strict_two_step_gain": bool(w2_cap["accuracy"] > w1_cap["accuracy"] and w3_cap["accuracy"] > w2_cap["accuracy"]),
            },
            "continuity": {
                "hop1_full_functional_agreement_rate": hop1_success/n,
                "hop2_full_functional_agreement_rate": hop2_success/n,
                "final_chain_full_functional_agreement_rate": chain_success/n,
            },
            "fixtures": rows,
            "success_definition": "W1->W2 and W2->W3 must both be real capability-improving weight updates, while target-free replay on the actual carried lineage reaches each generation's native functional decisions without consulting that native target during selection.",
            "claim_boundary": "Four small synthetic Mamba-130M histories and one correction curriculum, with W2 defined after epoch 1 and W3 after epoch 3. This tests repeated migration mechanics, not general intelligence, identity, or consciousness.",
        }
        write_report(report)
    except BaseException as exc:
        write_report({
            "status": "VELA_SEQUENTIAL_W1_W2_W3_CHAIN_V1_ERROR",
            "model": getattr(v3, "MODEL_ID", None), "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-50:],
        })
        raise


if __name__ == "__main__":
    run()
