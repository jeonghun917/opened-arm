from __future__ import annotations

import copy
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
CA_PATH = BASE / "rwkv7-causal-anchor-v2" / "rwkv7_causal_anchor_v2.py"
spec = importlib.util.spec_from_file_location("vela_rwkv7_anchor_v2", CA_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {CA_PATH}")
ca = importlib.util.module_from_spec(spec); spec.loader.exec_module(ca)
rw = ca.rw

DEP_PATH = BASE / "rwkv7-causal-dependency-v1" / "rwkv7_causal_dependency.py"
spec2 = importlib.util.spec_from_file_location("vela_rwkv7_dep_v1", DEP_PATH)
if spec2 is None or spec2.loader is None:
    raise RuntimeError(f"cannot load {DEP_PATH}")
dep = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(dep)


def single_from_anchor(src):
    return {
        "id": "single_" + src["id"],
        "segments": list(src["segments"]),
        "probes": ca.probe_specs(src),
    }


SUPERSEDED_RETURN = {
    "id": "superseded_return_to_alpha",
    "segments": [
        "Project Orion remains active. ",
        "The old codeword was ALPHA. ",
        "Correction: the current codeword is BETA, not ALPHA. ",
        "Unrelated telemetry packet 88 was archived. ",
        "New correction: the current codeword is ALPHA, not BETA. ",
        "Verification is incomplete. ",
        "A historical memo mentions BETA but is obsolete. ",
        "External action remains blocked."
    ],
    "probes": [
        ("codeword", "\nCurrent codeword:", [" ALPHA", " BETA"], " ALPHA"),
        ("verification", "\nVerification status:", [" incomplete", " complete"], " incomplete"),
        ("project", "\nProject Orion status:", [" active", " paused"], " active"),
        ("action", "\nExternal action is:", [" blocked", " allowed"], " blocked"),
    ],
}

FIXTURES = [
    copy.deepcopy(dep.FIXTURES[0]),
    copy.deepcopy(dep.FIXTURES[2]),
    single_from_anchor(ca.FIXTURES[0]),
    single_from_anchor(ca.FIXTURES[1]),
    single_from_anchor(ca.FIXTURES[2]),
    single_from_anchor(ca.FIXTURES[3]),
    SUPERSEDED_RETURN,
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


def fixed_eval_model(model, ns, tok):
    rows = []; correct = 0; kinds = {"correction": [0, 0], "control": [0, 0]}
    for fx in rw.HELDOUT:
        scores = {c: rw.candidate_score(model, ns, tok, rw.zero_state(model.args, model), fx["prompt"], c) for c in fx["candidates"]}
        chosen = max(scores, key=scores.get); ok = chosen == fx["expected"]
        correct += int(ok); kinds[fx["kind"]][0] += int(ok); kinds[fx["kind"]][1] += 1
        rows.append({"id": fx["id"], "chosen": chosen, "expected": fx["expected"], "correct": ok, "scores": scores})
    return {
        "accuracy": correct / len(rows),
        "correction_accuracy": kinds["correction"][0] / kinds["correction"][1],
        "control_accuracy": kinds["control"][0] / kinds["control"][1],
        "rows": rows,
    }


def run():
    try:
        from huggingface_hub import hf_hub_download
        torch.manual_seed(rw.SEED); random.seed(rw.SEED); torch.set_num_threads(2)
        weight_path = hf_hub_download(repo_id=rw.WEIGHT_REPO, filename=rw.WEIGHT_FILE, revision=rw.WEIGHT_REVISION)
        model, args, ns = rw.load_reference(weight_path)

        with tempfile.TemporaryDirectory() as td:
            vp = Path(td) / "vocab.txt"
            urllib.request.urlretrieve(rw.VOCAB_URL, vp)
            tok = rw.RWKVTokenizer(str(vp))

            prepared = []
            for fx in FIXTURES:
                ids = tok.encode("".join(fx["segments"]))
                starts, ends = ca.boundaries(tok, fx["segments"])
                positions = sorted(set(starts + ends + [0, len(ids)]))
                states = {p: ca.state_at(model, ns, ids[:p], args) for p in positions}
                prepared.append({"fx": fx, "ids": ids, "starts": starts, "ends": ends, "w1_states": states})

            baseline = fixed_eval_model(model, ns, tok)
            w1_keys = ca.save_keys(model, args)
            trainable = []
            for i in range(args.n_layer):
                key = f"blocks.{i}.att.key.weight"
                model.z[key].requires_grad_(True); trainable.append(model.z[key])
            opt = torch.optim.AdamW(trainable, lr=rw.LR, weight_decay=0.0)
            order = list(range(len(rw.TRAIN))); losses = []
            for epoch in range(rw.EPOCHS):
                random.Random(rw.SEED + epoch).shuffle(order); total = 0.0
                for idx in order:
                    opt.zero_grad(set_to_none=True)
                    loss = rw.train_example(model, ns, tok, *rw.TRAIN[idx])
                    loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
                    total += float(loss.detach())
                losses.append(total / len(order))
            model.eval(); w2_keys = ca.save_keys(model, args)
            after = fixed_eval_model(model, ns, tok)

            rows = []; valid_n = 0; success_n = 0; exact_oracle_n = 0
            replay_fracs = []; oracle_fracs = []
            for item in prepared:
                fx, ids, starts, ends, states = item["fx"], item["ids"], item["starts"], item["ends"], item["w1_states"]
                T = len(ids); specs = fx["probes"]
                ca.load_keys(model, w2_keys)
                native = ca.state_at(model, ns, ids, args)
                native_rows = ca.score_specs(model, ns, tok, native, specs)
                native_expected = sum(int(r["correct"]) for r in native_rows) / len(native_rows)
                valid = native_expected == 1.0
                valid_n += int(valid)

                event_scores = []
                for i, (s, e) in enumerate(zip(starts, ends)):
                    pre = states[s]
                    ca.load_keys(model, w1_keys)
                    with torch.no_grad():
                        _, old_after = rw.run_tokens(model, ns, ids[s:e], rw.clone_state(pre))
                    ca.load_keys(model, w2_keys)
                    with torch.no_grad():
                        _, new_after = rw.run_tokens(model, ns, ids[s:e], rw.clone_state(pre))
                    sem = ca.semantic_change(model, ns, tok, old_after, new_after, specs)
                    event_scores.append({
                        "segment": i, "start": s, "end": e, "text": fx["segments"][i],
                        "state_rms": rw.state_distance(old_after, new_after)["rms"], **sem,
                    })
                selected, stats = choose_candidates(event_scores)

                ca.load_keys(model, w2_keys)
                replay_cache = {}
                def replay_at(seg):
                    if seg not in replay_cache:
                        pos = starts[seg]
                        mig = ca.migrate_from(model, ns, ids, pos, states[pos])
                        rr = ca.score_specs(model, ns, tok, mig, specs)
                        replay_cache[seg] = {"state": mig, "rows": rr}
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

                comp = ca.compare_rows(ref["rows"], native_rows)
                success = valid and comp["decision_agreement"] == 1.0
                success_n += int(success)

                oracle = []
                if valid:
                    for i, pos in enumerate(starts):
                        mig = ca.migrate_from(model, ns, ids, pos, states[pos])
                        rr = ca.score_specs(model, ns, tok, mig, specs)
                        cc = ca.compare_rows(rr, native_rows)
                        if cc["decision_agreement"] == 1.0:
                            oracle.append((i, pos))
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
                    "fixture": fx["id"], "history_tokens": T, "fixture_valid": valid,
                    "w2_native_expected_accuracy": native_expected,
                    "selected_causal_segments": [x["segment"] for x in selected],
                    "detector_stats": stats, "prune_trace": prune_trace,
                    "target_free_selected_anchor": {
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
                "status": "VELA_RWKV7_TARGET_FREE_ANCHOR_SELECTOR_V1",
                "source_commit": rw.SOURCE_COMMIT,
                "weight_revision": rw.WEIGHT_REVISION,
                "weight_file": rw.WEIGHT_FILE,
                "capability": {
                    "baseline": baseline["accuracy"], "after": after["accuracy"],
                    "correction_before": baseline["correction_accuracy"], "correction_after": after["correction_accuracy"],
                    "control_before": baseline["control_accuracy"], "control_after": after["control_accuracy"],
                    "epoch_loss": losses,
                },
                "selector": "Same deployable-style rule as Mamba target-free selector: semantic W1-vs-W2 event drift, robust candidate threshold, conservative earliest start, then forward pruning only when current functional decision signatures agree. W2-native is evaluation-only.",
                "valid_fixture_count": valid_n,
                "valid_functional_success_count": success_n,
                "valid_functional_success_rate": success_n / max(valid_n, 1),
                "exact_oracle_latest_match_count": exact_oracle_n,
                "exact_oracle_latest_match_rate": exact_oracle_n / max(valid_n, 1),
                "mean_selected_replay_fraction_valid": sum(replay_fracs) / max(len(replay_fracs), 1),
                "mean_oracle_replay_fraction_valid": sum(oracle_fracs) / max(len(oracle_fracs), 1),
                "fixtures": rows,
                "success_definition": "Cross-architecture target-free support requires real RWKV-7 W2 capability gain and zero functional false negatives on valid fixtures without consulting W2-native full-history state during anchor selection.",
                "claim_boundary": "Seven small synthetic RWKV-7 0.1B histories with hand-specified semantic probes. One fixture is a new superseded-value return case and all invalid W2-native fixtures are excluded from selector success rate. Not generic causal discovery or identity proof.",
            }
            write_report(report)
    except BaseException as exc:
        write_report({
            "status": "VELA_RWKV7_TARGET_FREE_ANCHOR_SELECTOR_V1_ERROR",
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-50:],
        })
        raise


if __name__ == "__main__":
    run()
