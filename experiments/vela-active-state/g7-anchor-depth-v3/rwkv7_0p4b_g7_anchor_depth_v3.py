from __future__ import annotations

import importlib.util
import json
import os
import random
import tempfile
import traceback
import urllib.request
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
TARGET_FIXTURES = ("superseded_suffix4", "superseded_suffix8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the already-audited G7-v2 future streams, horizon definitions, and
# functional stability scorer. This run is diagnostic/adaptive: TARGET_FIXTURES
# are exactly the four migration-only failure cases discovered by G7-v2.
g7v2 = load_module(
    "vela_g7v2_for_anchor_depth",
    BASE / "g7-cause-isolation-v2" / "rwkv7_0p4b_g7_cause_isolation_v2.py",
)
chain = g7v2.chain


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def expected_accuracy(rows):
    return sum(int(row["correct"]) for row in rows) / max(len(rows), 1)


def anchor_state(model, ns, ids, pos, lineage):
    return chain.state_to_end(model, ns, ids, pos, lineage[pos])


def run():
    try:
        from huggingface_hub import hf_hub_download

        torch.manual_seed(chain.rw.SEED)
        random.seed(chain.rw.SEED)
        torch.set_num_threads(2)

        weight_path = hf_hub_download(
            repo_id=chain.scale.WEIGHT_REPO,
            filename=chain.scale.WEIGHT_FILE,
            revision=chain.scale.WEIGHT_REVISION,
        )
        model, args, ns = chain.scale.load_reference_scaled(weight_path)

        with tempfile.TemporaryDirectory() as td:
            vocab_path = Path(td) / "vocab.txt"
            urllib.request.urlretrieve(chain.rw.VOCAB_URL, vocab_path)
            tok = chain.rw.RWKVTokenizer(str(vocab_path))

            futures = {}
            for stream in g7v2.FUTURE_STREAMS:
                ids, meta = g7v2.build_future(tok, stream, g7v2.MAX_HORIZON)
                futures[stream["id"]] = {"ids": ids, "meta": meta}

            prepared = []
            for fx in chain.FIXTURES:
                if fx["id"] not in TARGET_FIXTURES:
                    continue
                ids = tok.encode("".join(fx["segments"]))
                starts, ends = chain.ca.boundaries(tok, fx["segments"])
                positions = sorted(set(starts + ends + [0, len(ids)]))
                states = {
                    p: chain.ca.state_at(model, ns, ids[:p], args)
                    for p in positions
                }
                prepared.append({
                    "fx": fx,
                    "ids": ids,
                    "starts": starts,
                    "ends": ends,
                    "w1_lineage": states,
                    "w1_origins": {p: "W1" for p in positions},
                })

            if [x["fx"]["id"] for x in prepared] != list(TARGET_FIXTURES):
                raise RuntimeError(
                    f"fixture mismatch: {[x['fx']['id'] for x in prepared]}"
                )

            trainable = chain.scale.configure_kvr(model, args)
            w2_loss = chain.train_stage(
                model, ns, tok, trainable, chain.W2_ROWS, chain.W2_LR, 101
            )
            model.eval()
            w2_weights = chain.save_kvr(model, args)
            w3_loss = chain.train_stage(
                model, ns, tok, trainable, chain.W3_ROWS, chain.W3_LR, 202
            )
            model.eval()
            w3_weights = chain.save_kvr(model, args)

            cases = []
            aggregate_rows = []

            for item in prepared:
                fx = item["fx"]
                ids = item["ids"]
                starts = item["starts"]
                ends = item["ends"]
                specs = fx["probes"]
                T = len(ids)
                w1_lineage = chain.clone_state_map(item["w1_lineage"])
                w1_origins = dict(item["w1_origins"])

                chain.load_kvr(model, w2_weights)
                w2_native = chain.ca.state_at(model, ns, ids, args)
                w2_native_rows = chain.ca.score_specs(model, ns, tok, w2_native, specs)
                hop1 = chain.target_free_select(
                    model, ns, tok, ids, starts, ends, w1_lineage, specs, fx["segments"]
                )
                hop1_comp = chain.compare(hop1["rows"], w2_native_rows)
                w2_lineage, w2_origins, _ = chain.rebuild_lineage(
                    model, ns, ids, starts, ends,
                    w1_lineage, w1_origins, hop1["chosen_seg"], "W2"
                )

                chain.load_kvr(model, w3_weights)
                w3_native = chain.ca.state_at(model, ns, ids, args)
                w3_native_rows = chain.ca.score_specs(model, ns, tok, w3_native, specs)
                hop2 = chain.target_free_select(
                    model, ns, tok, ids, starts, ends, w2_lineage, specs, fx["segments"]
                )
                selected_seg = hop2["chosen_seg"]
                selected_pos = starts[selected_seg]
                selected_origin = w2_origins[selected_pos]

                anchors = []
                for seg, pos in enumerate(starts):
                    if pos not in w2_lineage:
                        continue
                    origin = w2_origins.get(pos)
                    if origin not in {"W1", "W2"}:
                        continue
                    state = anchor_state(model, ns, ids, pos, w2_lineage)
                    rows = chain.ca.score_specs(model, ns, tok, state, specs)
                    comp = chain.compare(rows, w3_native_rows)
                    anchors.append({
                        "seg": seg,
                        "pos": pos,
                        "origin": origin,
                        "replay_fraction": (T - pos) / max(T, 1),
                        "immediate_decision_agreement": comp["decision_agreement"],
                        "immediate_expected_accuracy": expected_accuracy(rows),
                        "state": state,
                    })

                # Full W3 replay is a positive control. It must equal the W3-native
                # state path modulo deterministic numerical execution.
                full_replay_state = chain.ca.state_at(model, ns, ids, args)

                for stream_id, stream in futures.items():
                    native_control = g7v2.roll_path(
                        model, ns, tok, full_replay_state, w3_native, specs, stream["ids"]
                    )
                    native_stable = native_control["native_stable"]

                    anchor_rows = []
                    for anchor in anchors:
                        roll = g7v2.roll_path(
                            model, ns, tok, anchor["state"], w3_native, specs, stream["ids"]
                        )
                        qualified = bool(
                            native_stable
                            and anchor["immediate_decision_agreement"] == 1.0
                            and anchor["immediate_expected_accuracy"] == 1.0
                            and roll["path_stable_vs_native"]
                        )
                        anchor_rows.append({
                            "seg": anchor["seg"],
                            "pos": anchor["pos"],
                            "origin": anchor["origin"],
                            "replay_fraction": anchor["replay_fraction"],
                            "immediate_decision_agreement": anchor["immediate_decision_agreement"],
                            "immediate_expected_accuracy": anchor["immediate_expected_accuracy"],
                            "path_stable_vs_native": roll["path_stable_vs_native"],
                            "first_trace_divergence_step": roll["first_trace_divergence_step"],
                            "trace_agreement": roll["trace_agreement"],
                            "milestones": roll["milestones"],
                            "qualified_long_horizon": qualified,
                            "is_chain_v2_selected": anchor["seg"] == selected_seg,
                        })

                    successful_w2 = [
                        a for a in anchor_rows
                        if a["origin"] == "W2" and a["qualified_long_horizon"]
                    ]
                    successful_w1 = [
                        a for a in anchor_rows
                        if a["origin"] == "W1" and a["qualified_long_horizon"]
                    ]
                    selected_row = next(
                        a for a in anchor_rows if a["seg"] == selected_seg
                    )
                    latest_safe_w2 = max(successful_w2, key=lambda a: a["pos"]) if successful_w2 else None
                    earliest_safe_w2 = min(successful_w2, key=lambda a: a["pos"]) if successful_w2 else None

                    if not native_stable:
                        diagnosis = "native_unstable_excluded"
                    elif selected_row["qualified_long_horizon"]:
                        diagnosis = "selected_anchor_already_stable"
                    elif successful_w2:
                        diagnosis = "w2_anchor_choice_can_rescue"
                    elif successful_w1:
                        diagnosis = "w2_carried_state_failure_w1_anchor_rescues"
                    else:
                        diagnosis = "no_carried_anchor_rescues_full_replay_required"

                    row = {
                        "fixture": fx["id"],
                        "future_stream": stream_id,
                        "future_meta": stream["meta"],
                        "native_stable": native_stable,
                        "full_replay_control_stable": native_control["path_stable_vs_native"],
                        "hop1_decision_agreement": hop1_comp["decision_agreement"],
                        "w3_native_expected_accuracy": expected_accuracy(w3_native_rows),
                        "chain_v2_selected": {
                            "seg": selected_seg,
                            "pos": selected_pos,
                            "origin": selected_origin,
                            "qualified_long_horizon": selected_row["qualified_long_horizon"],
                        },
                        "eligible_anchor_count": len(anchor_rows),
                        "eligible_w2_anchor_count": sum(int(a["origin"] == "W2") for a in anchor_rows),
                        "successful_w2_anchor_count": len(successful_w2),
                        "successful_w1_anchor_count": len(successful_w1),
                        "latest_safe_w2": None if latest_safe_w2 is None else {
                            "seg": latest_safe_w2["seg"],
                            "pos": latest_safe_w2["pos"],
                            "replay_fraction": latest_safe_w2["replay_fraction"],
                        },
                        "earliest_safe_w2": None if earliest_safe_w2 is None else {
                            "seg": earliest_safe_w2["seg"],
                            "pos": earliest_safe_w2["pos"],
                            "replay_fraction": earliest_safe_w2["replay_fraction"],
                        },
                        "diagnosis": diagnosis,
                        "anchors": anchor_rows,
                    }
                    cases.append(row)
                    aggregate_rows.append({
                        "fixture": fx["id"],
                        "future_stream": stream_id,
                        "native_stable": native_stable,
                        "selected_stable": selected_row["qualified_long_horizon"],
                        "any_w2_rescue": bool(successful_w2),
                        "any_w1_rescue": bool(successful_w1),
                        "diagnosis": diagnosis,
                        "latest_safe_w2_seg": None if latest_safe_w2 is None else latest_safe_w2["seg"],
                    })

            native_stable_rows = [r for r in aggregate_rows if r["native_stable"]]
            failing_selected = [
                r for r in native_stable_rows if not r["selected_stable"]
            ]
            w2_rescued = [r for r in failing_selected if r["any_w2_rescue"]]
            w1_only = [
                r for r in failing_selected
                if not r["any_w2_rescue"] and r["any_w1_rescue"]
            ]
            full_replay_only = [
                r for r in failing_selected
                if not r["any_w2_rescue"] and not r["any_w1_rescue"]
            ]

            report = {
                "status": "VELA_G7_RWKV7_0P4B_ANCHOR_DEPTH_V3",
                "source_commit": os.environ.get("GITHUB_SHA"),
                "device": "cpu",
                "dtype": "float32",
                "model": {
                    "weight_repo": chain.scale.WEIGHT_REPO,
                    "weight_file": chain.scale.WEIGHT_FILE,
                    "weight_revision": chain.scale.WEIGHT_REVISION,
                },
                "protocol": {
                    "purpose": (
                        "Adaptive post-G7-v2 diagnostic: exhaustively test every carried "
                        "W1/W2 event-start anchor on the two fixtures that produced all four "
                        "native-stable migration-only failures."
                    ),
                    "target_fixtures": list(TARGET_FIXTURES),
                    "future_streams": [x["id"] for x in g7v2.FUTURE_STREAMS],
                    "horizons": list(g7v2.HORIZONS),
                    "qualification": (
                        "native stable at every horizon; anchor immediately matches W3-native "
                        "and remains functionally stable at every horizon"
                    ),
                    "oracle_usage": (
                        "YES, evaluation-only exhaustive ablation. W3-native outcomes are used "
                        "to label safe anchors. This run may diagnose selector/replay depth but "
                        "cannot itself define a production target-free selector."
                    ),
                    "full_replay_control": "W3 from-start replay / W3-native",
                },
                "training": {
                    "w2_mean_loss": w2_loss,
                    "w3_mean_loss": w3_loss,
                },
                "summary": {
                    "case_count": len(aggregate_rows),
                    "native_stable_case_count": len(native_stable_rows),
                    "selected_failure_count": len(failing_selected),
                    "w2_anchor_rescue_count": len(w2_rescued),
                    "w1_only_rescue_count": len(w1_only),
                    "full_replay_only_count": len(full_replay_only),
                    "w2_anchor_rescue_cases": w2_rescued,
                    "w1_only_rescue_cases": w1_only,
                    "full_replay_only_cases": full_replay_only,
                    "all_selected_failures_rescuable_by_w2_anchor": (
                        len(failing_selected) > 0 and len(w2_rescued) == len(failing_selected)
                    ),
                },
                "cases": cases,
                "interpretation_rule": {
                    "selector_depth_family": (
                        "All native-stable selected failures have at least one successful W2 anchor."
                    ),
                    "carried_w2_state_family": (
                        "A selected failure has no successful W2 anchor but a successful W1 anchor."
                    ),
                    "full_replay_required_family": (
                        "A native-stable selected failure has no successful carried W1/W2 anchor."
                    ),
                },
                "claim_boundary": (
                    "Adaptive diagnostic on the already-observed G7-v2 failures. It is not a "
                    "fresh gate and must not be reported as independent promotion evidence."
                ),
            }
            write_report(report)

    except BaseException as exc:
        write_report({
            "status": "VELA_G7_RWKV7_0P4B_ANCHOR_DEPTH_V3_ERROR",
            "source_commit": os.environ.get("GITHUB_SHA"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-80:],
        })
        raise


if __name__ == "__main__":
    run()
