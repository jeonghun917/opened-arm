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
HORIZONS = (128, 512, 2048)
MAX_HORIZON = max(HORIZONS)
CORE_FIXTURE_IDS = (
    "superseded_base",
    "superseded_suffix4",
    "superseded_suffix8",
    "superseded_old_value_echoes",
)
FUTURE_STREAMS = (
    {
        "id": "telemetry",
        "template": (
            "Neutral telemetry packet {i:04d} was archived. "
            "Sensor checksum marker {j:02d} was logged. "
        ),
    },
    {
        "id": "inventory",
        "template": (
            "Routine inventory entry {i:04d} was filed. "
            "Auxiliary schedule marker {j:02d} was recorded. "
        ),
    },
)
NATIVE_STABLE_MIN_CASES = 6


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v2 = load_module(
    "vela_rwkv7_chain_v2_for_g7v2",
    BASE / "rwkv7-0p4b-chain-v2" / "rwkv7_0p4b_chained_provenance.py",
)
chain = v2.chain


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def expected_accuracy(rows):
    return sum(int(row["correct"]) for row in rows) / max(len(rows), 1)


def rows_by_id(rows):
    return {row["id"]: row for row in rows}


def functional_flags(rows):
    by_id = rows_by_id(rows)

    def correct(pid: str) -> bool | None:
        row = by_id.get(pid)
        return None if row is None else bool(row["correct"])

    correction_ok = correct("codeword")
    control_vals = [correct(pid) for pid in ("project", "action")]
    persistent_vals = [correct(pid) for pid in ("verification", "project", "action")]
    control_known = [x for x in control_vals if x is not None]
    persistent_known = [x for x in persistent_vals if x is not None]
    return {
        "task_invariant_ok": all(bool(row["correct"]) for row in rows),
        "correction_ok": correction_ok,
        "control_ok": None if not control_known else all(control_known),
        "persistent_fact_ok": None if not persistent_known else all(persistent_known),
    }


def expected_margin(row):
    scores = row.get("scores") or {}
    expected = row.get("expected")
    if expected not in scores or len(scores) < 2:
        return None
    return float(scores[expected] - max(v for k, v in scores.items() if k != expected))


def margin_summary(rows):
    margins = {row["id"]: expected_margin(row) for row in rows}
    known = [x for x in margins.values() if x is not None]
    return {
        "per_probe_expected_margin": margins,
        "min_expected_margin": None if not known else min(known),
        "mean_expected_margin": None if not known else sum(known) / len(known),
    }


def build_future(tok, stream: dict, token_count: int) -> tuple[list[int], dict]:
    chunks = []
    ids: list[int] = []
    i = 0
    while len(ids) < token_count:
        chunks.append(stream["template"].format(i=i, j=i % 97))
        ids = tok.encode("".join(chunks))
        i += 1
    return ids[:token_count], {
        "id": stream["id"],
        "chunks_generated": i,
        "tokens_used": token_count,
        "semantic_exclusion": (
            "templates avoid fixture project/codeword/verification/action/correction terms"
        ),
    }


def state_from_anchor(model, ns, ids, pos, state):
    return chain.state_to_end(model, ns, ids, pos, state)


def previous_w2_anchor_seg(starts, origins, chosen_seg):
    eligible = [
        seg
        for seg, pos in enumerate(starts)
        if seg < chosen_seg and origins.get(pos) == "W2"
    ]
    return max(eligible) if eligible else None


def roll_path(model, ns, tok, initial_state, native_initial_state, specs, future_ids):
    state = chain.clone_state(initial_state)
    native_state = chain.clone_state(native_initial_state)
    trace_bits = []
    first_trace_divergence = None
    milestones = {}

    for step, tid in enumerate(future_ids, start=1):
        with torch.no_grad():
            native_logits, native_state = chain.rw.run_tokens(model, ns, [tid], native_state)
            logits, state = chain.rw.run_tokens(model, ns, [tid], state)
        native_choice = int(torch.argmax(native_logits).item())
        choice = int(torch.argmax(logits).item())
        agree = native_choice == choice
        trace_bits.append(agree)
        if not agree and first_trace_divergence is None:
            first_trace_divergence = step

        if step in HORIZONS:
            native_probe = chain.ca.score_specs(model, ns, tok, native_state, specs)
            probe = chain.ca.score_specs(model, ns, tok, state, specs)
            comp = chain.compare(probe, native_probe)
            milestones[str(step)] = {
                "functional_vs_native": comp,
                "native_expected_accuracy": expected_accuracy(native_probe),
                "path_expected_accuracy": expected_accuracy(probe),
                "native_flags": functional_flags(native_probe),
                "path_flags": functional_flags(probe),
                "native_margin": margin_summary(native_probe),
                "path_margin": margin_summary(probe),
            }

    native_stable = all(
        milestones[str(h)]["native_flags"]["task_invariant_ok"] for h in HORIZONS
    )
    path_stable = all(
        milestones[str(h)]["path_flags"]["task_invariant_ok"]
        and milestones[str(h)]["functional_vs_native"]["decision_agreement"] == 1.0
        for h in HORIZONS
    )
    return {
        "native_stable": native_stable,
        "path_stable_vs_native": path_stable,
        "first_trace_divergence_step": first_trace_divergence,
        "trace_agreement": {
            str(h): sum(trace_bits[:h]) / h for h in HORIZONS
        },
        "milestones": milestones,
    }


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

            future_streams = {}
            for stream in FUTURE_STREAMS:
                ids, meta = build_future(tok, stream, MAX_HORIZON)
                future_streams[stream["id"]] = {"ids": ids, "meta": meta}

            prepared = []
            for fx in chain.FIXTURES:
                if fx["id"] not in CORE_FIXTURE_IDS:
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

            if [x["fx"]["id"] for x in prepared] != list(CORE_FIXTURE_IDS):
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

            case_rows = []
            immediate_qualified = 0

            for item in prepared:
                fx = item["fx"]
                ids = item["ids"]
                starts = item["starts"]
                ends = item["ends"]
                specs = fx["probes"]
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
                    model,
                    ns,
                    ids,
                    starts,
                    ends,
                    w1_lineage,
                    w1_origins,
                    hop1["chosen_seg"],
                    "W2",
                )

                chain.load_kvr(model, w3_weights)
                w3_native = chain.ca.state_at(model, ns, ids, args)
                w3_native_rows = chain.ca.score_specs(model, ns, tok, w3_native, specs)
                hop2 = chain.target_free_select(
                    model, ns, tok, ids, starts, ends, w2_lineage, specs, fx["segments"]
                )
                hop2_comp = chain.compare(hop2["rows"], w3_native_rows)
                selected_seg = hop2["chosen_seg"]
                selected_pos = starts[selected_seg]
                selected_origin = w2_origins[selected_pos]
                _, _, selected_state = chain.rebuild_lineage(
                    model,
                    ns,
                    ids,
                    starts,
                    ends,
                    w2_lineage,
                    w2_origins,
                    selected_seg,
                    "W3",
                )
                selected_rows = chain.ca.score_specs(model, ns, tok, selected_state, specs)
                selected_initial_comp = chain.compare(selected_rows, w3_native_rows)

                older_seg = previous_w2_anchor_seg(starts, w2_origins, selected_seg)
                older_state = None
                older_rows = None
                if older_seg is not None:
                    older_pos = starts[older_seg]
                    older_state = state_from_anchor(
                        model, ns, ids, older_pos, w2_lineage[older_pos]
                    )
                    older_rows = chain.ca.score_specs(model, ns, tok, older_state, specs)

                initial_ok = (
                    expected_accuracy(w3_native_rows) == 1.0
                    and hop1_comp["decision_agreement"] == 1.0
                    and hop2_comp["decision_agreement"] == 1.0
                    and selected_initial_comp["decision_agreement"] == 1.0
                    and selected_origin == "W2"
                )
                immediate_qualified += int(initial_ok)

                initial_info = {
                    "w3_native_expected_accuracy": expected_accuracy(w3_native_rows),
                    "hop1_decision_agreement": hop1_comp["decision_agreement"],
                    "hop2_decision_agreement": hop2_comp["decision_agreement"],
                    "selected_final_decision_agreement": selected_initial_comp["decision_agreement"],
                    "selected_anchor_seg": selected_seg,
                    "selected_anchor_pos": selected_pos,
                    "selected_anchor_origin": selected_origin,
                    "older_w2_anchor_seg": older_seg,
                    "older_w2_anchor_pos": None if older_seg is None else starts[older_seg],
                    "native_margin": margin_summary(w3_native_rows),
                    "selected_margin": margin_summary(selected_rows),
                    "older_margin": None if older_rows is None else margin_summary(older_rows),
                    "qualified": initial_ok,
                }

                for stream_id, stream_data in future_streams.items():
                    selected_roll = roll_path(
                        model,
                        ns,
                        tok,
                        selected_state,
                        w3_native,
                        specs,
                        stream_data["ids"],
                    )
                    older_roll = None
                    if older_state is not None:
                        older_roll = roll_path(
                            model,
                            ns,
                            tok,
                            older_state,
                            w3_native,
                            specs,
                            stream_data["ids"],
                        )

                    native_stable = selected_roll["native_stable"]
                    selected_stable = selected_roll["path_stable_vs_native"]
                    older_stable = (
                        None
                        if older_roll is None
                        else older_roll["path_stable_vs_native"]
                    )
                    migration_excess_failure = bool(
                        initial_ok and native_stable and not selected_stable
                    )
                    selector_depth_rescue = bool(
                        migration_excess_failure and older_stable is True
                    )

                    case_rows.append({
                        "fixture": fx["id"],
                        "future_stream": stream_id,
                        "future_meta": stream_data["meta"],
                        "initial": initial_info,
                        "selected": selected_roll,
                        "older_w2_anchor": older_roll,
                        "native_stable_case": native_stable,
                        "migration_excess_failure": migration_excess_failure,
                        "selector_depth_rescue": selector_depth_rescue,
                    })

            native_stable_cases = [r for r in case_rows if r["native_stable_case"]]
            migration_excess = [r for r in case_rows if r["migration_excess_failure"]]
            rescues = [r for r in case_rows if r["selector_depth_rescue"]]
            per_fixture_native_stable = {
                fid: sum(
                    int(r["native_stable_case"])
                    for r in case_rows
                    if r["fixture"] == fid
                )
                for fid in CORE_FIXTURE_IDS
            }
            native_coverage_gate = (
                len(native_stable_cases) >= NATIVE_STABLE_MIN_CASES
                and all(v >= 1 for v in per_fixture_native_stable.values())
            )
            migration_excess_gate = len(migration_excess) == 0
            suite_pass_candidate = (
                immediate_qualified == len(CORE_FIXTURE_IDS)
                and native_coverage_gate
                and migration_excess_gate
            )

            report = {
                "status": "VELA_G7_RWKV7_0P4B_CAUSE_ISOLATION_V2",
                "source_commit": os.environ.get("GITHUB_SHA"),
                "device": "cpu",
                "dtype": "float32",
                "model": {
                    "weight_repo": chain.scale.WEIGHT_REPO,
                    "weight_file": chain.scale.WEIGHT_FILE,
                    "weight_revision": chain.scale.WEIGHT_REVISION,
                },
                "protocol": {
                    "question": (
                        "Separate migration-induced long-horizon drift from native backbone "
                        "instability, and test whether a one-step older carried W2 anchor "
                        "rescues native-stable migration failures."
                    ),
                    "chain_policy": "frozen chain-v2 provenance selector",
                    "fixtures": list(CORE_FIXTURE_IDS),
                    "future_streams": [x["id"] for x in FUTURE_STREAMS],
                    "horizons": list(HORIZONS),
                    "native_stable_rule": (
                        "W3-native task invariant must remain true at 128, 512, and 2048 "
                        "for a fixture/future case to enter the migration-only denominator."
                    ),
                    "migration_excess_rule": (
                        "On a native-stable case, selected migrated state must keep task "
                        "invariants and functional decision agreement=1.0 at every horizon."
                    ),
                    "native_coverage_gate": (
                        f"At least {NATIVE_STABLE_MIN_CASES}/"
                        f"{len(CORE_FIXTURE_IDS) * len(FUTURE_STREAMS)} cases native-stable "
                        "and at least one native-stable stream per fixture."
                    ),
                    "anchor_ablation": (
                        "Evaluation-only comparison of chain-v2 selected W2-origin anchor "
                        "against the immediately older available W2-origin anchor. "
                        "W3-native is evaluation-only and is not used to select either anchor."
                    ),
                    "oracle_usage": "none for selector or ablation anchor choice",
                },
                "training": {
                    "w2_mean_loss": w2_loss,
                    "w3_mean_loss": w3_loss,
                },
                "summary": {
                    "fixture_count": len(CORE_FIXTURE_IDS),
                    "future_stream_count": len(FUTURE_STREAMS),
                    "case_count": len(case_rows),
                    "initial_chain_qualified": immediate_qualified,
                    "native_stable_case_count": len(native_stable_cases),
                    "native_unstable_case_count": len(case_rows) - len(native_stable_cases),
                    "per_fixture_native_stable_streams": per_fixture_native_stable,
                    "native_coverage_gate": native_coverage_gate,
                    "migration_only_excess_failure_count": len(migration_excess),
                    "migration_only_excess_failure_cases": [
                        {
                            "fixture": r["fixture"],
                            "future_stream": r["future_stream"],
                        }
                        for r in migration_excess
                    ],
                    "selector_depth_rescue_count": len(rescues),
                    "selector_depth_rescue_cases": [
                        {
                            "fixture": r["fixture"],
                            "future_stream": r["future_stream"],
                            "selected_anchor_seg": r["initial"]["selected_anchor_seg"],
                            "older_w2_anchor_seg": r["initial"]["older_w2_anchor_seg"],
                        }
                        for r in rescues
                    ],
                    "migration_excess_gate": migration_excess_gate,
                    "formal_g7_v2_pass_candidate": suite_pass_candidate,
                },
                "cases": case_rows,
                "interpretation_rule": {
                    "migration_problem": (
                        "native_coverage_gate=true and migration_only_excess_failure_count>0"
                    ),
                    "selector_pruning_problem": (
                        "migration-only excess failures are rescued by the older W2 anchor"
                    ),
                    "backbone_problem": (
                        "native_coverage_gate=false or native instability is widespread"
                    ),
                    "clean_pass": (
                        "all immediate chain fixtures qualify, native coverage gate passes, "
                        "and migration-only excess failure count is zero"
                    ),
                },
                "claim_boundary": (
                    "Synthetic cause-isolation experiment on RWKV-7 0.4B. "
                    "Native-validity classification is preregistered by rule and does not "
                    "inspect migrated outcomes. It distinguishes failure source; it is not "
                    "a general identity or production long-horizon proof."
                ),
            }
            write_report(report)

    except BaseException as exc:
        write_report({
            "status": "VELA_G7_RWKV7_0P4B_CAUSE_ISOLATION_V2_ERROR",
            "source_commit": os.environ.get("GITHUB_SHA"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-80:],
        })
        raise


if __name__ == "__main__":
    run()
