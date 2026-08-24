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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# This imports the audited chain-v2 provenance policy. The workflow additionally
# verifies the relevant source blobs before execution so G7 cannot silently tune
# the selector after observing long-horizon outcomes.
v2 = load_module(
    "vela_rwkv7_chain_v2_for_g7",
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


def build_neutral_future(tok, token_count: int) -> tuple[list[int], dict]:
    """Build a deterministic future before inspecting any trajectory result.

    The text deliberately avoids project names, codewords, verification status,
    action status, and correction language used by the fixtures. It is intended
    to extend the recurrent trajectory without semantically overwriting the live
    facts being probed.
    """

    chunks = []
    i = 0
    ids: list[int] = []
    while len(ids) < token_count:
        chunk = (
            f"Neutral telemetry packet {i:04d} was archived. "
            f"Sensor checksum marker {i % 97:02d} was logged. "
        )
        chunks.append(chunk)
        ids = tok.encode("".join(chunks))
        i += 1
    return ids[:token_count], {
        "generator": "indexed neutral telemetry/checksum sentences",
        "chunks_generated": i,
        "tokens_used": token_count,
        "semantic_exclusion": "no fixture project/codeword/verification/action/correction terms",
    }


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


def expected_accuracy(rows):
    return sum(int(row["correct"]) for row in rows) / max(len(rows), 1)


def trace_horizon_summary(bits: list[bool], horizon: int) -> dict:
    part = bits[:horizon]
    agreed = sum(part)
    return {
        "steps": horizon,
        "decision_agreement_count": agreed,
        "decision_agreement": agreed / horizon,
        "divergence_count": horizon - agreed,
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
            future_ids, future_meta = build_neutral_future(tok, MAX_HORIZON)

            # Capture W1 event-boundary states before training, exactly as chain-v2.
            prepared = []
            for fx in chain.FIXTURES:
                ids = tok.encode("".join(fx["segments"]))
                starts, ends = chain.ca.boundaries(tok, fx["segments"])
                positions = sorted(set(starts + ends + [0, len(ids)]))
                states = {p: chain.ca.state_at(model, ns, ids[:p], args) for p in positions}
                prepared.append({
                    "fx": fx,
                    "ids": ids,
                    "starts": starts,
                    "ends": ends,
                    "w1_lineage": states,
                    "w1_origins": {p: "W1" for p in positions},
                })

            trainable = chain.scale.configure_kvr(model, args)

            # Recreate chain-v2 generations without adding any G7-driven tuning.
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

            fixture_results = []
            initial_qualified = 0
            minimum_pass_count = 0

            for item in prepared:
                fx = item["fx"]
                ids = item["ids"]
                starts = item["starts"]
                ends = item["ends"]
                specs = fx["probes"]
                w1_lineage = chain.clone_state_map(item["w1_lineage"])
                w1_origins = dict(item["w1_origins"])

                # Hop 1: actual W1 lineage -> W2.
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

                # Hop 2: actual carried W1/W2 lineage -> W3 using chain-v2 selector.
                chain.load_kvr(model, w3_weights)
                w3_native = chain.ca.state_at(model, ns, ids, args)
                w3_native_rows = chain.ca.score_specs(model, ns, tok, w3_native, specs)
                hop2 = chain.target_free_select(
                    model, ns, tok, ids, starts, ends, w2_lineage, specs, fx["segments"]
                )
                hop2_comp = chain.compare(hop2["rows"], w3_native_rows)
                anchor_origin = w2_origins[starts[hop2["chosen_seg"]]]
                _, w3_origins, w3_current = chain.rebuild_lineage(
                    model,
                    ns,
                    ids,
                    starts,
                    ends,
                    w2_lineage,
                    w2_origins,
                    hop2["chosen_seg"],
                    "W3",
                )
                w3_current_rows = chain.ca.score_specs(model, ns, tok, w3_current, specs)
                initial_final_comp = chain.compare(w3_current_rows, w3_native_rows)

                initial_ok = (
                    expected_accuracy(w3_native_rows) == 1.0
                    and hop1_comp["decision_agreement"] == 1.0
                    and hop2_comp["decision_agreement"] == 1.0
                    and initial_final_comp["decision_agreement"] == 1.0
                    and anchor_origin == "W2"
                )
                initial_qualified += int(initial_ok)

                # Roll both W3-native and migrated carried states through the same
                # predeclared future tokens. Per-step decision = next-token argmax
                # after consuming that future token. This is a compact continuous
                # trajectory trace; task-specific probes are separately measured at
                # the three preregistered horizons.
                native_state = chain.clone_state(w3_native)
                migrated_state = chain.clone_state(w3_current)
                decision_bits: list[bool] = []
                first_divergence = None
                disagreement_examples = []
                milestone_rows = {}

                for step, tid in enumerate(future_ids, start=1):
                    with torch.no_grad():
                        native_logits, native_state = chain.rw.run_tokens(
                            model, ns, [tid], native_state
                        )
                        migrated_logits, migrated_state = chain.rw.run_tokens(
                            model, ns, [tid], migrated_state
                        )
                    native_choice = int(torch.argmax(native_logits).item())
                    migrated_choice = int(torch.argmax(migrated_logits).item())
                    agree = native_choice == migrated_choice
                    decision_bits.append(agree)
                    if not agree and first_divergence is None:
                        first_divergence = step
                    if not agree and len(disagreement_examples) < 32:
                        disagreement_examples.append({
                            "step": step,
                            "future_input_token": int(tid),
                            "native_next_token_argmax": native_choice,
                            "migrated_next_token_argmax": migrated_choice,
                        })

                    if step in HORIZONS:
                        native_probe = chain.ca.score_specs(
                            model, ns, tok, native_state, specs
                        )
                        migrated_probe = chain.ca.score_specs(
                            model, ns, tok, migrated_state, specs
                        )
                        functional_comp = chain.compare(migrated_probe, native_probe)
                        native_flags = functional_flags(native_probe)
                        migrated_flags = functional_flags(migrated_probe)
                        milestone_rows[str(step)] = {
                            "functional_vs_native": functional_comp,
                            "native_expected_accuracy": expected_accuracy(native_probe),
                            "migrated_expected_accuracy": expected_accuracy(migrated_probe),
                            "native_flags": native_flags,
                            "migrated_flags": migrated_flags,
                            "native_probe_rows": native_probe,
                            "migrated_probe_rows": migrated_probe,
                        }

                trace_summaries = {
                    str(h): trace_horizon_summary(decision_bits, h) for h in HORIZONS
                }
                first_functional_divergence_horizon = next(
                    (
                        h
                        for h in HORIZONS
                        if milestone_rows[str(h)]["functional_vs_native"]["decision_agreement"] < 1.0
                    ),
                    None,
                )

                # Formal minimum gate candidate intentionally does not require raw
                # token-argmax identity. It requires native and migrated task
                # invariants plus task-specific functional agreement at every
                # preregistered horizon. Raw per-token divergence is reported in full
                # for audit rather than hidden or tuned around.
                horizon_minimum_ok = all(
                    milestone_rows[str(h)]["native_flags"]["task_invariant_ok"]
                    and milestone_rows[str(h)]["migrated_flags"]["task_invariant_ok"]
                    and milestone_rows[str(h)]["functional_vs_native"]["decision_agreement"] == 1.0
                    for h in HORIZONS
                )
                fixture_minimum_pass = initial_ok and horizon_minimum_ok
                minimum_pass_count += int(fixture_minimum_pass)

                fixture_results.append({
                    "fixture": fx["id"],
                    "initial_chain_qualification": {
                        "w3_native_valid": expected_accuracy(w3_native_rows) == 1.0,
                        "hop1_decision_agreement": hop1_comp["decision_agreement"],
                        "hop2_decision_agreement": hop2_comp["decision_agreement"],
                        "final_chain_decision_agreement": initial_final_comp["decision_agreement"],
                        "hop2_anchor_source_generation": anchor_origin,
                        "hop2_anchor_event": hop2["chosen_seg"],
                        "hop2_anchor_pos": hop2["anchor_pos"],
                        "final_source_generation": w3_origins[len(ids)],
                        "qualified": initial_ok,
                    },
                    "trajectory": {
                        "steps": MAX_HORIZON,
                        "per_step_next_token_argmax_agreement_bits": "".join(
                            "1" if bit else "0" for bit in decision_bits
                        ),
                        "first_trace_divergence_step": first_divergence,
                        "trace_by_horizon": trace_summaries,
                        "disagreement_examples_first32": disagreement_examples,
                        "first_functional_divergence_horizon": first_functional_divergence_horizon,
                        "functional_milestones": milestone_rows,
                    },
                    "formal_minimum_pass_candidate": fixture_minimum_pass,
                })

            suite_minimum_pass = (
                initial_qualified == len(prepared)
                and minimum_pass_count == len(prepared)
            )
            report = {
                "status": "VELA_G7_RWKV7_0P4B_LONG_HORIZON_TRAJECTORY_V1",
                "source_commit": os.environ.get("GITHUB_SHA"),
                "device": "cpu",
                "dtype": "float32",
                "model": {
                    "weight_repo": chain.scale.WEIGHT_REPO,
                    "weight_file": chain.scale.WEIGHT_FILE,
                    "weight_revision": chain.scale.WEIGHT_REVISION,
                },
                "protocol": {
                    "chain_policy": "RWKV-7 0.4B chain-v2 provenance selector, source blobs workflow-pinned",
                    "horizons": list(HORIZONS),
                    "future": future_meta,
                    "per_step_decision_definition": "next-token argmax after each shared neutral future token",
                    "functional_probe_definition": "fixture codeword/verification/project/action probes at 128/512/2048",
                    "selector_oracle_usage": "none; native W3 trajectory is evaluation-only",
                    "w2_lr": chain.W2_LR,
                    "w3_lr": chain.W3_LR,
                },
                "training": {
                    "w2_mean_loss": w2_loss,
                    "w3_mean_loss": w3_loss,
                },
                "summary": {
                    "fixture_count": len(prepared),
                    "initial_chain_qualified": initial_qualified,
                    "formal_minimum_pass_candidate_count": minimum_pass_count,
                    "formal_minimum_suite_pass_candidate": suite_minimum_pass,
                },
                "fixtures": fixture_results,
                "claim_boundary": (
                    "Formal G7 evidence candidate on four predeclared RWKV-7 0.4B chain fixtures and a deterministic neutral future. "
                    "Per-step argmax trace is diagnostic/trajectory evidence; the minimum gate candidate uses native+migrated task invariants and task-specific functional agreement at preregistered horizons. "
                    "Gate G7 remains PENDING until source/workflow/result are independently audited."
                ),
            }
            write_report(report)
    except Exception as exc:
        write_report({
            "status": "VELA_G7_RWKV7_0P4B_LONG_HORIZON_TRAJECTORY_V1_ERROR",
            "source_commit": os.environ.get("GITHUB_SHA"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    run()
