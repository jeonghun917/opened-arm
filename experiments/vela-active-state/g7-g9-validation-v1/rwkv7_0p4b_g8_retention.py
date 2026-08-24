from __future__ import annotations

import importlib.util
import io
import json
import os
import random
import statistics
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
POLICIES = (
    "every_event",
    "fixed_n2",
    "fixed_n4",
    "fixed_n8",
    "fixed_n16",
    "semantic_only",
    "hybrid_semantic_maxgap8",
)
SEMANTIC_MARKERS = (
    "Project ",
    "old codeword",
    "Correction:",
    "New correction:",
    "Verification ",
    "External action ",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v2 = load_module(
    "vela_rwkv7_chain_v2_for_g8",
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


def is_semantic_event(text: str) -> bool:
    return any(marker in text for marker in SEMANTIC_MARKERS)


def retain_boundary(policy: str, *, since_last: int, semantic: bool) -> bool:
    if policy == "every_event":
        return True
    if policy.startswith("fixed_n"):
        n = int(policy.removeprefix("fixed_n"))
        return since_last >= n
    if policy == "semantic_only":
        return semantic
    if policy == "hybrid_semantic_maxgap8":
        return semantic or since_last >= 8
    raise ValueError(f"unknown policy {policy}")


def retained_positions(policy: str, *, fx, ends, origins) -> tuple[int, ...]:
    positions = []
    since_last = 0
    for i, end in enumerate(ends):
        # Only checkpoints actually rebuilt under W2 are candidates for the W2
        # retention policy. Historical W1 states may remain in lineage but are not
        # counted as new W2 retained checkpoints here.
        if origins.get(end) != "W2":
            continue
        since_last += 1
        semantic = is_semantic_event(fx["segments"][i])
        if retain_boundary(policy, since_last=since_last, semantic=semantic):
            positions.append(end)
            since_last = 0
    return tuple(sorted(set(positions)))


def replay_from_position(model, ns, ids, position, state):
    return chain.state_to_end(model, ns, ids, position, state)


def event_index_for_position(starts, position):
    for i, start in enumerate(starts):
        if start == position:
            return i
    return len(starts)


def nearest_rank(values, quantile: float):
    if not values:
        return None
    xs = sorted(values)
    rank = max(1, int((quantile * len(xs)) + 0.999999999))
    return xs[min(rank - 1, len(xs) - 1)]


def measure_state_io(state, rounds=3):
    writes = []
    reads = []
    sizes = []
    for _ in range(rounds):
        buf = io.BytesIO()
        t0 = time.perf_counter()
        torch.save(state, buf)
        writes.append((time.perf_counter() - t0) * 1000.0)
        data = buf.getvalue()
        sizes.append(len(data))
        t1 = time.perf_counter()
        torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        reads.append((time.perf_counter() - t1) * 1000.0)
    return {
        "serialized_bytes": int(statistics.median(sizes)),
        "write_ms_median": statistics.median(writes),
        "read_ms_median": statistics.median(reads),
        "rounds": rounds,
    }


def evaluate_anchor(model, ns, tok, ids, specs, w2_lineage, pos, w3_native_rows):
    if pos is None:
        state = chain.ca.state_at(model, ns, ids, model.args)
        mode = "full_replay"
    else:
        state = replay_from_position(model, ns, ids, pos, w2_lineage[pos])
        mode = "retained_checkpoint"
    rows = chain.ca.score_specs(model, ns, tok, state, specs)
    comp = chain.compare(rows, w3_native_rows)
    return state, rows, comp, mode


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

            policy_fixture_rows = {name: [] for name in POLICIES}
            representative_io = None

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
                w2_lineage, w2_origins, _ = chain.rebuild_lineage(
                    model, ns, ids, starts, ends, w1_lineage, w1_origins,
                    hop1["chosen_seg"], "W2"
                )

                if representative_io is None:
                    w2_positions = [p for p, origin in w2_origins.items() if origin == "W2"]
                    if w2_positions:
                        representative_io = measure_state_io(w2_lineage[w2_positions[-1]])

                chain.load_kvr(model, w3_weights)
                w3_native = chain.ca.state_at(model, ns, ids, args)
                w3_native_rows = chain.ca.score_specs(model, ns, tok, w3_native, specs)
                native_valid = all(row["correct"] for row in w3_native_rows)
                hop2 = chain.target_free_select(
                    model, ns, tok, ids, starts, ends, w2_lineage, specs, fx["segments"]
                )
                safe_pos = hop2["anchor_pos"]
                safe_origin = w2_origins.get(safe_pos)

                for policy in POLICIES:
                    retained = retained_positions(
                        policy, fx=fx, ends=ends, origins=w2_origins
                    )
                    usable = [p for p in retained if p <= safe_pos]
                    selected_pos = max(usable) if usable else None
                    state, rows, comp, mode = evaluate_anchor(
                        model, ns, tok, ids, specs, w2_lineage,
                        selected_pos, w3_native_rows
                    )
                    pre_fallback_ok = native_valid and comp["decision_agreement"] == 1.0
                    fallback_used = not pre_fallback_ok or selected_pos is None
                    if fallback_used:
                        _, fallback_rows, fallback_comp, _ = evaluate_anchor(
                            model, ns, tok, ids, specs, w2_lineage,
                            None, w3_native_rows
                        )
                        recovered = native_valid and fallback_comp["decision_agreement"] == 1.0
                    else:
                        fallback_rows = None
                        fallback_comp = None
                        recovered = True

                    # Missing/corrupt chosen-checkpoint injection: remove the chosen
                    # retained checkpoint, then use the previous safe retained W2
                    # checkpoint if one exists, otherwise full replay.
                    if selected_pos is not None:
                        after_removal = [p for p in usable if p != selected_pos]
                        recovery_pos = max(after_removal) if after_removal else None
                        _, _, recovery_comp, recovery_mode = evaluate_anchor(
                            model, ns, tok, ids, specs, w2_lineage,
                            recovery_pos, w3_native_rows
                        )
                        missing_corrupt_recovered = (
                            native_valid and recovery_comp["decision_agreement"] == 1.0
                        )
                    else:
                        recovery_pos = None
                        recovery_mode = "full_replay"
                        missing_corrupt_recovered = native_valid

                    if selected_pos is None:
                        replay_event_index = 0
                        replayed_tokens = T
                    else:
                        replay_event_index = event_index_for_position(starts, selected_pos)
                        replayed_tokens = T - selected_pos
                    replayed_events = len(starts) - replay_event_index

                    policy_fixture_rows[policy].append({
                        "fixture": fx["id"],
                        "native_valid": native_valid,
                        "chain_v2_safe_anchor_pos": safe_pos,
                        "chain_v2_safe_anchor_source_generation": safe_origin,
                        "retained_positions": list(retained),
                        "stored_checkpoint_count": len(retained),
                        "safe_anchor_available": selected_pos is not None,
                        "selected_retained_checkpoint_pos": selected_pos,
                        "selected_mode": mode,
                        "replayed_events": replayed_events,
                        "replayed_tokens": replayed_tokens,
                        "functional_success_before_fallback": pre_fallback_ok,
                        "full_replay_fallback_used": fallback_used,
                        "recovered_successfully": recovered,
                        "missing_or_corrupt_injection": {
                            "removed_checkpoint_pos": selected_pos,
                            "recovery_checkpoint_pos": recovery_pos,
                            "recovery_mode": recovery_mode,
                            "recovered_successfully": missing_corrupt_recovered,
                        },
                        "functional_vs_w3_native": comp,
                    })

            if representative_io is None:
                raise RuntimeError("no W2-origin checkpoint available for I/O benchmark")

            aggregate = {}
            for policy in POLICIES:
                rows = policy_fixture_rows[policy]
                counts = [row["stored_checkpoint_count"] for row in rows]
                replay_events = [row["replayed_events"] for row in rows]
                replay_tokens = [row["replayed_tokens"] for row in rows]
                state_bytes = representative_io["serialized_bytes"]
                total_checkpoints = sum(counts)
                aggregate[policy] = {
                    "fixture_count": len(rows),
                    "stored_checkpoint_count_total": total_checkpoints,
                    "estimated_storage_bytes_total": total_checkpoints * state_bytes,
                    "storage_bytes_per_fixture_event": (
                        total_checkpoints * state_bytes
                        / max(sum(len(item["starts"]) for item in prepared), 1)
                    ),
                    "safe_anchor_availability_rate": sum(
                        int(row["safe_anchor_available"]) for row in rows
                    ) / len(rows),
                    "functional_success_before_fallback_rate": sum(
                        int(row["functional_success_before_fallback"]) for row in rows
                    ) / len(rows),
                    "full_replay_fallback_rate": sum(
                        int(row["full_replay_fallback_used"]) for row in rows
                    ) / len(rows),
                    "recovery_success_rate": sum(
                        int(row["recovered_successfully"]) for row in rows
                    ) / len(rows),
                    "missing_or_corrupt_recovery_success_rate": sum(
                        int(row["missing_or_corrupt_injection"]["recovered_successfully"])
                        for row in rows
                    ) / len(rows),
                    "replayed_events": {
                        "p50": nearest_rank(replay_events, 0.50),
                        "p95": nearest_rank(replay_events, 0.95),
                        "worst": max(replay_events),
                    },
                    "replayed_tokens": {
                        "p50": nearest_rank(replay_tokens, 0.50),
                        "p95": nearest_rank(replay_tokens, 0.95),
                        "worst": max(replay_tokens),
                    },
                    "estimated_checkpoint_write_ms_total": (
                        total_checkpoints * representative_io["write_ms_median"]
                    ),
                    "estimated_checkpoint_read_ms_per_selected_migration": (
                        representative_io["read_ms_median"]
                    ),
                }

            every_storage = aggregate["every_event"]["estimated_storage_bytes_total"]
            for policy in POLICIES:
                aggregate[policy]["storage_amplification_vs_every_event"] = (
                    aggregate[policy]["estimated_storage_bytes_total"]
                    / max(every_storage, 1)
                )

            safety_qualified = [
                policy
                for policy in POLICIES
                if aggregate[policy]["recovery_success_rate"] == 1.0
                and aggregate[policy]["missing_or_corrupt_recovery_success_rate"] == 1.0
            ]

            report = {
                "status": "VELA_G8_RWKV7_0P4B_RETENTION_POLICY_V1",
                "source_commit": os.environ.get("GITHUB_SHA"),
                "device": "cpu",
                "model": {
                    "weight_repo": chain.scale.WEIGHT_REPO,
                    "weight_file": chain.scale.WEIGHT_FILE,
                    "weight_revision": chain.scale.WEIGHT_REVISION,
                },
                "protocol": {
                    "policies": list(POLICIES),
                    "semantic_markers_predeclared": list(SEMANTIC_MARKERS),
                    "hybrid_max_gap": 8,
                    "safe_anchor_rule": "latest retained W2-origin checkpoint at or before frozen chain-v2 selected anchor; else full replay",
                    "missing_corrupt_rule": "remove chosen retained checkpoint, use previous safe retained checkpoint or full replay",
                    "percentile_rule": "nearest-rank",
                    "selector_oracle_usage": "none; W3-native is evaluation-only",
                },
                "training": {
                    "w2_mean_loss": w2_loss,
                    "w3_mean_loss": w3_loss,
                },
                "checkpoint_io_microbenchmark": representative_io,
                "aggregate": aggregate,
                "safety_qualified_policies": safety_qualified,
                "fixtures_by_policy": policy_fixture_rows,
                "claim_boundary": (
                    "G8 policy evidence on four predeclared RWKV-7 0.4B chain fixtures. Storage uses measured serialized recurrent-state size and median local serialization I/O multiplied by policy write counts. "
                    "No checkpoint interval/policy is frozen by this result; Gate G8 remains PENDING until source/workflow/result audit and later integration-level I/O confirmation."
                ),
            }
            write_report(report)
    except Exception as exc:
        write_report({
            "status": "VELA_G8_RWKV7_0P4B_RETENTION_POLICY_V1_ERROR",
            "source_commit": os.environ.get("GITHUB_SHA"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    run()
