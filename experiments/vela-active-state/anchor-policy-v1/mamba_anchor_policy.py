from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
CONCLUSION = BASE / "conclusion-round-v1" / "mamba_conclusion_round.py"
spec = importlib.util.spec_from_file_location("vela_conclusion", CONCLUSION)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {CONCLUSION}")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

MODEL_ID = c.MODEL_ID
SEED = 917

HISTORY_SEGMENTS = {
    "main": [
        "Project Orion remains active. ",
        "The old codeword was ALPHA. ",
        "Correction: the current codeword is BETA, not ALPHA. ",
        "The verification step is incomplete, so external action remains blocked until verification succeeds. ",
        "Hypothesis one is weakened by the latest evidence while hypothesis two remains unresolved.",
    ],
    "correction_early": [
        "The old codeword was ALPHA. ",
        "Correction: the current codeword is BETA, not ALPHA. ",
        "Project Orion remains active. ",
        "Verification is incomplete. ",
        "Several neutral observations are logged. ",
        "No external action is authorized. ",
        "Hypothesis two remains unresolved. ",
        "More neutral telemetry is recorded.",
    ],
    "correction_late": [
        "Project Orion remains active. ",
        "Verification is incomplete. ",
        "Several neutral observations are logged. ",
        "No external action is authorized. ",
        "Hypothesis two remains unresolved. ",
        "More neutral telemetry is recorded. ",
        "The old codeword was ALPHA. ",
        "Correction: the current codeword is BETA, not ALPHA.",
    ],
}
CORRECTION_INDEX = {"main": 2, "correction_early": 1, "correction_late": 7}
GRID_SPACING = [4, 8, 16, 32]
CODE_PROBE = [{"id":"codeword","suffix":"\nCurrent codeword:","candidates":[" BETA"," ALPHA"],"expected":" BETA"}]


def write_report(report):
    path = os.environ.get("VELA_RESULT_PATH")
    if path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def tokenized_segments(tok, segments):
    full = "".join(segments)
    full_ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids
    boundaries = [0]
    # Prefix tokenization is used for boundaries so BPE boundary effects cannot desync positions.
    prefix = ""
    for seg in segments:
        prefix += seg
        boundaries.append(int(tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]))
    return full, full_ids, boundaries


def build_w1_snapshots(model, tok):
    out = {}
    model.eval()
    with torch.no_grad():
        for name, segments in HISTORY_SEGMENTS.items():
            text, ids, boundaries = tokenized_segments(tok, segments)
            T = int(ids.shape[1])
            wanted = {0, T, *boundaries}
            for spacing in GRID_SPACING:
                wanted.update(range(0, T + 1, spacing))
                wanted.add(T)
            snaps = {0: None}
            # Prefix calls are slower than one recurrent pass but make checkpoint semantics explicit.
            for pos in sorted(p for p in wanted if 0 < p <= T):
                snaps[pos] = c.clone_cache(model(ids[:, :pos], use_cache=True, return_dict=True).cache_params)
            out[name] = {
                "text": text,
                "ids": ids,
                "T": T,
                "boundaries": boundaries,
                "snapshots": snaps,
                "full": c.clone_cache(snaps[T]),
            }
    return out


def replay_from(model, item, checkpoint_pos):
    ids = item["ids"]; T = item["T"]
    if checkpoint_pos == 0:
        with torch.no_grad():
            return c.clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
    state = c.clone_cache(item["snapshots"][checkpoint_pos])
    if checkpoint_pos < T:
        with torch.no_grad():
            state, _ = c.run_tokens(model, ids[:, checkpoint_pos:], state, checkpoint_pos)
    return c.clone_cache(state)


def nearest_periodic_before(pos, spacing):
    return (pos // spacing) * spacing


def event_transition_drift(model, item, event_index):
    start = item["boundaries"][event_index]
    end = item["boundaries"][event_index + 1]
    pre = c.clone_cache(item["snapshots"][start])
    w1_post = c.clone_cache(item["snapshots"][end])
    if start == 0:
        with torch.no_grad():
            w2_post = c.clone_cache(model(item["ids"][:, :end], use_cache=True, return_dict=True).cache_params)
    else:
        with torch.no_grad():
            w2_post, _ = c.run_tokens(model, item["ids"][:, start:end], pre, start)
    dist = c.cache_distance(w2_post, w1_post)
    return {"event_index": event_index, "start": start, "end": end, "tokens": end-start, "transition_drift": dist}


def functional_eval(model, tok, state, T, native, probes):
    rows = c.probe_state(model, tok, state, T, probes)
    native_rows = c.probe_state(model, tok, native, T, probes)
    return {"comparison": c.compare_probe(rows, native_rows), "rows": rows, "native_rows": native_rows}


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        torch.manual_seed(SEED); random.seed(SEED)
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).cpu().eval()

        w1 = build_w1_snapshots(model, tok)
        main = w1["main"]
        baseline = c.v3.evaluate(model, tok)
        training = c.train_model(model, tok, SEED, main["full"], main["ids"])
        after = c.v3.evaluate(model, tok)

        # E13: periodic checkpoint sparsity around the known causal correction event.
        main_corr_start = main["boundaries"][CORRECTION_INDEX["main"]]
        with torch.no_grad():
            native_main = c.clone_cache(model(main["ids"], use_cache=True, return_dict=True).cache_params)
        native_probe = c.probe_state(model, tok, native_main, main["T"])
        sparsity = []
        for spacing in GRID_SPACING:
            cp = nearest_periodic_before(main_corr_start, spacing)
            # If exact periodic point was not materialized due T clipping, use nearest stored <= cp.
            cp = max(p for p in main["snapshots"] if p <= cp)
            migrated = replay_from(model, main, cp)
            probe = c.probe_state(model, tok, migrated, main["T"])
            sparsity.append({
                "spacing_tokens": spacing,
                "checkpoint_count": int(math.ceil(main["T"] / spacing)) + 1,
                "correction_start": main_corr_start,
                "selected_checkpoint": cp,
                "replayed_tokens": main["T"] - cp,
                "replay_fraction": (main["T"] - cp) / max(main["T"], 1),
                "state_error_vs_w2_native": c.cache_distance(migrated, native_main),
                "functional_vs_w2_native": c.compare_probe(probe, native_probe),
            })
        semantic_cp = main_corr_start
        semantic_migrated = replay_from(model, main, semantic_cp)
        semantic_probe = c.probe_state(model, tok, semantic_migrated, main["T"])
        semantic_anchor = {
            "selected_checkpoint": semantic_cp,
            "replayed_tokens": main["T"] - semantic_cp,
            "replay_fraction": (main["T"] - semantic_cp) / max(main["T"], 1),
            "state_error_vs_w2_native": c.cache_distance(semantic_migrated, native_main),
            "functional_vs_w2_native": c.compare_probe(semantic_probe, native_probe),
        }

        # E14: automatic causal-event detector. No W2-native full history is used to rank events.
        detector_histories = []
        hits = 0
        for name, item in w1.items():
            drifts = [event_transition_drift(model, item, i) for i in range(len(HISTORY_SEGMENTS[name]))]
            ranked = sorted(drifts, key=lambda r: r["transition_drift"]["rms"], reverse=True)
            selected = ranked[0]
            expected_idx = CORRECTION_INDEX[name]
            hit = selected["event_index"] == expected_idx
            hits += int(hit)
            with torch.no_grad():
                native = c.clone_cache(model(item["ids"], use_cache=True, return_dict=True).cache_params)
            migrated = replay_from(model, item, selected["start"])
            probes = c.HARD_PROBES if name == "main" else CODE_PROBE
            fe = functional_eval(model, tok, migrated, item["T"], native, probes)
            # Oracle correction anchor only for evaluation, not detector selection.
            oracle_start = item["boundaries"][expected_idx]
            oracle = replay_from(model, item, oracle_start)
            oracle_fe = functional_eval(model, tok, oracle, item["T"], native, probes)
            detector_histories.append({
                "history": name,
                "tokens": item["T"],
                "expected_causal_event_index": expected_idx,
                "selected_event_index": selected["event_index"],
                "detector_hit": hit,
                "selected_replay_fraction": (item["T"] - selected["start"]) / max(item["T"], 1),
                "ranked_events": ranked,
                "selected_migration": {
                    "checkpoint": selected["start"],
                    "state_error_vs_w2_native": c.cache_distance(migrated, native),
                    "functional_vs_w2_native": fe["comparison"],
                },
                "oracle_correction_anchor": {
                    "checkpoint": oracle_start,
                    "state_error_vs_w2_native": c.cache_distance(oracle, native),
                    "functional_vs_w2_native": oracle_fe["comparison"],
                },
            })

        report = {
            "status": "VELA_CAUSAL_ANCHOR_POLICY_V1",
            "model": MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "capability": {
                "baseline_accuracy": baseline["accuracy"],
                "after_accuracy": after["accuracy"],
                "baseline_correction": baseline["correction_accuracy"],
                "after_correction": after["correction_accuracy"],
                "baseline_control": baseline["control_accuracy"],
                "after_control": after["control_accuracy"],
                "training": training,
            },
            "E13_periodic_anchor_sparsity": {
                "history_tokens": main["T"],
                "correction_start_token": main_corr_start,
                "periodic_rows": sparsity,
                "semantic_event_anchor": semantic_anchor,
            },
            "E14_local_transition_drift_detector": {
                "detector": "rank events by W1-post vs W2-post state RMS when both start from the same stored W1 pre-event checkpoint",
                "uses_w2_native_for_ranking": False,
                "hit_rate": hits / len(detector_histories),
                "histories": detector_histories,
            },
            "success_definition": "A useful anchor policy should preserve W2 capability, recover W2-native functional decisions with substantially less than full-history replay, and identify replay-relevant causal events without using the W2-native full-history target for selection.",
            "claim_boundary": "Narrow synthetic Mamba-1 130M correction curriculum. Detector and checkpoint density are engineering probes, not a general causal-discovery theorem or identity proof. No architecture promotion is implied.",
        }
        write_report(report)
    except BaseException as exc:
        write_report({
            "status": "VELA_CAUSAL_ANCHOR_POLICY_V1_ERROR",
            "model": MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-50:],
            "claim_boundary": "Harness/runtime failure only; no architecture verdict.",
        })
        raise


if __name__ == "__main__":
    run()
