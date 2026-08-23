from __future__ import annotations

import importlib.util
import json
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
V1_PATH = BASE / "sequential-upgrade-v1" / "mamba_w1_w2_w3_chain.py"
spec = importlib.util.spec_from_file_location("vela_seq_v1", V1_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {V1_PATH}")
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
sel, dep, v3, v4 = v1.sel, v1.dep, v1.v3, v1.v4

# Keep migration/replay inference-only. Training still uses the original grad-enabled path.
_orig_run_slice = dep.run_slice
def _safe_run_slice(model, ids, start, end, cache):
    with torch.no_grad():
        return _orig_run_slice(model, ids, start, end, cache)
dep.run_slice = _safe_run_slice

PREFIX_FX = next(fx for fx in sel.FIXTURES if fx["id"] == "single_early")
TARGET_KEY = ("color", "A")
DISTRACTOR_COUNTS = [0, 2, 4]
EPS = 1e-8

NEUTRAL = [
    "Telemetry packet 101 was archived without changing any tracked value. ",
    "A neutral audit marker was recorded and no current value was replaced. ",
    "Background sensor metadata was retained without changing the active task. ",
    "An unrelated timestamp was logged and no decision variable changed. ",
]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def clone_cache(x):
    return None if x is None else v4.clone_cache(x)


def full_lineage(model, ids, positions):
    out = {}
    cur = None
    last = 0
    for p in sorted(set(positions)):
        if p < last:
            raise ValueError("positions must be monotonic")
        if p > last:
            cur = dep.run_slice(model, ids, last, p, cur)
        out[p] = clone_cache(cur)
        last = p
    return out


def row_map(ev):
    return {(r["pair"], r["variant"]): r for r in ev["rows"]}


def target_spec():
    for held in v3.HELDOUT:
        if (held["pair"], held["variant"]) == TARGET_KEY:
            hist, query = held["prompt"].rsplit("\n", 1)
            if not hist.endswith(" "):
                hist += " "
            probe = ("w3_only_color_correction", "\n" + query, list(held["candidates"]), held["expected"])
            return held, hist, probe
    raise RuntimeError(f"missing heldout target {TARGET_KEY}")


def extend_w2_lineage(model, ids, starts, ends, lineage, origins, first_suffix_seg, current):
    out_lineage = v1.clone_cache_map(lineage)
    out_origins = dict(origins)
    cur = clone_cache(current)
    for i in range(first_suffix_seg, len(starts)):
        s, e = starts[i], ends[i]
        out_lineage[s] = clone_cache(cur)
        out_origins[s] = "W2"
        cur = dep.run_slice(model, ids, s, e, cur)
        out_lineage[e] = clone_cache(cur)
        out_origins[e] = "W2"
    return out_lineage, out_origins, clone_cache(cur)


def expected_accuracy(rows):
    return sum(int(r["correct"]) for r in rows) / max(len(rows), 1)


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        torch.manual_seed(v3.SEED)
        random.seed(v3.SEED)
        tok = AutoTokenizer.from_pretrained(v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(v3.MODEL_ID, torch_dtype=torch.float32).cpu().eval()

        target, target_history, target_probe = target_spec()
        specs = [target_probe]

        prefix_segments = list(PREFIX_FX["segments"])
        prefix_text = "".join(prefix_segments)
        prefix_ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids
        prefix_T = int(prefix_ids.shape[1])
        prefix_starts, prefix_ends = dep.boundaries(tok, prefix_segments)
        prefix_positions = sorted(set(prefix_starts + prefix_ends + [0, prefix_T]))
        w1_prefix_lineage = full_lineage(model, prefix_ids, prefix_positions)
        w1_prefix_origins = {p: "W1" for p in prefix_positions}

        w1_eval = v3.evaluate(model, tok)
        w1_weights = v1.save_xproj(model)
        losses, snapshots, evals = v1.train_with_generation_snapshots(model, tok)
        w2_weights = snapshots[v1.W2_EPOCH]
        w3_weights = snapshots[v1.W3_EPOCH]
        w2_eval = evals[v1.W2_EPOCH]
        w3_eval = evals[v1.W3_EPOCH]
        w2_target_row = row_map(w2_eval)[TARGET_KEY]
        w3_target_row = row_map(w3_eval)[TARGET_KEY]

        # Fixture definition is fixed before any anchor scoring. This target is required to be
        # wrong in W2 and correct in W3 on the original held-out prompt.
        target_precondition = bool((not w2_target_row["correct"]) and w3_target_row["correct"])

        v1.load_xproj(model, w2_weights)
        with torch.no_grad():
            w2_native_prefix = clone_cache(model(prefix_ids, use_cache=True, return_dict=True).cache_params)
        prefix_specs = PREFIX_FX["probes"]
        hop1 = v1.target_free_select(
            model, tok, prefix_ids, prefix_starts, prefix_ends,
            v1.clone_cache_map(w1_prefix_lineage), prefix_specs, prefix_segments,
        )
        w2_native_prefix_rows = v1.score_rows(model, tok, w2_native_prefix, prefix_T, prefix_specs)
        hop1_comp = v1.compare_rows(hop1["rows"], w2_native_prefix_rows)
        w2_prefix_lineage, w2_prefix_origins, w2_prefix_current = v1.rebuild_lineage(
            model, prefix_ids, prefix_starts, prefix_ends,
            v1.clone_cache_map(w1_prefix_lineage), dict(w1_prefix_origins),
            hop1["chosen_seg"], "W2",
        )
        hop1_state_error = v3.cache_distance(w2_prefix_current, w2_native_prefix)

        rows = []
        for count in DISTRACTOR_COUNTS:
            # The W3-only correction is always the final event, so W3-native validity is tested
            # without post-correction distractor interference. Only the amount of prior W2-era
            # neutral history changes across fixtures.
            suffix = list(NEUTRAL[:count]) + [target_history]
            segments = prefix_segments + suffix
            ids = tok("".join(segments), return_tensors="pt", add_special_tokens=False).input_ids
            T = int(ids.shape[1])
            starts, ends = dep.boundaries(tok, segments)
            first_suffix_seg = len(prefix_segments)
            target_seg = len(segments) - 1
            assert starts[first_suffix_seg] == prefix_T
            assert target_seg >= first_suffix_seg
            assert torch.equal(ids[:, :prefix_T], prefix_ids)

            v1.load_xproj(model, w2_weights)
            w2_lineage, w2_origins, w2_carried_current = extend_w2_lineage(
                model, ids, starts, ends,
                w2_prefix_lineage, w2_prefix_origins,
                first_suffix_seg, w2_prefix_current,
            )
            with torch.no_grad():
                w2_native_full = clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            carried_error = v3.cache_distance(w2_carried_current, w2_native_full)

            v1.load_xproj(model, w3_weights)
            with torch.no_grad():
                w3_native = clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            w3_native_rows = v1.score_rows(model, tok, w3_native, T, specs)
            native_acc = expected_accuracy(w3_native_rows)

            # Critical rule: native validity is measured before selector evaluation; selector
            # output is never used to include/exclude fixtures.
            hop2 = v1.target_free_select(model, tok, ids, starts, ends, w2_lineage, specs, segments)
            hop2_comp = v1.compare_rows(hop2["rows"], w3_native_rows)
            hop2_origin = w2_origins[starts[hop2["chosen_seg"]]]

            _, w3_origins, w3_current = v1.rebuild_lineage(
                model, ids, starts, ends, w2_lineage, w2_origins,
                hop2["chosen_seg"], "W3",
            )
            chain_rows = v1.score_rows(model, tok, w3_current, T, specs)
            chain_comp = v1.compare_rows(chain_rows, w3_native_rows)

            v1.load_xproj(model, w1_weights)
            w1_full_lineage = full_lineage(model, ids, sorted(set(starts + ends + [0, T])))
            v1.load_xproj(model, w3_weights)
            direct = v1.target_free_select(model, tok, ids, starts, ends, w1_full_lineage, specs, segments)
            direct_comp = v1.compare_rows(direct["rows"], w3_native_rows)

            fixture_pass = bool(
                target_precondition
                and hop1_comp["decision_agreement"] == 1.0
                and hop1_state_error["rms"] > EPS
                and carried_error["rms"] > EPS
                and native_acc == 1.0
                and hop2_origin == "W2"
                and hop2["anchor_pos"] >= prefix_T
                and hop2_comp["decision_agreement"] == 1.0
                and chain_comp["decision_agreement"] == 1.0
            )
            rows.append({
                "fixture": f"color_final_after_{count}_w2_neutral_events",
                "history_tokens": T,
                "w2_era_start_pos": prefix_T,
                "target_event_index": target_seg,
                "target_event_anchor_pos": starts[target_seg],
                "fixed_target": {
                    "pair": target["pair"],
                    "variant": target["variant"],
                    "expected": target["expected"],
                    "w2_heldout_correct": w2_target_row["correct"],
                    "w3_heldout_correct": w3_target_row["correct"],
                },
                "hop1_w1_to_w2_prefix": {
                    "selected_anchor_event": hop1["chosen_seg"],
                    "anchor_pos": hop1["anchor_pos"],
                    "vs_w2_native": hop1_comp,
                    "state_error_vs_w2_native_prefix": hop1_state_error,
                },
                "w2_carried_before_w3": {
                    "state_error_vs_w2_native_full": carried_error,
                    "origin_at_w2_era_start": w2_origins[prefix_T],
                    "origin_at_final_before_w3": w2_origins[T],
                },
                "w3_native_probe_rows": w3_native_rows,
                "w3_native_expected_accuracy": native_acc,
                "hop2_actual_chained_lineage_to_w3": {
                    "selected_anchor_event": hop2["chosen_seg"],
                    "anchor_pos": hop2["anchor_pos"],
                    "anchor_source_generation": hop2_origin,
                    "anchor_is_in_w2_era": bool(hop2["anchor_pos"] >= prefix_T),
                    "selected_segments": hop2["selected_segments"],
                    "prune_trace": hop2["prune_trace"],
                    "vs_w3_native": hop2_comp,
                },
                "final_chain_w3": {
                    "functional_vs_w3_native": chain_comp,
                    "state_error_vs_w3_native": v3.cache_distance(w3_current, w3_native),
                    "lineage_origin_at_final": w3_origins[T],
                },
                "direct_w1_to_w3_control": {
                    "selected_anchor_event": direct["chosen_seg"],
                    "anchor_pos": direct["anchor_pos"],
                    "vs_w3_native": direct_comp,
                    "chain_vs_direct_decision_agreement": v1.agreement(chain_rows, direct["rows"]),
                },
                "fixture_pass": fixture_pass,
            })

        strict_gain = bool(w2_eval["accuracy"] > w1_eval["accuracy"] and w3_eval["accuracy"] > w2_eval["accuracy"])
        native_valid_count = sum(int(r["w3_native_expected_accuracy"] == 1.0) for r in rows)
        w2_source_count = sum(int(r["hop2_actual_chained_lineage_to_w3"]["anchor_source_generation"] == "W2") for r in rows)
        passed_count = sum(int(r["fixture_pass"]) for r in rows)
        suite_pass = bool(
            target_precondition
            and strict_gain
            and len(rows) == 3
            and native_valid_count == 3
            and w2_source_count == 3
            and passed_count == 3
        )

        report = {
            "status": "VELA_SEQUENTIAL_W1_W2_W3_CHAINED_SOURCE_CLEAN_V3",
            "model": v3.MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "generations": {
                "W1": {"training_epoch": 0, "accuracy": w1_eval["accuracy"]},
                "W2": {"training_epoch": v1.W2_EPOCH, "accuracy": w2_eval["accuracy"]},
                "W3": {"training_epoch": v1.W3_EPOCH, "accuracy": w3_eval["accuracy"]},
                "epoch_mean_loss": losses,
            },
            "fixture_design": {
                "selection_rule": "Fixed before anchor scoring: one W2-wrong/W3-correct held-out color correction, placed as the final W2-era event after 0/2/4 predetermined neutral events. No selector result is used to choose or exclude fixtures.",
                "target_precondition": target_precondition,
                "target_key": list(TARGET_KEY),
                "distractor_counts": DISTRACTOR_COUNTS,
            },
            "gate": {
                "strict_two_step_capability_gain": strict_gain,
                "fixture_count": len(rows),
                "w3_native_valid_fixture_count": native_valid_count,
                "w2_origin_hop2_anchor_count": w2_source_count,
                "fixture_pass_count": passed_count,
                "suite_pass": suite_pass,
            },
            "fixtures": rows,
            "success_definition": "All three predeclared clean fixtures must have W3-native expected accuracy 1/1, carry a nonzero W1->W2 migration-state difference into the W2 era, and the unchanged target-free selector must naturally choose a W2-origin hop2 checkpoint and reach 100% W3-native decision agreement. Direct W1->W3 remains control only.",
            "claim_boundary": "Three synthetic Mamba-130M chained-source fixtures using one fixed W3-only color correction under three predetermined W2-era neutral-history lengths. No claim of generic long-history robustness, final-backbone fitness, or identity continuity.",
        }
        write_report(report)
        if not suite_pass:
            raise SystemExit(3)
    except SystemExit:
        raise
    except BaseException as exc:
        write_report({
            "status": "VELA_SEQUENTIAL_W1_W2_W3_CHAINED_SOURCE_CLEAN_V3_ERROR",
            "model": getattr(v3, "MODEL_ID", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-45:],
        })
        raise


if __name__ == "__main__":
    run()
