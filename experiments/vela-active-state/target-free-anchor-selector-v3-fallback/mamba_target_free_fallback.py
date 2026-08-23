from __future__ import annotations

import importlib.util
import json
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
STRESS_PATH = BASE / "target-free-anchor-selector-v2-stress" / "mamba_target_free_stress.py"
spec = importlib.util.spec_from_file_location("vela_target_free_stress_v2", STRESS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {STRESS_PATH}")
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)

dep, v2, v3, v4, fa = stress.dep, stress.v2, stress.v3, stress.v4, stress.fa


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def invariant_pass(rows):
    """Synthetic deploy-time validator: all hand-specified task invariants must hold."""
    return all(bool(r["correct"]) for r in rows)


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        torch.manual_seed(v3.SEED)
        random.seed(v3.SEED)
        tok = AutoTokenizer.from_pretrained(v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            v3.MODEL_ID, torch_dtype=torch.float32
        ).cpu().eval()

        prepared = []
        for fx in stress.FIXTURES:
            ids = tok("".join(fx["segments"]), return_tensors="pt", add_special_tokens=False).input_ids
            T = int(ids.shape[1])
            starts, ends = dep.boundaries(tok, fx["segments"])
            positions = sorted(set(starts + ends + [0, T]))
            caches = {p: dep.prefix_cache(model, ids, p) for p in positions}
            prepared.append({
                "fx": fx,
                "ids": ids,
                "T": T,
                "starts": starts,
                "ends": ends,
                "w1_caches": caches,
            })

        baseline = v3.evaluate(model, tok)
        losses = v2.train_upgrade(model, tok)
        after = v3.evaluate(model, tok)

        rows = []
        valid_count = 0
        eligible_count = 0
        detected_count = 0
        fallback_count = 0
        recovered_count = 0
        residual_unsafe_count = 0

        for item in prepared:
            fx, ids, T = item["fx"], item["ids"], item["T"]
            starts, caches = item["starts"], item["w1_caches"]
            specs = fx["probes"]

            with torch.no_grad():
                native = v4.clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            native_rows = fa.score_probe_specs(model, tok, native, T, specs)
            native_expected_accuracy = sum(int(r["correct"]) for r in native_rows) / len(native_rows)
            valid = native_expected_accuracy == 1.0
            valid_count += int(valid)

            if not valid:
                rows.append({
                    "fixture": fx["id"],
                    "fixture_valid": False,
                    "eligible_for_forced_miss": False,
                    "reason": "W2-native does not satisfy all hand-specified invariants",
                })
                continue

            # Evaluation-only oracle frontier. It is used only to construct a known-bad
            # injected selector miss; it is never available to the deployable selector.
            anchor_evals = []
            safe = []
            for i, s in enumerate(starts):
                migrated = dep.run_slice(model, ids, s, T, caches[s])
                probe_rows = fa.score_probe_specs(model, tok, migrated, T, specs)
                comp = fa.compare_rows(probe_rows, native_rows)
                inv_ok = invariant_pass(probe_rows)
                anchor_evals.append((i, s, probe_rows, comp, inv_ok))
                if comp["decision_agreement"] == 1.0:
                    safe.append((i, s))

            latest_safe = max(safe, key=lambda x: x[1]) if safe else None
            if latest_safe is None:
                rows.append({
                    "fixture": fx["id"],
                    "fixture_valid": True,
                    "eligible_for_forced_miss": False,
                    "reason": "no safe anchor frontier found",
                })
                continue

            # Inject the smallest possible unsafe-late-anchor error after the latest safe
            # frontier. This models a detector/selector that missed a still-live cause.
            unsafe_late = [
                x for x in anchor_evals
                if x[1] > latest_safe[1] and not x[4]
            ]
            if not unsafe_late:
                rows.append({
                    "fixture": fx["id"],
                    "fixture_valid": True,
                    "eligible_for_forced_miss": False,
                    "oracle_latest_safe_anchor": {
                        "event_index": latest_safe[0],
                        "anchor_pos": latest_safe[1],
                    },
                    "reason": "no later anchor violates the synthetic invariants",
                })
                continue

            eligible_count += 1
            bad = min(unsafe_late, key=lambda x: x[1])
            bad_i, bad_s, bad_rows, bad_comp, bad_inv_ok = bad

            validation_detected = not bad_inv_ok
            detected_count += int(validation_detected)

            fallback_attempted = validation_detected
            fallback_count += int(fallback_attempted)
            fallback_rows = None
            fallback_comp = None
            fallback_ok = False
            if fallback_attempted:
                # Full replay is the conservative recovery path: no old W1 cache is kept.
                fallback_state = dep.run_slice(model, ids, 0, T, None)
                fallback_rows = fa.score_probe_specs(model, tok, fallback_state, T, specs)
                fallback_ok = invariant_pass(fallback_rows)
                fallback_comp = fa.compare_rows(fallback_rows, native_rows)
                recovered_count += int(fallback_ok and fallback_comp["decision_agreement"] == 1.0)
                residual_unsafe_count += int(not fallback_ok)

            rows.append({
                "fixture": fx["id"],
                "fixture_valid": True,
                "eligible_for_forced_miss": True,
                "history_tokens": T,
                "oracle_latest_safe_anchor": {
                    "event_index": latest_safe[0],
                    "anchor_pos": latest_safe[1],
                    "replay_fraction": (T - latest_safe[1]) / max(T, 1),
                },
                "injected_unsafe_late_anchor": {
                    "event_index": bad_i,
                    "anchor_pos": bad_s,
                    "replay_fraction": (T - bad_s) / max(T, 1),
                    "vs_w2_native": bad_comp,
                    "invariant_pass": bad_inv_ok,
                },
                "validation_detected": validation_detected,
                "fallback_attempted": fallback_attempted,
                "fallback_full_replay": {
                    "invariant_pass": fallback_ok,
                    "vs_w2_native": fallback_comp,
                },
            })

        detection_rate = detected_count / max(eligible_count, 1)
        recovery_rate = recovered_count / max(fallback_count, 1)
        suite_pass = bool(
            eligible_count >= 3
            and detected_count == eligible_count
            and fallback_count == eligible_count
            and recovered_count == fallback_count
            and residual_unsafe_count == 0
        )

        write_report({
            "status": "VELA_TARGET_FREE_SELECTOR_FORCED_MISS_FALLBACK_V1",
            "model": v3.MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "capability": {
                "baseline": baseline["accuracy"],
                "after": after["accuracy"],
                "correction_before": baseline["correction_accuracy"],
                "correction_after": after["correction_accuracy"],
                "control_before": baseline["control_accuracy"],
                "control_after": after["control_accuracy"],
                "epoch_loss": losses,
            },
            "forced_miss_fallback": {
                "fixture_count": len(rows),
                "valid_fixture_count": valid_count,
                "eligible_forced_miss_count": eligible_count,
                "validation_detected_count": detected_count,
                "validation_detection_rate": detection_rate,
                "fallback_attempt_count": fallback_count,
                "fallback_recovery_count": recovered_count,
                "fallback_recovery_rate": recovery_rate,
                "residual_unsafe_after_fallback_count": residual_unsafe_count,
                "suite_pass": suite_pass,
            },
            "validator": "Hand-specified fixture invariants/probe expectations only; no W2-native state is used by validation. The oracle frontier is evaluation-only and is used solely to inject a known unsafe late anchor for this failure-path test.",
            "fixtures": rows,
            "success_definition": "On at least three nontrivial valid fixtures with a known unsafe later anchor, post-replay validation must detect every injected selector miss and conservative full replay must recover every case with zero residual unsafe state.",
            "claim_boundary": "Synthetic Mamba-130M failure-injection test. It validates control-flow mechanics for validation->full-replay fallback under hand-specified invariants; it does not prove generic miss detection, causal discovery, or identity continuity.",
        })
    except BaseException as exc:
        write_report({
            "status": "VELA_TARGET_FREE_SELECTOR_FORCED_MISS_FALLBACK_V1_ERROR",
            "model": getattr(v3, "MODEL_ID", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-50:],
        })
        raise


if __name__ == "__main__":
    run()
