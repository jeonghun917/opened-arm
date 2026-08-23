from __future__ import annotations

import copy
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
sel, dep, v2, v3, v4, fa = stress.sel, stress.dep, stress.v2, stress.v3, stress.v4, stress.fa

NEUTRAL = stress.NEUTRAL
BASES = stress.BASES
DEP_BASES = stress.DEP_BASES


def clone(src, new_id, prefix=0, middle=0, suffix=0, adversarial=None):
    return stress.clone_fixture(
        src,
        new_id,
        prefix=prefix,
        middle=middle,
        suffix=suffix,
        adversarial=adversarial,
    )


def fam(base, label, variants):
    rows = []
    for name, kwargs in variants:
        if not kwargs:
            fx = copy.deepcopy(base)
            fx["id"] = f"{label}_{name}"
        else:
            fx = clone(base, f"{label}_{name}", **kwargs)
        rows.append(fx)
    return rows


# This suite intentionally never runs the migration selector. It only asks whether
# a model processing the entire history natively can recover the intended current
# state. This isolates model/history robustness from checkpoint selection.
FIXTURES = []

# A: same-slot overwrite chain that previously produced invalid W2-native states.
FIXTURES += fam(
    DEP_BASES["superseded_chain"],
    "superseded",
    [
        ("base", {}),
        ("prefix2", {"prefix": 2}),
        ("prefix4", {"prefix": 4}),
        ("prefix8", {"prefix": 8}),
        ("middle2", {"middle": 2}),
        ("middle4", {"middle": 4}),
        ("middle8", {"middle": 8}),
        ("suffix2", {"suffix": 2}),
        ("suffix4", {"suffix": 4}),
        ("suffix8", {"suffix": 8}),
        ("both4", {"prefix": 4, "suffix": 4}),
        ("both8", {"prefix": 8, "suffix": 8}),
    ],
)

# B: a normally valid single correction, with distance/noise increased around it.
FIXTURES += fam(
    BASES["single_middle"],
    "single_middle",
    [
        ("base", {}),
        ("both2", {"prefix": 2, "suffix": 2}),
        ("both4", {"prefix": 4, "suffix": 4}),
        ("both6", {"prefix": 6, "suffix": 6}),
        ("both8", {"prefix": 8, "suffix": 8}),
        ("both10", {"prefix": 10, "suffix": 10}),
        ("both12", {"prefix": 12, "suffix": 12}),
        ("middle4", {"middle": 4}),
        ("middle8", {"middle": 8}),
        ("middle12", {"middle": 12}),
    ],
)

# C: overwrite control known to work in the prior dependency experiment. If this
# remains healthy while A fails, the problem is not simply "all overwrites fail".
FIXTURES += fam(
    DEP_BASES["late_overwrite_with_long_prefix"],
    "late_overwrite_control",
    [
        ("base", {}),
        ("prefix4", {"prefix": 4}),
        ("prefix8", {"prefix": 8}),
        ("prefix12", {"prefix": 12}),
        ("suffix4", {"suffix": 4}),
        ("suffix8", {"suffix": 8}),
        ("both8", {"prefix": 8, "suffix": 8}),
    ],
)

# D: independent-persistent control at comparable lengths. This checks whether
# degradation is generic sequence-length pressure or specific to overwrite logic.
FIXTURES += fam(
    DEP_BASES["independent_persistent"],
    "independent_control",
    [
        ("base", {}),
        ("prefix4", {"prefix": 4}),
        ("prefix8", {"prefix": 8}),
        ("prefix12", {"prefix": 12}),
        ("middle4", {"middle": 4}),
        ("middle8", {"middle": 8}),
        ("suffix8", {"suffix": 8}),
        ("both8", {"prefix": 8, "suffix": 8}),
    ],
)

# E: old-value echoes from the prior invalid stress fixture.
FIXTURES.append(
    clone(
        DEP_BASES["superseded_chain"],
        "superseded_old_value_echoes",
        prefix=3,
        middle=4,
        suffix=3,
        adversarial=[
            "A retired report contains ALPHA and BETA as historical codewords only. ",
            "A checksum label contains BETA but does not update the current codeword. ",
        ],
    )
)


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def score_native(model, tok, fx):
    ids = tok("".join(fx["segments"]), return_tensors="pt", add_special_tokens=False).input_ids
    T = int(ids.shape[1])
    with torch.no_grad():
        native = v4.clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
    probes = fa.score_probe_specs(model, tok, native, T, fx["probes"])
    acc = sum(int(r["correct"]) for r in probes) / max(len(probes), 1)
    return {
        "history_tokens": T,
        "events": len(fx["segments"]),
        "expected_accuracy": acc,
        "all_correct": acc == 1.0,
        "probe_results": [
            {
                "id": r["id"],
                "chosen": r["chosen"],
                "expected": r.get("expected"),
                "correct": bool(r["correct"]),
            }
            for r in probes
        ],
    }


def family_name(fid):
    for prefix in (
        "superseded_",
        "single_middle_",
        "late_overwrite_control_",
        "independent_control_",
    ):
        if fid.startswith(prefix):
            return prefix[:-1]
    if fid == "superseded_old_value_echoes":
        return "superseded"
    return "other"


def summarize(rows, key):
    families = {}
    for row in rows:
        family = row["family"]
        slot = families.setdefault(family, {"count": 0, "full_success": 0, "failed": []})
        slot["count"] += 1
        ok = bool(row[key]["all_correct"])
        slot["full_success"] += int(ok)
        if not ok:
            slot["failed"].append({
                "fixture": row["fixture"],
                "history_tokens": row[key]["history_tokens"],
                "wrong_probes": [p["id"] for p in row[key]["probe_results"] if not p["correct"]],
            })
    for slot in families.values():
        slot["full_success_rate"] = slot["full_success"] / max(slot["count"], 1)
    return families


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        torch.manual_seed(v3.SEED)
        random.seed(v3.SEED)
        tok = AutoTokenizer.from_pretrained(v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(v3.MODEL_ID, torch_dtype=torch.float32).cpu().eval()

        rows = []
        for fx in FIXTURES:
            rows.append({
                "fixture": fx["id"],
                "family": family_name(fx["id"]),
                "W1_native": score_native(model, tok, fx),
            })

        baseline_capability = v3.evaluate(model, tok)
        losses = v2.train_upgrade(model, tok)
        upgraded_capability = v3.evaluate(model, tok)

        by_id = {fx["id"]: fx for fx in FIXTURES}
        for row in rows:
            row["W2_native"] = score_native(model, tok, by_id[row["fixture"]])
            row["changed_from_W1"] = (
                row["W1_native"]["expected_accuracy"] != row["W2_native"]["expected_accuracy"]
                or [p["chosen"] for p in row["W1_native"]["probe_results"]]
                != [p["chosen"] for p in row["W2_native"]["probe_results"]]
            )

        w1_summary = summarize(rows, "W1_native")
        w2_summary = summarize(rows, "W2_native")
        failures = [
            {
                "fixture": r["fixture"],
                "family": r["family"],
                "history_tokens": r["W2_native"]["history_tokens"],
                "wrong_probes": [p["id"] for p in r["W2_native"]["probe_results"] if not p["correct"]],
            }
            for r in rows
            if not r["W2_native"]["all_correct"]
        ]

        report = {
            "status": "VELA_MAMBA_NATIVE_LONG_HISTORY_ROBUSTNESS_V1",
            "model": v3.MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "purpose": "Isolate native full-history model robustness from migration/selector behavior. No checkpoint selector or migrated state is used in this experiment.",
            "capability": {
                "W1": baseline_capability,
                "W2": upgraded_capability,
                "epoch_loss": losses,
            },
            "suite": {
                "fixture_count": len(rows),
                "W1_full_success_count": sum(int(r["W1_native"]["all_correct"]) for r in rows),
                "W2_full_success_count": sum(int(r["W2_native"]["all_correct"]) for r in rows),
                "W2_failure_count": len(failures),
                "W2_failures": failures,
                "W1_by_family": w1_summary,
                "W2_by_family": w2_summary,
            },
            "fixtures": rows,
            "interpretation_rule": {
                "generic_length_failure": "Comparable-length independent and overwrite controls also degrade as tokens increase.",
                "overwrite_specific_failure": "Superseded family degrades while comparable controls remain correct.",
                "distance_noise_failure": "Single-middle family crosses from correct to incorrect as neutral prefix/suffix or middle distance increases.",
                "upgrade_specific_regression": "A fixture correct under W1 becomes incorrect under W2 after the capability upgrade.",
            },
            "claim_boundary": "Deterministic synthetic histories on Mamba-130M with hand-specified probes. This diagnoses native history handling only; it is not a general long-context benchmark or identity proof.",
        }
        write_report(report)
    except BaseException as exc:
        write_report({
            "status": "VELA_MAMBA_NATIVE_LONG_HISTORY_ROBUSTNESS_V1_ERROR",
            "model": getattr(v3, "MODEL_ID", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-60:],
        })
        raise


if __name__ == "__main__":
    run()
