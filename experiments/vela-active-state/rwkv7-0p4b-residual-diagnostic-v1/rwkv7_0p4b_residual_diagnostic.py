from __future__ import annotations

import gc
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rw = load_module("vela_rwkv7_learned", BASE / "rwkv7-learned-upgrade-v1" / "rwkv7_learned_upgrade.py")
v1 = load_module("vela_rwkv7_sup_v1", BASE / "rwkv7-supersession-adaptation-v1" / "rwkv7_supersession_adaptation.py")
v2 = load_module("vela_rwkv7_retention_v2", BASE / "rwkv7-retention-adaptation-v2" / "rwkv7_retention_adaptation.py")
scale = load_module("vela_rwkv7_scale_0p4b", BASE / "rwkv7-scale-0p4b-v1" / "rwkv7_scale_0p4b.py")
native = load_module("vela_rwkv7_native_long", BASE / "rwkv7-native-long-history-v1" / "rwkv7_native_long_history.py")
rob = load_module("vela_longhist", BASE / "long-history-native-robustness-v1" / "mamba_long_history_native_robustness.py")
narrow = load_module("vela_rwkv7_0p4b_sweep_v2", BASE / "rwkv7-0p4b-narrow-sweep-v2" / "rwkv7_0p4b_narrow_sweep.py")

ARM = os.environ.get("VELA_RWKV7_0P4B_RESIDUAL_ARM", "baseline_supheavy")

FRESH_NEUTRAL = [
    "A routine billing record was archived. ",
    "The auxiliary beacon completed recalibration. ",
    "A noncritical scheduling note was filed. ",
    "The spare sensor completed a self-check. ",
    "A shipping manifest was closed without changes. ",
    "An unrelated cooling log was rotated. ",
    "A housekeeping task completed normally. ",
    "The secondary clock reported nominal drift. ",
    "A maintenance ticket was marked informational. ",
    "An inventory scan found no actionable change. ",
]


def noise(n: int, offset: int = 0) -> str:
    return "".join(FRESH_NEUTRAL[(offset + i) % len(FRESH_NEUTRAL)] for i in range(n))


# Disjoint from the 38-fixture text. These rows teach persistence across irrelevant suffixes,
# not the ALPHA/BETA/GAMMA chain itself.
STATUS_TRAIN = [
    ("Project Lyra remains active. Verification is incomplete. " + noise(3, 0) + "\nVerification status:", " incomplete"),
    ("Project Lyra remains active. Verification is complete. " + noise(3, 2) + "\nVerification status:", " complete"),
    ("Project Vega remains active. Verification remains incomplete. " + noise(6, 1) + "\nVerification status:", " incomplete"),
    ("Project Vega remains active. Verification remains complete. " + noise(6, 4) + "\nVerification status:", " complete"),
    ("Project Draco remains active. Audit state is pending. " + noise(4, 3) + "\nAudit state:", " pending"),
    ("Project Draco remains active. Audit state is cleared. " + noise(4, 5) + "\nAudit state:", " cleared"),
    ("Project Mira remains active. Inspection state is pending. " + noise(7, 0) + "\nInspection state:", " pending"),
    ("Project Mira remains active. Inspection state is cleared. " + noise(7, 2) + "\nInspection state:", " cleared"),
]

STATUS_HELDOUT = [
    {"id": "verification-incomplete-5", "prompt": "Project Cygnus remains active. Verification is incomplete. " + noise(5, 5) + "\nVerification status:", "candidates": [" incomplete", " complete"], "expected": " incomplete"},
    {"id": "verification-complete-9", "prompt": "Project Cygnus remains active. Verification is complete. " + noise(9, 1) + "\nVerification status:", "candidates": [" incomplete", " complete"], "expected": " complete"},
    {"id": "audit-pending-5", "prompt": "Project Phoenix remains active. Audit state is pending. " + noise(5, 4) + "\nAudit state:", "candidates": [" pending", " cleared"], "expected": " pending"},
    {"id": "audit-cleared-8", "prompt": "Project Phoenix remains active. Audit state is cleared. " + noise(8, 6) + "\nAudit state:", "candidates": [" pending", " cleared"], "expected": " cleared"},
]

SUPHEAVY = list(narrow.SUP_HEAVY)
STATUS4 = list(STATUS_TRAIN[:4])
STATUS8 = list(STATUS_TRAIN)

ARMS = {
    "baseline_supheavy": {
        "stages": [(SUPHEAVY, 1e-5)],
        "purpose": "reproduce the 11/13 best and identify whether its two failures are actually overwrite failures",
    },
    "joint_status4": {
        "stages": [(SUPHEAVY + STATUS4, 1e-5)],
        "purpose": "jointly add four suffix-retention examples without widening trainable scope",
    },
    "joint_status8": {
        "stages": [(SUPHEAVY + STATUS8, 1e-5)],
        "purpose": "jointly add eight suffix-retention examples without widening trainable scope",
    },
    "patch_status_lr5e6": {
        "stages": [(SUPHEAVY, 1e-5), (STATUS8, 5e-6)],
        "purpose": "first reproduce the strong supersession solution, then apply a low-LR status-retention patch",
    },
    "patch_status_lr1e5": {
        "stages": [(SUPHEAVY, 1e-5), (STATUS8, 1e-5)],
        "purpose": "first reproduce the strong supersession solution, then apply an equal-LR status-retention patch",
    },
}


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train_pass(model, ns, tok, trainable, rows, lr, seed_offset):
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    order = list(range(len(rows)))
    random.Random(rw.SEED + seed_offset).shuffle(order)
    total = 0.0
    for idx in order:
        opt.zero_grad(set_to_none=True)
        loss = rw.train_example(model, ns, tok, *rows[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        total += float(loss.detach())
    return total / max(len(rows), 1)


def score_status_heldout(model, ns, tok):
    rows = []
    for fx in STATUS_HELDOUT:
        pids = tok.encode(fx["prompt"])
        with torch.no_grad():
            out, st = rw.run_tokens(model, ns, pids, rw.zero_state(model.args, model))
            st = rw.clone_state(st)
        if out is None:
            raise RuntimeError(f"empty prompt for {fx['id']}")
        scores = {c: v1.score_candidate_from_logits(model, ns, tok, out, st, c) for c in fx["candidates"]}
        chosen = max(scores, key=scores.get)
        rows.append({"id": fx["id"], "chosen": chosen, "expected": fx["expected"], "correct": chosen == fx["expected"], "scores": scores})
    correct = sum(int(x["correct"]) for x in rows)
    return {"accuracy": correct / max(len(rows), 1), "correct": correct, "count": len(rows), "rows": rows}


def probe_margin(row):
    expected = row["expected"]
    exp_score = row["scores"][expected]
    other = max(v for k, v in row["scores"].items() if k != expected)
    return exp_score - other


def evaluate_detailed_suite(model, ns, tok):
    fixtures = []
    by_family = {}
    sup_probe = {}
    for fx in rob.FIXTURES:
        out = native.score_fixture(model, ns, tok, fx)
        family = rob.family_name(fx["id"])
        probes = []
        for p in out["probe_results"]:
            q = dict(p)
            q["expected_margin"] = probe_margin(p)
            probes.append(q)
            if family == "superseded":
                s = sup_probe.setdefault(p["id"], {"correct": 0, "count": 0})
                s["correct"] += int(p["correct"])
                s["count"] += 1
        row = {
            "fixture": fx["id"],
            "family": family,
            "history_tokens": out["history_tokens"],
            "all_correct": bool(out["all_correct"]),
            "expected_accuracy": out["expected_accuracy"],
            "probe_results": probes,
            "wrong_probes": [p["id"] for p in probes if not p["correct"]],
        }
        fixtures.append(row)
        slot = by_family.setdefault(family, {"count": 0, "full_success": 0})
        slot["count"] += 1
        slot["full_success"] += int(row["all_correct"])
    for slot in by_family.values():
        slot["full_success_rate"] = slot["full_success"] / max(slot["count"], 1)
    for slot in sup_probe.values():
        slot["accuracy"] = slot["correct"] / max(slot["count"], 1)
    residual = [
        {"fixture": r["fixture"], "history_tokens": r["history_tokens"], "wrong_probes": r["wrong_probes"], "probe_results": r["probe_results"]}
        for r in fixtures if r["family"] == "superseded" and not r["all_correct"]
    ]
    focus_ids = {"superseded_suffix2", "superseded_suffix4", "superseded_suffix8"}
    focus = [r for r in fixtures if r["fixture"] in focus_ids]
    return {
        "fixture_count": len(fixtures),
        "full_success_count": sum(int(r["all_correct"]) for r in fixtures),
        "by_family": by_family,
        "superseded_probe_accuracy": sup_probe,
        "superseded_residual_failures": residual,
        "suffix_focus": focus,
        "fixtures": fixtures,
    }


def compact_eval(heldout, status, suite):
    sup = suite["by_family"]["superseded"]
    p = suite["superseded_probe_accuracy"]
    return {
        "correction_accuracy": heldout["correction_accuracy"],
        "control_accuracy": heldout["control_accuracy"],
        "status_holdout_accuracy": status["accuracy"],
        "superseded_full_success": sup["full_success"],
        "superseded_count": sup["count"],
        "superseded_codeword_correct": (p.get("codeword") or {}).get("correct"),
        "superseded_verification_correct": (p.get("verification") or {}).get("correct"),
        "overall_full_success": suite["full_success_count"],
        "overall_count": suite["fixture_count"],
    }


def run():
    if ARM not in ARMS:
        raise RuntimeError(f"unknown arm {ARM}; choices={sorted(ARMS)}")
    cfg = ARMS[ARM]
    try:
        from huggingface_hub import hf_hub_download

        torch.manual_seed(rw.SEED)
        random.seed(rw.SEED)
        torch.set_num_threads(2)
        weight_path = hf_hub_download(repo_id=scale.WEIGHT_REPO, filename=scale.WEIGHT_FILE, revision=scale.WEIGHT_REVISION)
        model, args, ns = scale.load_reference_scaled(weight_path)

        with tempfile.TemporaryDirectory() as td:
            vp = Path(td) / "vocab.txt"
            urllib.request.urlretrieve(rw.VOCAB_URL, vp)
            tok = rw.RWKVTokenizer(str(vp))
            trainable = scale.configure_kvr(model, args)

            stage_rows = []
            total_steps = 0
            for stage_idx, (rows, lr) in enumerate(cfg["stages"], start=1):
                loss = train_pass(model, ns, tok, trainable, list(rows), lr, stage_idx)
                total_steps += len(rows)
                stage_rows.append({"stage": stage_idx, "examples": len(rows), "learning_rate": lr, "mean_loss": loss})

            heldout = v1.evaluate_heldout(model, ns, tok)
            status = score_status_heldout(model, ns, tok)
            suite = evaluate_detailed_suite(model, ns, tok)
            compact = compact_eval(heldout, status, suite)
            core_retention = heldout["correction_accuracy"] == 1.0 and heldout["control_accuracy"] == 1.0
            status_generalizes = status["accuracy"] == 1.0
            overwrite_probe_13 = compact["superseded_codeword_correct"] == compact["superseded_count"]
            full_13 = compact["superseded_full_success"] == compact["superseded_count"]

            write_report({
                "status": "VELA_RWKV7_0P4B_RESIDUAL_DIAGNOSTIC_V1",
                "arm": ARM,
                "source_commit": rw.SOURCE_COMMIT,
                "weight_revision": scale.WEIGHT_REVISION,
                "weight_file": scale.WEIGHT_FILE,
                "device": "cpu",
                "dtype": "float32",
                "design": {
                    "question": "Are the two 11/13 residual superseded fixtures actually semantic-overwrite failures, and can a disjoint suffix-retention patch remove them without harming correction/control?",
                    "purpose": cfg["purpose"],
                    "trainable_scope": "blocks.*.att.{key,value,receptance}.weight",
                    "trainable_numel": int(sum(p.numel() for p in trainable)),
                    "stages": stage_rows,
                    "optimizer_steps": total_steps,
                    "evaluation_suite": "unchanged 38-fixture suite plus disjoint four-case status-retention holdout",
                    "selector_or_migration_used": False,
                },
                "evaluation": {
                    "heldout": heldout,
                    "status_retention_holdout": status,
                    "long_history": suite,
                },
                "summary": {
                    **compact,
                    "core_retention_qualified": core_retention,
                    "status_generalization_qualified": status_generalizes,
                    "superseded_codeword_13_of_13": overwrite_probe_13,
                    "superseded_full_fixture_13_of_13": full_13,
                    "residual_failure_count": len(suite["superseded_residual_failures"]),
                    "residual_failure_probe_sets": [r["wrong_probes"] for r in suite["superseded_residual_failures"]],
                },
                "interpretation_rule": {
                    "not_overwrite_residual": "If codeword is 13/13 but failed superseded fixtures only miss verification, the remaining 11/13 full-fixture gap is persistent-fact/suffix retention, not semantic overwrite.",
                    "targeted_patch_success": "Correction/control remain 100%, disjoint status holdout is 100%, and full superseded fixtures improve without reducing codeword 13/13.",
                },
                "claim_boundary": "Repeatedly inspected synthetic 38-fixture suite plus a small disjoint status-retention holdout on one released RWKV-7 0.4B checkpoint. This separates failure mode and tests a narrow patch; it is not final backbone promotion evidence.",
            })

        del model
        gc.collect()
    except BaseException as exc:
        write_report({
            "status": "VELA_RWKV7_0P4B_RESIDUAL_DIAGNOSTIC_V1_ERROR",
            "arm": ARM,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-120:],
        })
        raise


if __name__ == "__main__":
    run()
