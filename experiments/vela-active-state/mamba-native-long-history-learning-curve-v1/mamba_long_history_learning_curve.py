from __future__ import annotations

import importlib.util
import json
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
ROBUST_PATH = BASE / "long-history-native-robustness-v1" / "mamba_long_history_native_robustness.py"
spec = importlib.util.spec_from_file_location("vela_longhist_v1", ROBUST_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {ROBUST_PATH}")
rob = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rob)

# Same 38-fixture native-history suite; only training duration changes.
EPOCHS = [0, 1, 3, 5, 10]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def evaluate_suite(model, tok):
    rows = []
    for fx in rob.FIXTURES:
        out = rob.score_native(model, tok, fx)
        rows.append({
            "fixture": fx["id"],
            "family": rob.family_name(fx["id"]),
            "history_tokens": out["history_tokens"],
            "all_correct": bool(out["all_correct"]),
            "expected_accuracy": out["expected_accuracy"],
            "wrong_probes": [r["id"] for r in out["probe_results"] if not r["correct"]],
        })
    by_family = {}
    for row in rows:
        slot = by_family.setdefault(row["family"], {"count": 0, "full_success": 0, "failed": []})
        slot["count"] += 1
        slot["full_success"] += int(row["all_correct"])
        if not row["all_correct"]:
            slot["failed"].append({
                "fixture": row["fixture"],
                "history_tokens": row["history_tokens"],
                "wrong_probes": row["wrong_probes"],
            })
    for slot in by_family.values():
        slot["full_success_rate"] = slot["full_success"] / max(slot["count"], 1)
    successes = sum(int(r["all_correct"]) for r in rows)
    return {
        "fixture_count": len(rows),
        "full_success_count": successes,
        "full_success_rate": successes / max(len(rows), 1),
        "by_family": by_family,
        "fixtures": rows,
    }


def configure_trainable_xproj(model):
    trainable = []
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if ".mixer.x_proj.weight" in name:
            p.requires_grad_(True)
            trainable.append(p)
    if not trainable:
        raise RuntimeError("no x_proj weights found")
    return trainable


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        torch.manual_seed(rob.v3.SEED)
        random.seed(rob.v3.SEED)
        tok = AutoTokenizer.from_pretrained(rob.v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            rob.v3.MODEL_ID, torch_dtype=torch.float32
        ).cpu().eval()

        trainable = configure_trainable_xproj(model)
        optimizer = torch.optim.AdamW(trainable, lr=rob.v3.LR, weight_decay=0.0)
        order = list(range(len(rob.v3.TRAIN)))

        checkpoints = {0: rob.v2.save_xproj(model)}
        epoch_losses = []
        for epoch in range(1, max(EPOCHS) + 1):
            random.Random(rob.v3.SEED + epoch - 1).shuffle(order)
            model.train()
            total = 0.0
            for idx in order:
                prompt, gold = rob.v3.TRAIN[idx]
                optimizer.zero_grad(set_to_none=True)
                loss = rob.v3.train_loss(model, tok, prompt, gold)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                total += float(loss.detach())
            model.eval()
            epoch_losses.append(total / max(len(order), 1))
            if epoch in EPOCHS:
                checkpoints[epoch] = rob.v2.save_xproj(model)

        curve = []
        for epoch in EPOCHS:
            rob.v2.load_xproj(model, checkpoints[epoch])
            model.eval()
            heldout = rob.v3.evaluate(model, tok)
            suite = evaluate_suite(model, tok)
            curve.append({
                "epoch": epoch,
                "heldout": {
                    "accuracy": heldout["accuracy"],
                    "correction_accuracy": heldout["correction_accuracy"],
                    "control_accuracy": heldout["control_accuracy"],
                },
                "long_history": suite,
            })

        overall = [x["long_history"]["full_success_rate"] for x in curve]
        superseded = [
            x["long_history"]["by_family"].get("superseded", {}).get("full_success_rate", 0.0)
            for x in curve
        ]
        single_middle = [
            x["long_history"]["by_family"].get("single_middle", {}).get("full_success_rate", 0.0)
            for x in curve
        ]
        late_overwrite = [
            x["long_history"]["by_family"].get("late_overwrite_control", {}).get("full_success_rate", 0.0)
            for x in curve
        ]
        independent = [
            x["long_history"]["by_family"].get("independent_control", {}).get("full_success_rate", 0.0)
            for x in curve
        ]

        if superseded[-1] > superseded[1]:
            superseded_trend = "improves_with_more_training"
        elif max(superseded[1:]) == superseded[1] and superseded[-1] <= superseded[1]:
            superseded_trend = "no_clear_training_recovery"
        else:
            superseded_trend = "non_monotonic_or_mixed"

        write_report({
            "status": "VELA_MAMBA_NATIVE_LONG_HISTORY_LEARNING_CURVE_V1",
            "model": rob.v3.MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "epochs": EPOCHS,
            "epoch_mean_loss": epoch_losses,
            "curve": curve,
            "summary": {
                "overall_full_success_rates": overall,
                "superseded_full_success_rates": superseded,
                "single_middle_full_success_rates": single_middle,
                "late_overwrite_control_full_success_rates": late_overwrite,
                "independent_control_full_success_rates": independent,
                "superseded_training_trend": superseded_trend,
            },
            "success_definition": "Measure the same 38 native full-history fixtures at epochs 0,1,3,5,10 under the same x_proj learned-upgrade recipe, separating undertraining recovery from a weakness that persists under longer training.",
            "claim_boundary": "Single Mamba-130M learned-upgrade recipe and synthetic 38-fixture suite. A rising curve supports an undertraining explanation; a flat curve only shows persistence under this recipe and does not prove an architectural impossibility.",
        })
    except BaseException as exc:
        write_report({
            "status": "VELA_MAMBA_NATIVE_LONG_HISTORY_LEARNING_CURVE_V1_ERROR",
            "model": getattr(rob.v3, "MODEL_ID", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-60:],
        })
        raise


if __name__ == "__main__":
    run()
