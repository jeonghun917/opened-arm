from __future__ import annotations

import importlib.util
import json
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
ROBUST_PATH = BASE / "mamba-native-long-history-robustness-v1" / "mamba_native_long_history_robustness.py"
spec = importlib.util.spec_from_file_location("vela_longhist_v1", ROBUST_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {ROBUST_PATH}")
rob = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rob)

# Reuse the exact 38-fixture native-history suite from robustness-v1 so that the
# only intended independent variable is training duration / checkpoint epoch.
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
    by_family: dict[str, dict] = {}
    for fx in rob.FIXTURES:
        out = rob.evaluate_fixture(model, tok, fx)
        ok = bool(out["all_correct"])
        wrong = [r["id"] for r in out["probe_results"] if not r["correct"]]
        rows.append({
            "fixture": fx["id"],
            "family": fx["family"],
            "history_tokens": out["history_tokens"],
            "all_correct": ok,
            "expected_accuracy": out["expected_accuracy"],
            "wrong_probes": wrong,
        })
        fam = by_family.setdefault(fx["family"], {"count": 0, "full_success": 0, "failed": []})
        fam["count"] += 1
        fam["full_success"] += int(ok)
        if not ok:
            fam["failed"].append({"fixture": fx["id"], "history_tokens": out["history_tokens"], "wrong_probes": wrong})
    for fam in by_family.values():
        fam["full_success_rate"] = fam["full_success"] / max(fam["count"], 1)
    return {
        "fixture_count": len(rows),
        "full_success_count": sum(int(r["all_correct"]) for r in rows),
        "full_success_rate": sum(int(r["all_correct"]) for r in rows) / max(len(rows), 1),
        "by_family": by_family,
        "fixtures": rows,
    }


def save_trainable(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n].to(device=p.device, dtype=p.dtype))


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        torch.manual_seed(rob.v3.SEED)
        random.seed(rob.v3.SEED)
        tok = AutoTokenizer.from_pretrained(rob.v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(rob.v3.MODEL_ID, torch_dtype=torch.float32).cpu().eval()

        # Keep exactly the same learned-upgrade recipe as the prior experiments.
        rob.v3.freeze_for_upgrade(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rob.v3.LR)
        train_rows = rob.v3.build_train_rows(tok)

        checkpoints = {0: save_trainable(model)}
        epoch_losses = []
        max_epoch = max(EPOCHS)
        for epoch in range(1, max_epoch + 1):
            random.Random(rob.v3.SEED + epoch).shuffle(train_rows)
            total = 0.0
            model.train()
            for row in train_rows:
                optimizer.zero_grad(set_to_none=True)
                loss = rob.v3.train_loss(model, tok, row)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            model.eval()
            epoch_losses.append(total / max(len(train_rows), 1))
            if epoch in EPOCHS:
                checkpoints[epoch] = save_trainable(model)

        curve = []
        for epoch in EPOCHS:
            load_trainable(model, checkpoints[epoch])
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

        # Compact trend labels; descriptive only, not an architecture verdict.
        overwrite = [x["long_history"]["by_family"].get("superseded", {}).get("full_success_rate", 0.0) for x in curve]
        overall = [x["long_history"]["full_success_rate"] for x in curve]
        if overwrite[-1] > overwrite[1]:
            overwrite_trend = "improves_with_more_training"
        elif max(overwrite[1:]) == overwrite[1] and overwrite[-1] <= overwrite[1]:
            overwrite_trend = "no_clear_training_recovery"
        else:
            overwrite_trend = "non_monotonic_or_mixed"

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
                "superseded_full_success_rates": overwrite,
                "superseded_training_trend": overwrite_trend,
            },
            "success_definition": "Measure the exact same 38 native full-history fixtures at progressively longer training checkpoints (0,1,3,5,10 epochs) to distinguish undertraining from a persistent overwrite/supersession weakness.",
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
