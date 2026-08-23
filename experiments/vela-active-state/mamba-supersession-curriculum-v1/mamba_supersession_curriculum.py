from __future__ import annotations

import gc
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

EPOCHS = 10

# Eight value triplets, deliberately separate from the ALPHA/BETA/GAMMA test chain.
# Each produces two directional examples => 16 examples, matching the baseline budget.
TRIPLES = [
    (" CAT", " DOG", " BIRD"),
    (" EAST", " WEST", " NORTH"),
    (" UP", " DOWN", " LEFT"),
    (" HOT", " COLD", " WARM"),
    (" one", " two", " three"),
    (" Mars", " Venus", " Earth"),
    (" LEFT", " RIGHT", " CENTER"),
    (" ON", " OFF", " AUTO"),
]

SUP_TEMPLATES = [
    "Initial value:{a}. Update: replace it with{b}. Later update: replace that with{c}.\nFinal current value:",
    "The value began as{a}. It was changed to{b}. A newer correction supersedes both: use{c}.\nCurrent value:",
    "Old value:{a}. Intermediate value:{b}. Latest authoritative value:{c}.\nUse the latest value:",
]


def build_supersession_train():
    rows = []
    for i, (a, b, c) in enumerate(TRIPLES):
        t = SUP_TEMPLATES[i % len(SUP_TEMPLATES)]
        rows.append((t.format(a=a, b=b, c=c), c))
        rows.append((t.format(a=c, b=b, c=a), a))
    return rows


SUP_TRAIN = build_supersession_train()
BASELINE_TRAIN = list(rob.v3.TRAIN)

# Exactly 16 examples: one direction from each of the eight baseline pairs plus
# one direction from each of the eight supersession triplets. Direction alternates
# deterministically to avoid making one target-side dominant.
MIXED_TRAIN = [
    BASELINE_TRAIN[2 * i + (i % 2)] for i in range(8)
] + [
    SUP_TRAIN[2 * i + ((i + 1) % 2)] for i in range(8)
]

CURRICULA = {
    "baseline": BASELINE_TRAIN,
    "mixed": MIXED_TRAIN,
    "supersession_only": SUP_TRAIN,
}


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


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


def evaluate_suite(model, tok):
    rows = []
    by_family = {}
    for fx in rob.FIXTURES:
        out = rob.score_native(model, tok, fx)
        family = rob.family_name(fx["id"])
        wrong = [r["id"] for r in out["probe_results"] if not r["correct"]]
        row = {
            "fixture": fx["id"],
            "family": family,
            "history_tokens": out["history_tokens"],
            "all_correct": bool(out["all_correct"]),
            "expected_accuracy": out["expected_accuracy"],
            "wrong_probes": wrong,
        }
        rows.append(row)
        slot = by_family.setdefault(family, {"count": 0, "full_success": 0, "failed": []})
        slot["count"] += 1
        slot["full_success"] += int(row["all_correct"])
        if not row["all_correct"]:
            slot["failed"].append({
                "fixture": row["fixture"],
                "history_tokens": row["history_tokens"],
                "wrong_probes": wrong,
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


def train_arm(name, train_rows, tok, AutoModelForCausalLM):
    # Every arm begins from exactly the same pretrained checkpoint and seed.
    torch.manual_seed(rob.v3.SEED)
    random.seed(rob.v3.SEED)
    model = AutoModelForCausalLM.from_pretrained(
        rob.v3.MODEL_ID, torch_dtype=torch.float32
    ).cpu().eval()
    trainable = configure_trainable_xproj(model)
    optimizer = torch.optim.AdamW(trainable, lr=rob.v3.LR, weight_decay=0.0)
    order = list(range(len(train_rows)))
    losses = []
    step_count = 0

    for epoch in range(EPOCHS):
        random.Random(rob.v3.SEED + epoch).shuffle(order)
        model.train()
        total = 0.0
        for idx in order:
            prompt, gold = train_rows[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = rob.v3.train_loss(model, tok, prompt, gold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += float(loss.detach())
            step_count += 1
        model.eval()
        losses.append(total / max(len(order), 1))

    heldout = rob.v3.evaluate(model, tok)
    long_history = evaluate_suite(model, tok)
    result = {
        "curriculum": name,
        "train_examples_per_epoch": len(train_rows),
        "batch_size": 1,
        "epochs": EPOCHS,
        "optimizer_steps": step_count,
        "epoch_mean_loss": losses,
        "heldout": {
            "accuracy": heldout["accuracy"],
            "correction_accuracy": heldout["correction_accuracy"],
            "control_accuracy": heldout["control_accuracy"],
        },
        "long_history": long_history,
    }
    del model
    gc.collect()
    return result


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        tok = AutoTokenizer.from_pretrained(rob.v3.MODEL_ID)
        arms = {}
        for name, train_rows in CURRICULA.items():
            arms[name] = train_arm(name, train_rows, tok, AutoModelForCausalLM)

        def fam_success(arm_name, family):
            slot = arms[arm_name]["long_history"]["by_family"].get(family, {})
            return int(slot.get("full_success", 0)), int(slot.get("count", 0))

        b_sup, n_sup = fam_success("baseline", "superseded")
        m_sup, _ = fam_success("mixed", "superseded")
        s_sup, _ = fam_success("supersession_only", "superseded")

        if m_sup > b_sup:
            curriculum_signal = "mixed_curriculum_improves_supersession"
        elif s_sup > b_sup:
            curriculum_signal = "supersession_heavy_only_improves_supersession"
        else:
            curriculum_signal = "no_supersession_recovery_under_tested_curricula"

        report = {
            "status": "VELA_MAMBA_SUPERSESSION_CURRICULUM_V1",
            "model": rob.v3.MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "design": {
                "arms": list(CURRICULA.keys()),
                "epochs_per_arm": EPOCHS,
                "examples_per_epoch_per_arm": 16,
                "batch_size": 1,
                "optimizer_steps_per_arm": 160,
                "trainable_pattern": "*.mixer.x_proj.weight",
                "learning_rate": rob.v3.LR,
                "evaluation_suite": "unchanged 38-fixture native long-history robustness suite",
                "selector_or_migration_used": False,
            },
            "arms": arms,
            "summary": {
                "superseded_count": n_sup,
                "baseline_superseded_full_success": b_sup,
                "mixed_superseded_full_success": m_sup,
                "supersession_only_full_success": s_sup,
                "baseline_overall_full_success": arms["baseline"]["long_history"]["full_success_count"],
                "mixed_overall_full_success": arms["mixed"]["long_history"]["full_success_count"],
                "supersession_only_overall_full_success": arms["supersession_only"]["long_history"]["full_success_count"],
                "curriculum_signal": curriculum_signal,
            },
            "interpretation_rule": {
                "curriculum_support": "A budget-matched arm containing three-stage supersession examples improves the held-out superseded family above the baseline arm.",
                "strong_curriculum_support": "The mixed arm improves supersession while retaining ordinary held-out correction capability, showing the weakness is at least partly teachable without requiring a supersession-only curriculum.",
                "no_recovery": "If both supersession-exposed arms remain at baseline on the superseded family, this specific curriculum and x_proj-only update recipe did not recover the weakness; that still does not prove an architectural impossibility.",
            },
            "claim_boundary": "Single Mamba-130M model, x_proj-only learned update, 160 optimizer steps per arm, synthetic curricula and the existing 38-fixture suite. This is a curriculum-sensitivity diagnostic, not a final backbone verdict.",
        }
        write_report(report)
    except BaseException as exc:
        write_report({
            "status": "VELA_MAMBA_SUPERSESSION_CURRICULUM_V1_ERROR",
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
