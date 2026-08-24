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

ARM = os.environ.get("VELA_RWKV7_ARM", "kvr_balanced_lr1e5_e5")

# Preserve the exact v2 winning curriculum as the baseline.
BALANCED = list(v2.BALANCED_TRAIN)  # 8 supersession + 4 correction + 4 control

# Shift pressure toward supersession while still explicitly rehearsing both
# ordinary correction and disjoint identity controls. All values remain disjoint
# from the ALPHA/BETA/RED/BLUE/LOW/HIGH evaluation values.
SUP10 = list(v2.SUP_ALL[:10])
CORR3 = [list(rw.TRAIN)[i] for i in (0, 3, 4)]
CTRL3 = list(v2.CONTROL_TRAIN[:3])
SUP_HEAVY = SUP10 + CORR3 + CTRL3  # 10 + 3 + 3 = 16

ARMS = {
    "kvr_balanced_lr5e6_e5": {
        "train": BALANCED, "lr": 5e-6, "epochs": 5,
        "purpose": "test whether a gentler step size improves retention-safe generalization",
    },
    "kvr_balanced_lr1e5_e5": {
        "train": BALANCED, "lr": 1e-5, "epochs": 5,
        "purpose": "exact recipe replication of the v2 retention-safe winner",
    },
    "kvr_balanced_lr2e5_e5": {
        "train": BALANCED, "lr": 2e-5, "epochs": 5,
        "purpose": "test whether more aggressive KVR adaptation gains supersession before retention breaks",
    },
    "kvr_balanced_lr1e5_e10": {
        "train": BALANCED, "lr": 1e-5, "epochs": 10,
        "purpose": "test whether the v2 winner was still training-limited at 80 steps",
    },
    "kvr_supheavy_lr1e5_e5": {
        "train": SUP_HEAVY, "lr": 1e-5, "epochs": 5,
        "purpose": "increase supersession pressure from 8/16 to 10/16 while retaining correction/control rehearsal",
    },
    "kvr_supheavy_lr1e5_e10": {
        "train": SUP_HEAVY, "lr": 1e-5, "epochs": 10,
        "purpose": "combine supersession-heavy curriculum with additional optimizer steps",
    },
}


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def configure_kvr(model, args):
    for p in model.z.values():
        if isinstance(p, torch.Tensor):
            p.requires_grad_(False)
    trainable = []
    for i in range(args.n_layer):
        for suffix in ("key.weight", "value.weight", "receptance.weight"):
            p = model.z[f"blocks.{i}.att.{suffix}"]
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def summarize_checkpoint(epoch, step_count, model, ns, tok):
    heldout = v1.evaluate_heldout(model, ns, tok)
    suite = v1.evaluate_suite(model, ns, tok)
    sup = suite["by_family"]["superseded"]
    retention_ok = heldout["correction_accuracy"] == 1.0 and heldout["control_accuracy"] == 1.0
    return {
        "epoch": epoch,
        "optimizer_steps": step_count,
        "heldout": {
            "accuracy": heldout["accuracy"],
            "correction_accuracy": heldout["correction_accuracy"],
            "control_accuracy": heldout["control_accuracy"],
        },
        "retention_qualified": retention_ok,
        "superseded_full_success": sup["full_success"],
        "superseded_count": sup["count"],
        "long_history": suite,
    }


def checkpoint_epochs(epochs: int):
    pts = {0, 1, 3, 5}
    if epochs >= 10:
        pts.update({7, 10})
    return {x for x in pts if x <= epochs}


def best_checkpoint(checkpoints, require_retention=True):
    pool = [c for c in checkpoints if c["retention_qualified"] or not require_retention]
    if not pool:
        return None
    return max(
        pool,
        key=lambda c: (
            c["superseded_full_success"],
            c["long_history"]["full_success_count"],
            c["heldout"]["accuracy"],
            -c["epoch"],
        ),
    )


def compact(cp):
    if cp is None:
        return None
    return {
        "epoch": cp["epoch"],
        "optimizer_steps": cp["optimizer_steps"],
        "superseded_full_success": cp["superseded_full_success"],
        "superseded_count": cp["superseded_count"],
        "overall_full_success": cp["long_history"]["full_success_count"],
        "overall_count": cp["long_history"]["fixture_count"],
        "heldout_accuracy": cp["heldout"]["accuracy"],
        "correction_accuracy": cp["heldout"]["correction_accuracy"],
        "control_accuracy": cp["heldout"]["control_accuracy"],
        "retention_qualified": cp["retention_qualified"],
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

        weight_path = hf_hub_download(
            repo_id=rw.WEIGHT_REPO,
            filename=rw.WEIGHT_FILE,
            revision=rw.WEIGHT_REVISION,
        )
        model, args, ns = rw.load_reference(weight_path)

        with tempfile.TemporaryDirectory() as td:
            vp = Path(td) / "vocab.txt"
            urllib.request.urlretrieve(rw.VOCAB_URL, vp)
            tok = rw.RWKVTokenizer(str(vp))

            trainable = configure_kvr(model, args)
            optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=0.0)
            train_rows = list(cfg["train"])
            order = list(range(len(train_rows)))
            cp_epochs = checkpoint_epochs(cfg["epochs"])
            checkpoints = [summarize_checkpoint(0, 0, model, ns, tok)]
            losses = []
            steps = 0

            for epoch in range(1, cfg["epochs"] + 1):
                random.Random(rw.SEED + epoch).shuffle(order)
                total = 0.0
                for idx in order:
                    optimizer.zero_grad(set_to_none=True)
                    loss = rw.train_example(model, ns, tok, *train_rows[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    total += float(loss.detach())
                    steps += 1
                losses.append(total / max(len(order), 1))
                if epoch in cp_epochs:
                    checkpoints.append(summarize_checkpoint(epoch, steps, model, ns, tok))

            raw_best = best_checkpoint(checkpoints, require_retention=False)
            safe_best = best_checkpoint(checkpoints, require_retention=True)
            report = {
                "status": "VELA_RWKV7_KVR_SWEEP_V3",
                "arm": ARM,
                "source_commit": rw.SOURCE_COMMIT,
                "weight_revision": rw.WEIGHT_REVISION,
                "weight_file": rw.WEIGHT_FILE,
                "device": "cpu",
                "dtype": "float32",
                "design": {
                    "question": "How far can the retention-safe RWKV-7 KVR recipe push semantic supersession under narrow LR, step-count, and curriculum-pressure changes?",
                    "purpose": cfg["purpose"],
                    "trainable_scope": "blocks.*.att.{key,value,receptance}.weight",
                    "trainable_numel": int(sum(p.numel() for p in trainable)),
                    "learning_rate": cfg["lr"],
                    "epochs": cfg["epochs"],
                    "examples_per_epoch": len(train_rows),
                    "batch_size": 1,
                    "optimizer_steps": steps,
                    "checkpoint_epochs": sorted(cp_epochs),
                    "curriculum": "balanced_8sup_4corr_4ctrl" if train_rows == BALANCED else "supheavy_10sup_3corr_3ctrl",
                    "evaluation_suite": "unchanged 38-fixture native long-history robustness suite",
                    "selector_or_migration_used": False,
                    "retention_gate": "heldout correction_accuracy == 1.0 and control_accuracy == 1.0",
                },
                "epoch_mean_loss": losses,
                "checkpoints": checkpoints,
                "summary": {
                    "raw_best": compact(raw_best),
                    "retention_qualified_best": compact(safe_best),
                    "retention_safe_supersession_recovery": bool(safe_best and safe_best["superseded_full_success"] > 0),
                },
                "claim_boundary": "Released RWKV-7 0.1B, synthetic curricula, CPU float32, KVR-only adaptation, fixed held-out and 38-fixture diagnostics. This is a local hyperparameter/curriculum sweep, not a final backbone verdict.",
            }
            write_report(report)

        del model
        gc.collect()
    except BaseException as exc:
        write_report({
            "status": "VELA_RWKV7_KVR_SWEEP_V3_ERROR",
            "arm": ARM,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-100:],
        })
        raise


if __name__ == "__main__":
    run()
