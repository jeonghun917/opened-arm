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
curr = load_module("vela_sup_curr", BASE / "mamba-supersession-curriculum-v1" / "mamba_supersession_curriculum.py")

EPOCHS = 5
CHECKPOINT_EPOCHS = {0, 1, 3, 5}
ARM = os.environ.get("VELA_RWKV7_ARM", "key_balanced_lr5e5")

# Keep evaluation values ALPHA/BETA/RED/BLUE/LOW/HIGH out of these retention examples.
CONTROL_TRAIN = [
    ("Codeword: CAT.\nCodeword:", " CAT"),
    ("Codeword: EAST.\nCodeword:", " EAST"),
    ("Codeword: UP.\nCodeword:", " UP"),
    ("Codeword: HOT.\nCodeword:", " HOT"),
]

SUP_ALL = list(curr.SUP_TRAIN)  # 16 examples
SUP_HALF = [SUP_ALL[2 * i + (i % 2)] for i in range(8)]
CORR_HALF = [list(rw.TRAIN)[i] for i in (0, 3, 4, 7)]
BALANCED_TRAIN = SUP_HALF + CORR_HALF + CONTROL_TRAIN  # 8 supersession + 4 correction + 4 retention controls

ARMS = {
    "key_sup_only_lr5e5": {
        "train": SUP_ALL,
        "lr": 5e-5,
        "scope": "key",
        "purpose": "replicate the v1 supersession-only pressure under the fixed evaluator",
    },
    "key_balanced_lr5e5": {
        "train": BALANCED_TRAIN,
        "lr": 5e-5,
        "scope": "key",
        "purpose": "test whether explicit retention examples prevent control collapse while preserving supersession learning",
    },
    "key_balanced_lr1e5": {
        "train": BALANCED_TRAIN,
        "lr": 1e-5,
        "scope": "key",
        "purpose": "separate curriculum retention from learning-rate-induced forgetting",
    },
    "kvr_balanced_lr1e5": {
        "train": BALANCED_TRAIN,
        "lr": 1e-5,
        "scope": "kvr",
        "purpose": "test whether a wider recurrent update scope improves supersession without the full-model instability seen in Mamba",
    },
}


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def configure_trainable(model, args, scope: str):
    for p in model.z.values():
        if isinstance(p, torch.Tensor):
            p.requires_grad_(False)
    trainable = []
    suffixes = ["key.weight"] if scope == "key" else ["key.weight", "value.weight", "receptance.weight"]
    for i in range(args.n_layer):
        for suffix in suffixes:
            name = f"blocks.{i}.att.{suffix}"
            p = model.z[name]
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


def best_checkpoint(checkpoints, require_retention: bool):
    pool = [c for c in checkpoints if (c["retention_qualified"] or not require_retention)]
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


def compact_checkpoint(cp):
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

            trainable = configure_trainable(model, args, cfg["scope"])
            opt = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=0.0)
            train_rows = list(cfg["train"])
            order = list(range(len(train_rows)))
            losses = []
            checkpoints = [summarize_checkpoint(0, 0, model, ns, tok)]
            step_count = 0

            for epoch in range(1, EPOCHS + 1):
                random.Random(rw.SEED + epoch).shuffle(order)
                total = 0.0
                for idx in order:
                    opt.zero_grad(set_to_none=True)
                    loss = rw.train_example(model, ns, tok, *train_rows[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                    total += float(loss.detach())
                    step_count += 1
                losses.append(total / max(len(order), 1))
                if epoch in CHECKPOINT_EPOCHS:
                    checkpoints.append(summarize_checkpoint(epoch, step_count, model, ns, tok))

            raw_best = best_checkpoint(checkpoints, require_retention=False)
            safe_best = best_checkpoint(checkpoints, require_retention=True)
            report = {
                "status": "VELA_RWKV7_RETENTION_ADAPTATION_V2",
                "arm": ARM,
                "source_commit": rw.SOURCE_COMMIT,
                "weight_revision": rw.WEIGHT_REVISION,
                "weight_file": rw.WEIGHT_FILE,
                "device": "cpu",
                "dtype": "float32",
                "design": {
                    "question": "Can RWKV-7 improve semantic supersession while preserving ordinary correction and control behavior?",
                    "purpose": cfg["purpose"],
                    "curriculum": "supersession-only" if ARM == "key_sup_only_lr5e5" else "8 supersession + 4 ordinary correction + 4 disjoint identity-control examples",
                    "trainable_scope": cfg["scope"],
                    "trainable_numel": int(sum(p.numel() for p in trainable)),
                    "learning_rate": cfg["lr"],
                    "epochs": EPOCHS,
                    "examples_per_epoch": len(train_rows),
                    "batch_size": 1,
                    "optimizer_steps": step_count,
                    "checkpoint_epochs": sorted(CHECKPOINT_EPOCHS),
                    "evaluation_suite": "unchanged 38-fixture native long-history robustness suite",
                    "selector_or_migration_used": False,
                    "retention_gate": "heldout correction_accuracy == 1.0 and control_accuracy == 1.0",
                },
                "epoch_mean_loss": losses,
                "checkpoints": checkpoints,
                "summary": {
                    "raw_best": compact_checkpoint(raw_best),
                    "retention_qualified_best": compact_checkpoint(safe_best),
                    "retention_qualified_supersession_recovery": bool(safe_best and safe_best["superseded_full_success"] > 0),
                    "any_supersession_recovery": any(c["superseded_full_success"] > 0 for c in checkpoints[1:]),
                },
                "claim_boundary": "Released RWKV-7 0.1B, synthetic curricula, CPU float32, 80 optimizer steps per arm, fixed held-out and 38-fixture diagnostics. This is a retention/teachability screen, not a final backbone verdict.",
            }
            write_report(report)

        del model
        gc.collect()
    except BaseException as exc:
        write_report({
            "status": "VELA_RWKV7_RETENTION_ADAPTATION_V2_ERROR",
            "arm": ARM,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-100:],
        })
        raise


if __name__ == "__main__":
    run()
