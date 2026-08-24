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

ARM = os.environ.get("VELA_RWKV7_0P4B_ARM", "balanced_lr1e5")
CHECKPOINT_EPOCHS = {0, 1, 2, 3}

SUP_ALL = list(v2.SUP_ALL)
BALANCED = list(v2.BALANCED_TRAIN)  # 8 sup + 4 correction + 4 controls
SUP_HEAVY = SUP_ALL[:10] + list(v2.CORR_HALF[:3]) + list(v2.CONTROL_TRAIN[:3])  # 10 + 3 + 3

ARMS = {
    "balanced_lr7p5e6": {"lr": 7.5e-6, "train": BALANCED, "curriculum": "balanced_8sup_4corr_4ctrl"},
    "balanced_lr1e5": {"lr": 1.0e-5, "train": BALANCED, "curriculum": "balanced_8sup_4corr_4ctrl"},
    "balanced_lr1p5e5": {"lr": 1.5e-5, "train": BALANCED, "curriculum": "balanced_8sup_4corr_4ctrl"},
    "balanced_lr2e5": {"lr": 2.0e-5, "train": BALANCED, "curriculum": "balanced_8sup_4corr_4ctrl"},
    "supheavy_lr1e5": {"lr": 1.0e-5, "train": SUP_HEAVY, "curriculum": "supheavy_10sup_3corr_3ctrl"},
}


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


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


def best(checkpoints, require_retention=True):
    pool = [c for c in checkpoints if c["retention_qualified"] or not require_retention]
    if not pool:
        return None
    return max(pool, key=lambda c: (
        c["superseded_full_success"],
        c["long_history"]["full_success_count"],
        c["heldout"]["accuracy"],
        -c["epoch"],
    ))


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
            repo_id=scale.WEIGHT_REPO,
            filename=scale.WEIGHT_FILE,
            revision=scale.WEIGHT_REVISION,
        )
        model, args, ns = scale.load_reference_scaled(weight_path)

        with tempfile.TemporaryDirectory() as td:
            vp = Path(td) / "vocab.txt"
            urllib.request.urlretrieve(rw.VOCAB_URL, vp)
            tok = rw.RWKVTokenizer(str(vp))

            trainable = scale.configure_kvr(model, args)
            optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=0.0)
            train_rows = list(cfg["train"])
            order = list(range(len(train_rows)))
            checkpoints = [scale.summarize(0, 0, model, ns, tok)]
            losses = []
            steps = 0

            for epoch in range(1, 4):
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
                losses.append(total / len(order))
                checkpoints.append(scale.summarize(epoch, steps, model, ns, tok))

            safe_best = best(checkpoints, True)
            raw_best = best(checkpoints, False)
            report = {
                "status": "VELA_RWKV7_0P4B_NARROW_SWEEP_V2",
                "arm": ARM,
                "source_commit": rw.SOURCE_COMMIT,
                "weight_repo": scale.WEIGHT_REPO,
                "weight_revision": scale.WEIGHT_REVISION,
                "weight_file": scale.WEIGHT_FILE,
                "device": "cpu",
                "dtype": "float32",
                "design": {
                    "question": "Can a narrow 0.4B KVR sweep close the remaining 11/13 semantic-supersession gap without losing held-out correction/control?",
                    "trainable_scope": "blocks.*.att.{key,value,receptance}.weight",
                    "trainable_numel": int(sum(p.numel() for p in trainable)),
                    "learning_rate": cfg["lr"],
                    "epochs": 3,
                    "examples_per_epoch": len(train_rows),
                    "optimizer_steps": steps,
                    "checkpoint_epochs": sorted(CHECKPOINT_EPOCHS),
                    "curriculum": cfg["curriculum"],
                    "evaluation_suite": "unchanged 38-fixture native long-history robustness suite",
                    "selector_or_migration_used": False,
                    "retention_gate": "heldout correction_accuracy == 1.0 and control_accuracy == 1.0",
                    "reference_0p4b_v1": "11/13 supersession, 30/38 overall, correction/control 100% at epoch 1",
                },
                "epoch_mean_loss": losses,
                "checkpoints": checkpoints,
                "summary": {
                    "raw_best": compact(raw_best),
                    "retention_qualified_best": compact(safe_best),
                    "hits_13_of_13_retention_safe": bool(safe_best and safe_best["superseded_full_success"] == 13),
                    "beats_v1_11_of_13": bool(safe_best and safe_best["superseded_full_success"] > 11),
                },
                "claim_boundary": "Five-arm local tuning on a repeatedly inspected synthetic suite. A 13/13 result would justify moving to chained migration tests, not prove general semantic overwrite capability.",
            }
            write_report(report)

        del model
        gc.collect()
    except BaseException as exc:
        write_report({
            "status": "VELA_RWKV7_0P4B_NARROW_SWEEP_V2_ERROR",
            "arm": ARM,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-120:],
        })
        raise


if __name__ == "__main__":
    run()
