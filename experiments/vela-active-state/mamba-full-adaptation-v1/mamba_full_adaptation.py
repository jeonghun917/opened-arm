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
CURR_PATH = BASE / "mamba-supersession-curriculum-v1" / "mamba_supersession_curriculum.py"
spec = importlib.util.spec_from_file_location("vela_sup_curr_v1", CURR_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {CURR_PATH}")
curr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curr)
rob = curr.rob

EPOCHS = 10
TRAIN_ROWS = list(curr.SUP_TRAIN)
LR = 1e-5


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        torch.manual_seed(rob.v3.SEED)
        random.seed(rob.v3.SEED)
        torch.set_num_threads(min(4, os.cpu_count() or 2))

        tok = AutoTokenizer.from_pretrained(rob.v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            rob.v3.MODEL_ID, torch_dtype=torch.float32
        ).cpu().eval()

        trainable = []
        names = []
        for name, p in model.named_parameters():
            p.requires_grad_(True)
            trainable.append(p)
            names.append(name)

        optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
        order = list(range(len(TRAIN_ROWS)))
        losses = []
        steps = 0

        for epoch in range(EPOCHS):
            random.Random(rob.v3.SEED + epoch).shuffle(order)
            model.train()
            total = 0.0
            for idx in order:
                prompt, gold = TRAIN_ROWS[idx]
                optimizer.zero_grad(set_to_none=True)
                loss = rob.v3.train_loss(model, tok, prompt, gold)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                total += float(loss.detach())
                steps += 1
            model.eval()
            losses.append(total / max(len(order), 1))

        heldout = rob.v3.evaluate(model, tok)
        long_history = curr.evaluate_suite(model, tok)
        sup = long_history["by_family"].get("superseded", {})

        report = {
            "status": "VELA_MAMBA_FULL_ADAPTATION_V1",
            "model": rob.v3.MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "design": {
                "question": "Does full-model adaptation recover held-out semantic supersession after x_proj-only and all-mixer failed?",
                "curriculum": "same 16-example three-stage supersession curriculum used in supersession-curriculum-v1",
                "trainable_scope": "all model parameters",
                "trainable_tensor_count": len(trainable),
                "trainable_numel": int(sum(p.numel() for p in trainable)),
                "epochs": EPOCHS,
                "examples_per_epoch": len(TRAIN_ROWS),
                "batch_size": 1,
                "optimizer_steps": steps,
                "learning_rate": LR,
                "evaluation_suite": "unchanged 38-fixture native long-history robustness suite",
                "selector_or_migration_used": False,
            },
            "epoch_mean_loss": losses,
            "heldout": {
                "accuracy": heldout["accuracy"],
                "correction_accuracy": heldout["correction_accuracy"],
                "control_accuracy": heldout["control_accuracy"],
            },
            "long_history": long_history,
            "summary": {
                "superseded_full_success": int(sup.get("full_success", 0)),
                "superseded_count": int(sup.get("count", 0)),
                "overall_full_success": int(long_history["full_success_count"]),
                "fixture_count": int(long_history["fixture_count"]),
            },
            "interpretation_rule": {
                "recovery": "A clear held-out supersession gain supports the hypothesis that earlier trainable scopes were too restrictive.",
                "no_recovery": "No held-out supersession gain under full-model adaptation strengthens, but does not by itself prove, a backbone/scale or training-distribution limitation.",
            },
            "claim_boundary": "Single Mamba-130M model, 16 synthetic supersession examples, 160 optimizer steps, LR=1e-5, CPU execution, and the existing 38-fixture suite. Not a final architecture verdict.",
        }
        write_report(report)

        del optimizer
        del model
        gc.collect()
    except BaseException as exc:
        write_report({
            "status": "VELA_MAMBA_FULL_ADAPTATION_V1_ERROR",
            "model": getattr(rob.v3, "MODEL_ID", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-100:],
        })
        raise


if __name__ == "__main__":
    run()
