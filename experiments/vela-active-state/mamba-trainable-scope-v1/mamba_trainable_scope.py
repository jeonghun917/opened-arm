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
SCOPES = ["xproj_only", "all_mixer"]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def configure_scope(model, scope):
    trainable = []
    names = []
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        ok = False
        if scope == "xproj_only":
            ok = ".mixer.x_proj.weight" in name
        elif scope == "all_mixer":
            ok = ".mixer." in name
        else:
            raise ValueError(scope)
        if ok:
            p.requires_grad_(True)
            trainable.append(p)
            names.append(name)
    if not trainable:
        raise RuntimeError(f"no trainable parameters for scope={scope}")
    return trainable, names


def train_scope(scope, tok, AutoModelForCausalLM):
    torch.manual_seed(rob.v3.SEED)
    random.seed(rob.v3.SEED)
    model = AutoModelForCausalLM.from_pretrained(
        rob.v3.MODEL_ID, torch_dtype=torch.float32
    ).cpu().eval()
    trainable, names = configure_scope(model, scope)
    optimizer = torch.optim.AdamW(trainable, lr=rob.v3.LR, weight_decay=0.0)
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
    result = {
        "scope": scope,
        "trainable_tensor_count": len(trainable),
        "trainable_numel": int(sum(p.numel() for p in trainable)),
        "example_trainable_names": names[:24],
        "train_examples_per_epoch": len(TRAIN_ROWS),
        "batch_size": 1,
        "epochs": EPOCHS,
        "optimizer_steps": steps,
        "learning_rate": rob.v3.LR,
        "epoch_mean_loss": losses,
        "heldout": {
            "accuracy": heldout["accuracy"],
            "correction_accuracy": heldout["correction_accuracy"],
            "control_accuracy": heldout["control_accuracy"],
        },
        "long_history": long_history,
        "superseded_full_success": int(sup.get("full_success", 0)),
        "superseded_count": int(sup.get("count", 0)),
    }
    del optimizer
    del model
    gc.collect()
    return result


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        tok = AutoTokenizer.from_pretrained(rob.v3.MODEL_ID)
        arms = {scope: train_scope(scope, tok, AutoModelForCausalLM) for scope in SCOPES}
        x = arms["xproj_only"]
        m = arms["all_mixer"]
        if m["superseded_full_success"] > x["superseded_full_success"]:
            signal = "broader_recurrent_update_recovers_supersession"
        elif m["long_history"]["full_success_count"] > x["long_history"]["full_success_count"]:
            signal = "broader_recurrent_update_improves_other_history_but_not_supersession"
        else:
            signal = "no_recovery_from_mixer_scope_expansion"

        write_report({
            "status": "VELA_MAMBA_TRAINABLE_SCOPE_V1",
            "model": rob.v3.MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "design": {
                "question": "Is the supersession weakness caused by x_proj-only adaptation being too narrow?",
                "curriculum": "same 16-example three-stage supersession curriculum used in supersession-curriculum-v1",
                "scopes": SCOPES,
                "epochs_per_scope": EPOCHS,
                "examples_per_epoch": len(TRAIN_ROWS),
                "batch_size": 1,
                "optimizer_steps_per_scope": EPOCHS * len(TRAIN_ROWS),
                "evaluation_suite": "unchanged 38-fixture native long-history robustness suite",
                "selector_or_migration_used": False,
            },
            "arms": arms,
            "summary": {
                "xproj_superseded_full_success": x["superseded_full_success"],
                "all_mixer_superseded_full_success": m["superseded_full_success"],
                "superseded_count": x["superseded_count"],
                "xproj_overall_full_success": x["long_history"]["full_success_count"],
                "all_mixer_overall_full_success": m["long_history"]["full_success_count"],
                "scope_signal": signal,
            },
            "interpretation_rule": {
                "supports_xproj_bottleneck": "all_mixer improves the held-out superseded family over xproj_only under the same curriculum and step budget.",
                "does_not_support_xproj_bottleneck_yet": "all_mixer remains at xproj_only on supersession; this rejects only the simple x_proj-only bottleneck hypothesis and does not prove Mamba cannot learn supersession with broader or full-model adaptation.",
            },
            "claim_boundary": "Single Mamba-130M, synthetic 16-example supersession curriculum, 160 optimizer steps per scope, and the existing 38-fixture native-history suite. This is a parameter-scope diagnostic, not a final architecture verdict.",
        })
    except BaseException as exc:
        write_report({
            "status": "VELA_MAMBA_TRAINABLE_SCOPE_V1_ERROR",
            "model": getattr(rob.v3, "MODEL_ID", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-80:],
        })
        raise


if __name__ == "__main__":
    run()
