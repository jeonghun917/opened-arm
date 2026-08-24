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
RW_PATH = BASE / "rwkv7-learned-upgrade-v1" / "rwkv7_learned_upgrade.py"
spec = importlib.util.spec_from_file_location("vela_rwkv7_learned", RW_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {RW_PATH}")
rw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rw)

NATIVE_PATH = BASE / "rwkv7-native-long-history-v1" / "rwkv7_native_long_history.py"
spec2 = importlib.util.spec_from_file_location("vela_rwkv7_native", NATIVE_PATH)
if spec2 is None or spec2.loader is None:
    raise RuntimeError(f"cannot load {NATIVE_PATH}")
native = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(native)

CURR_PATH = BASE / "mamba-supersession-curriculum-v1" / "mamba_supersession_curriculum.py"
spec3 = importlib.util.spec_from_file_location("vela_sup_curr", CURR_PATH)
if spec3 is None or spec3.loader is None:
    raise RuntimeError(f"cannot load {CURR_PATH}")
curr = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(curr)

ROB_PATH = BASE / "long-history-native-robustness-v1" / "mamba_long_history_native_robustness.py"
spec4 = importlib.util.spec_from_file_location("vela_longhist", ROB_PATH)
if spec4 is None or spec4.loader is None:
    raise RuntimeError(f"cannot load {ROB_PATH}")
rob = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(rob)

EPOCHS = 5
LR = rw.LR
CHECKPOINT_EPOCHS = {0, 1, 3, 5}
TRAIN = list(curr.SUP_TRAIN)


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def score_candidate_from_logits(model, ns, tok, base_out, base_state, candidate):
    cids = tok.encode(candidate)
    if not cids:
        raise RuntimeError("empty candidate")
    out = base_out
    st = rw.clone_state(base_state)
    total = 0.0
    with torch.no_grad():
        for j, tid in enumerate(cids):
            total += float(torch.log_softmax(out.float(), dim=-1)[tid])
            if j + 1 < len(cids):
                out, st = rw.run_tokens(model, ns, [tid], st)
    return total


def evaluate_heldout(model, ns, tok):
    rows = []
    kinds = {"correction": [0, 0], "control": [0, 0]}
    for fx in rw.HELDOUT:
        pids = tok.encode(fx["prompt"])
        with torch.no_grad():
            out, st = rw.run_tokens(model, ns, pids, rw.zero_state(model.args, model))
            st = rw.clone_state(st)
        if out is None:
            raise RuntimeError(f"empty prompt for {fx['id']}")
        scores = {
            c: score_candidate_from_logits(model, ns, tok, out, st, c)
            for c in fx["candidates"]
        }
        chosen = max(scores, key=scores.get)
        ok = chosen == fx["expected"]
        kinds[fx["kind"]][0] += int(ok)
        kinds[fx["kind"]][1] += 1
        rows.append({"id": fx["id"], "chosen": chosen, "expected": fx["expected"], "correct": ok, "scores": scores})
    correct = sum(int(r["correct"]) for r in rows)
    return {
        "accuracy": correct / max(len(rows), 1),
        "correction_accuracy": kinds["correction"][0] / max(kinds["correction"][1], 1),
        "control_accuracy": kinds["control"][0] / max(kinds["control"][1], 1),
        "rows": rows,
    }


def evaluate_suite(model, ns, tok):
    rows = []
    for fx in rob.FIXTURES:
        out = native.score_fixture(model, ns, tok, fx)
        rows.append({
            "fixture": fx["id"],
            "family": rob.family_name(fx["id"]),
            "history_tokens": out["history_tokens"],
            "all_correct": bool(out["all_correct"]),
            "expected_accuracy": out["expected_accuracy"],
            "wrong_probes": [p["id"] for p in out["probe_results"] if not p["correct"]],
        })
    by_family = {}
    for row in rows:
        slot = by_family.setdefault(row["family"], {"count": 0, "full_success": 0})
        slot["count"] += 1
        slot["full_success"] += int(row["all_correct"])
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


def summarize_checkpoint(epoch, model, ns, tok):
    heldout = evaluate_heldout(model, ns, tok)
    suite = evaluate_suite(model, ns, tok)
    return {
        "epoch": epoch,
        "optimizer_steps": epoch * len(TRAIN),
        "heldout": {
            "accuracy": heldout["accuracy"],
            "correction_accuracy": heldout["correction_accuracy"],
            "control_accuracy": heldout["control_accuracy"],
        },
        "long_history": suite,
    }


def run():
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

            trainable = []
            for p in model.z.values():
                if isinstance(p, torch.Tensor):
                    p.requires_grad_(False)
            for i in range(args.n_layer):
                key = f"blocks.{i}.att.key.weight"
                model.z[key].requires_grad_(True)
                trainable.append(model.z[key])

            opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
            order = list(range(len(TRAIN)))
            checkpoints = [summarize_checkpoint(0, model, ns, tok)]
            losses = []

            for epoch in range(1, EPOCHS + 1):
                random.Random(rw.SEED + epoch).shuffle(order)
                total = 0.0
                for idx in order:
                    opt.zero_grad(set_to_none=True)
                    loss = rw.train_example(model, ns, tok, *TRAIN[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                    total += float(loss.detach())
                losses.append(total / max(len(order), 1))
                if epoch in CHECKPOINT_EPOCHS:
                    checkpoints.append(summarize_checkpoint(epoch, model, ns, tok))

            superseded_curve = []
            overall_curve = []
            for cp in checkpoints:
                sup = cp["long_history"]["by_family"]["superseded"]
                superseded_curve.append({
                    "epoch": cp["epoch"],
                    "full_success": sup["full_success"],
                    "count": sup["count"],
                })
                overall_curve.append({
                    "epoch": cp["epoch"],
                    "full_success": cp["long_history"]["full_success_count"],
                    "count": cp["long_history"]["fixture_count"],
                })

            report = {
                "status": "VELA_RWKV7_SUPERSESSION_ADAPTATION_V1",
                "source_commit": rw.SOURCE_COMMIT,
                "weight_revision": rw.WEIGHT_REVISION,
                "weight_file": rw.WEIGHT_FILE,
                "device": "cpu",
                "dtype": "float32",
                "design": {
                    "question": "Can a released RWKV-7 0.1B model learn held-out three-stage semantic supersession under a known-working recurrent update scope?",
                    "curriculum": "same 16-example three-stage supersession curriculum used for Mamba diagnostics",
                    "trainable_scope": "blocks.*.att.key.weight",
                    "trainable_numel": int(sum(p.numel() for p in trainable)),
                    "epochs": EPOCHS,
                    "examples_per_epoch": len(TRAIN),
                    "batch_size": 1,
                    "optimizer_steps": EPOCHS * len(TRAIN),
                    "learning_rate": LR,
                    "evaluation_suite": "unchanged 38-fixture native long-history robustness suite",
                    "selector_or_migration_used": False,
                },
                "epoch_mean_loss": losses,
                "checkpoints": checkpoints,
                "summary": {
                    "superseded_curve": superseded_curve,
                    "overall_curve": overall_curve,
                    "recovery_observed": any(x["full_success"] > 0 for x in superseded_curve[1:]),
                },
                "claim_boundary": "RWKV-7 0.1B, one recurrent parameter family, 80 optimizer steps, synthetic supersession curriculum. This is a candidate-backbone teachability diagnostic, not a final architecture verdict.",
            }
            write_report(report)

        del model
        gc.collect()
    except BaseException as exc:
        write_report({
            "status": "VELA_RWKV7_SUPERSESSION_ADAPTATION_V1_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-100:],
        })
        raise


if __name__ == "__main__":
    run()
