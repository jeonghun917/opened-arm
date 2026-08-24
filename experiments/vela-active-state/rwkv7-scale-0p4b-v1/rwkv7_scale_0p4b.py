from __future__ import annotations

import gc
import importlib.util
import json
import os
import random
import re
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

WEIGHT_REPO = "BlinkDL/rwkv7-g1"
WEIGHT_REVISION = "4aa0981b66af4f727181a147bcea1457ad7c84cb"
WEIGHT_FILE = "rwkv7-g1a-0.4b-20250905-ctx4096.pth"
N_LAYER = 24
N_EMBD = 1024
HEAD_SIZE = 64
VOCAB_SIZE = 65536
LR = 1e-5
EPOCHS = 5
CHECKPOINT_EPOCHS = {0, 1, 3, 5}
TRAIN = list(v2.BALANCED_TRAIN)  # exact v2 winning 8 supersession + 4 correction + 4 control


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_reference_scaled(weight_path):
    source = urllib.request.urlopen(rw.SOURCE_URL, timeout=60).read().decode("utf-8")
    source = source.split("# RWKV Tokenizer (slow version)")[0]
    source = source.replace("MyModule = torch.jit.ScriptModule", "MyModule = nn.Module")
    source = source.replace("MyFunction = torch.jit.script_method", "MyFunction = (lambda f: f)")
    source = source.replace("MyStatic = torch.jit.script", "MyStatic = (lambda f: f)")
    source = source.replace("DTYPE = torch.half # better", "DTYPE = torch.float32")
    source = source.replace("map_location='cuda'", "map_location='cpu'")
    source = re.sub(
        r"try:\n\s*time_mixing = torch\.compile\(time_mixing__,.*?\nexcept:\n\s*time_mixing = torch\.jit\.script\(time_mixing__\)",
        "time_mixing = time_mixing__",
        source,
        flags=re.S,
    )
    source = re.sub(
        r"try:\n\s*channel_mixing = torch\.compile\(channel_mixing__,.*?\nexcept:\n\s*channel_mixing = torch\.jit\.script\(channel_mixing__\)",
        "channel_mixing = channel_mixing__",
        source,
        flags=re.S,
    )
    ns = {"__name__": "rwkv7_grad_reference_scaled"}
    exec(compile(source, rw.SOURCE_PATH, "exec"), ns, ns)
    args = ns["args"]
    args.MODEL_NAME = weight_path.removesuffix(".pth")
    args.n_layer = N_LAYER
    args.n_embd = N_EMBD
    args.vocab_size = VOCAB_SIZE
    args.head_size = HEAD_SIZE
    ns["DTYPE"] = torch.float32
    model = ns["RWKV_RNN"](args).cpu().eval()
    return model, args, ns


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


def summarize(epoch, steps, model, ns, tok):
    heldout = v1.evaluate_heldout(model, ns, tok)
    suite = v1.evaluate_suite(model, ns, tok)
    sup = suite["by_family"]["superseded"]
    safe = heldout["correction_accuracy"] == 1.0 and heldout["control_accuracy"] == 1.0
    return {
        "epoch": epoch,
        "optimizer_steps": steps,
        "heldout": {
            "accuracy": heldout["accuracy"],
            "correction_accuracy": heldout["correction_accuracy"],
            "control_accuracy": heldout["control_accuracy"],
        },
        "retention_qualified": safe,
        "superseded_full_success": sup["full_success"],
        "superseded_count": sup["count"],
        "long_history": suite,
    }


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
    try:
        from huggingface_hub import hf_hub_download

        torch.manual_seed(rw.SEED)
        random.seed(rw.SEED)
        torch.set_num_threads(2)

        weight_path = hf_hub_download(
            repo_id=WEIGHT_REPO,
            filename=WEIGHT_FILE,
            revision=WEIGHT_REVISION,
        )
        model, args, ns = load_reference_scaled(weight_path)

        with tempfile.TemporaryDirectory() as td:
            vp = Path(td) / "vocab.txt"
            urllib.request.urlretrieve(rw.VOCAB_URL, vp)
            tok = rw.RWKVTokenizer(str(vp))

            trainable = configure_kvr(model, args)
            optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
            order = list(range(len(TRAIN)))
            checkpoints = [summarize(0, 0, model, ns, tok)]
            losses = []
            steps = 0

            for epoch in range(1, EPOCHS + 1):
                random.Random(rw.SEED + epoch).shuffle(order)
                total = 0.0
                for idx in order:
                    optimizer.zero_grad(set_to_none=True)
                    loss = rw.train_example(model, ns, tok, *TRAIN[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    total += float(loss.detach())
                    steps += 1
                losses.append(total / len(order))
                if epoch in CHECKPOINT_EPOCHS:
                    checkpoints.append(summarize(epoch, steps, model, ns, tok))

            safe_best = best(checkpoints, require_retention=True)
            raw_best = best(checkpoints, require_retention=False)
            report = {
                "status": "VELA_RWKV7_SCALE_0P4B_V1",
                "source_commit": rw.SOURCE_COMMIT,
                "weight_repo": WEIGHT_REPO,
                "weight_revision": WEIGHT_REVISION,
                "weight_file": WEIGHT_FILE,
                "device": "cpu",
                "dtype": "float32",
                "model_shape": {"layers": N_LAYER, "embedding_dim": N_EMBD, "head_size": HEAD_SIZE},
                "design": {
                    "question": "Does scaling RWKV-7 from 0.1B to 0.4B improve retention-safe semantic supersession under the same winning KVR recipe?",
                    "curriculum": "same v2 winning 8 supersession + 4 correction + 4 control examples",
                    "trainable_scope": "blocks.*.att.{key,value,receptance}.weight",
                    "trainable_numel": int(sum(p.numel() for p in trainable)),
                    "learning_rate": LR,
                    "epochs": EPOCHS,
                    "examples_per_epoch": len(TRAIN),
                    "optimizer_steps": steps,
                    "checkpoint_epochs": sorted(CHECKPOINT_EPOCHS),
                    "evaluation_suite": "unchanged 38-fixture native long-history robustness suite",
                    "selector_or_migration_used": False,
                    "retention_gate": "heldout correction_accuracy == 1.0 and control_accuracy == 1.0",
                    "reference_0p1b": "v2 winner = 3/13 supersession, 9/38 overall, correction/control 100%",
                },
                "epoch_mean_loss": losses,
                "checkpoints": checkpoints,
                "summary": {
                    "raw_best": compact(raw_best),
                    "retention_qualified_best": compact(safe_best),
                    "beats_0p1b_v2_supersession_3_of_13": bool(safe_best and safe_best["superseded_full_success"] > 3),
                },
                "claim_boundary": "Single released RWKV-7 0.4B checkpoint, synthetic curriculum, CPU float32, 80 KVR-only optimizer steps and the fixed 38-fixture suite. This is a scale diagnostic, not final backbone promotion evidence.",
            }
            write_report(report)

        del model
        gc.collect()
    except BaseException as exc:
        write_report({
            "status": "VELA_RWKV7_SCALE_0P4B_V1_ERROR",
            "weight_revision": WEIGHT_REVISION,
            "weight_file": WEIGHT_FILE,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-120:],
        })
        raise


if __name__ == "__main__":
    run()
