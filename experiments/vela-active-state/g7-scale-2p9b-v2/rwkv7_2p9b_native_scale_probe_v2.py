from __future__ import annotations

import ctypes
import gc
import importlib.util
import json
import os
import random
import resource
import tempfile
import traceback
import urllib.request
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
HORIZONS = (128, 512, 2048)
MAX_HORIZON = max(HORIZONS)
FUTURE_STREAMS = (
    {
        "id": "telemetry",
        "template": (
            "Neutral telemetry packet {i:04d} was archived. "
            "Sensor checksum marker {j:02d} was logged. "
        ),
    },
    {
        "id": "inventory",
        "template": (
            "Routine inventory entry {i:04d} was filed. "
            "Auxiliary schedule marker {j:02d} was recorded. "
        ),
    },
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chain = load_module(
    "vela_rwkv7_chain_v1_for_2p9b_native_v2",
    BASE / "rwkv7-0p4b-chain-v1" / "rwkv7_0p4b_chained_upgrade.py",
)

chain.scale.WEIGHT_REPO = "BlinkDL/rwkv7-g1"
chain.scale.WEIGHT_REVISION = "ede85bf8ab2e59aff7d7ca909fbbc73317866d89"
chain.scale.WEIGHT_FILE = "rwkv7-g1i-2.9b-20260805-ctx16384.pth"
chain.scale.N_LAYER = 32
chain.scale.N_EMBD = 2560
chain.scale.HEAD_SIZE = 64
chain.scale.VOCAB_SIZE = 65536

FIXTURE = chain.ROB_BY_ID["superseded_old_value_echoes"]


def write_report(report: dict) -> None:
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def rss_gib() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def trim_heap() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def train_stage_lowpeak(model, ns, tok, trainable, rows, lr, seed_offset, stage_name):
    # Same AdamW/KVR adaptation semantics as chain-v1. foreach=False avoids the
    # tensor-list peak allocation on a memory-constrained CPU runner.
    opt = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=0.0,
        foreach=False,
    )
    order = list(range(len(rows)))
    random.Random(chain.rw.SEED + seed_offset).shuffle(order)
    total = 0.0
    for n, idx in enumerate(order, start=1):
        opt.zero_grad(set_to_none=True)
        loss = chain.rw.train_example(model, ns, tok, *rows[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        total += float(loss.detach())
        print(
            json.dumps(
                {
                    "heartbeat": stage_name,
                    "step": n,
                    "steps": len(order),
                    "loss": float(loss.detach()),
                    "max_rss_gib": round(rss_gib(), 3),
                }
            ),
            flush=True,
        )
    mean_loss = total / max(len(order), 1)
    del opt
    trim_heap()
    print(
        json.dumps(
            {
                "stage_complete": stage_name,
                "mean_loss": mean_loss,
                "max_rss_gib": round(rss_gib(), 3),
            }
        ),
        flush=True,
    )
    return mean_loss


def expected_accuracy(rows) -> float:
    return sum(int(row["correct"]) for row in rows) / max(len(rows), 1)


def functional_flags(rows) -> dict:
    by_id = {row["id"]: row for row in rows}

    def correct(pid: str):
        row = by_id.get(pid)
        return None if row is None else bool(row["correct"])

    correction_ok = correct("codeword")
    control_vals = [correct(pid) for pid in ("project", "action")]
    persistent_vals = [correct(pid) for pid in ("verification", "project", "action")]
    control_known = [x for x in control_vals if x is not None]
    persistent_known = [x for x in persistent_vals if x is not None]
    return {
        "task_invariant_ok": all(bool(row["correct"]) for row in rows),
        "correction_ok": correction_ok,
        "control_ok": None if not control_known else all(control_known),
        "persistent_fact_ok": None if not persistent_known else all(persistent_known),
    }


def expected_margin(row):
    scores = row.get("scores") or {}
    expected = row.get("expected")
    if expected not in scores or len(scores) < 2:
        return None
    return float(scores[expected] - max(v for k, v in scores.items() if k != expected))


def margin_summary(rows) -> dict:
    margins = {row["id"]: expected_margin(row) for row in rows}
    known = [x for x in margins.values() if x is not None]
    return {
        "per_probe_expected_margin": margins,
        "min_expected_margin": None if not known else min(known),
        "mean_expected_margin": None if not known else sum(known) / len(known),
    }


def build_future(tok, stream: dict, token_count: int):
    chunks = []
    ids = []
    i = 0
    while len(ids) < token_count:
        chunks.append(stream["template"].format(i=i, j=i % 97))
        ids = tok.encode("".join(chunks))
        i += 1
    return ids[:token_count], {
        "id": stream["id"],
        "chunks_generated": i,
        "tokens_used": token_count,
        "semantic_exclusion": "templates avoid fixture project/codeword/verification/action/correction terms",
    }


def roll_native(model, ns, tok, initial_state, specs, future_ids):
    state = chain.clone_state(initial_state)
    milestones = {}
    for step, tid in enumerate(future_ids, start=1):
        with torch.no_grad():
            _, state = chain.rw.run_tokens(model, ns, [tid], state)
        if step in HORIZONS:
            rows = chain.ca.score_specs(model, ns, tok, state, specs)
            milestones[str(step)] = {
                "native_expected_accuracy": expected_accuracy(rows),
                "native_flags": functional_flags(rows),
                "native_margin": margin_summary(rows),
            }
            print(
                json.dumps(
                    {
                        "heartbeat": "native_roll",
                        "step": step,
                        "expected_accuracy": milestones[str(step)]["native_expected_accuracy"],
                        "task_invariant_ok": milestones[str(step)]["native_flags"]["task_invariant_ok"],
                        "max_rss_gib": round(rss_gib(), 3),
                    }
                ),
                flush=True,
            )
    stable = all(milestones[str(h)]["native_flags"]["task_invariant_ok"] for h in HORIZONS)
    return {"native_stable": stable, "milestones": milestones}


def run() -> None:
    try:
        from huggingface_hub import hf_hub_download

        torch.manual_seed(chain.rw.SEED)
        random.seed(chain.rw.SEED)
        torch.set_num_threads(2)

        print(
            json.dumps(
                {
                    "stage": "start",
                    "checked_out_head": os.environ.get("VELA_CHECKED_OUT_HEAD"),
                    "github_sha": os.environ.get("GITHUB_SHA"),
                    "wrapper_blob": os.environ.get("VELA_WRAPPER_BLOB"),
                    "max_rss_gib": round(rss_gib(), 3),
                }
            ),
            flush=True,
        )

        weight_path = hf_hub_download(
            repo_id=chain.scale.WEIGHT_REPO,
            filename=chain.scale.WEIGHT_FILE,
            revision=chain.scale.WEIGHT_REVISION,
        )
        print(json.dumps({"stage": "weight_downloaded", "path": weight_path}), flush=True)
        model, args, ns = chain.scale.load_reference_scaled(weight_path)
        print(
            json.dumps(
                {"stage": "model_loaded", "max_rss_gib": round(rss_gib(), 3)},
            ),
            flush=True,
        )

        with tempfile.TemporaryDirectory() as td:
            vocab_path = Path(td) / "vocab.txt"
            urllib.request.urlretrieve(chain.rw.VOCAB_URL, vocab_path)
            tok = chain.rw.RWKVTokenizer(str(vocab_path))

            trainable = chain.scale.configure_kvr(model, args)
            trainable_numel = int(sum(p.numel() for p in trainable))
            print(
                json.dumps(
                    {
                        "stage": "trainable_configured",
                        "trainable_numel": trainable_numel,
                        "max_rss_gib": round(rss_gib(), 3),
                    }
                ),
                flush=True,
            )

            w2_loss = train_stage_lowpeak(
                model, ns, tok, trainable, chain.W2_ROWS, chain.W2_LR, 101, "W2"
            )
            model.eval()
            w3_loss = train_stage_lowpeak(
                model, ns, tok, trainable, chain.W3_ROWS, chain.W3_LR, 202, "W3"
            )
            model.eval()

            ids = tok.encode("".join(FIXTURE["segments"]))
            specs = FIXTURE["probes"]
            native_state = chain.ca.state_at(model, ns, ids, args)
            immediate_rows = chain.ca.score_specs(model, ns, tok, native_state, specs)
            immediate_accuracy = expected_accuracy(immediate_rows)
            immediate_flags = functional_flags(immediate_rows)
            immediate_qualified = immediate_accuracy == 1.0 and immediate_flags["task_invariant_ok"]

            cases = []
            for stream in FUTURE_STREAMS:
                future_ids, meta = build_future(tok, stream, MAX_HORIZON)
                roll = roll_native(model, ns, tok, native_state, specs, future_ids)
                cases.append(
                    {
                        "fixture": FIXTURE["id"],
                        "future_stream": stream["id"],
                        "future_meta": meta,
                        "native_stable_case": bool(immediate_qualified and roll["native_stable"]),
                        "native_roll": roll,
                    }
                )

            native_stable_count = sum(int(x["native_stable_case"]) for x in cases)
            report = {
                "status": "VELA_G7_RWKV7_2P9B_NATIVE_SCALE_PROBE_V2",
                "source_commit": os.environ.get("GITHUB_SHA"),
                "actual_checked_out_head": os.environ.get("VELA_CHECKED_OUT_HEAD"),
                "wrapper_blob": os.environ.get("VELA_WRAPPER_BLOB"),
                "device": "cpu",
                "dtype": "float32",
                "model": {
                    "repo": chain.scale.WEIGHT_REPO,
                    "revision": chain.scale.WEIGHT_REVISION,
                    "file": chain.scale.WEIGHT_FILE,
                    "layers": chain.scale.N_LAYER,
                    "embedding_dim": chain.scale.N_EMBD,
                    "head_size": chain.scale.HEAD_SIZE,
                    "vocab_size": chain.scale.VOCAB_SIZE,
                },
                "protocol": {
                    "fixture": FIXTURE["id"],
                    "future_streams": [x["id"] for x in FUTURE_STREAMS],
                    "horizons": list(HORIZONS),
                    "adaptation": "same sequential W2 sup-heavy then W3 status8 KVR-only AdamW recipe as chain-v1",
                    "optimizer_low_peak_change": "AdamW foreach=False; algorithm/learning-rate/rows/order/clip/weight-decay unchanged",
                    "removed_as_irrelevant_to_native_scale_readout": "W1/W2 lineage construction, W1/W2/W3 KVR RAM snapshots, migration/selector evaluation",
                    "trainable_numel": trainable_numel,
                    "w2_lr": chain.W2_LR,
                    "w3_lr": chain.W3_LR,
                    "w2_examples": len(chain.W2_ROWS),
                    "w3_examples": len(chain.W3_ROWS),
                },
                "training": {
                    "w2_mean_loss": w2_loss,
                    "w3_mean_loss": w3_loss,
                    "max_rss_gib": rss_gib(),
                },
                "immediate_native": {
                    "expected_accuracy": immediate_accuracy,
                    "flags": immediate_flags,
                    "margin": margin_summary(immediate_rows),
                    "qualified": immediate_qualified,
                },
                "cases": cases,
                "summary": {
                    "native_stable_case_count": native_stable_count,
                    "native_stable_case_total": len(cases),
                    "immediate_native_qualified": immediate_qualified,
                },
                "scale_probe": {
                    "reference_0p4b_native_stable_streams": "0/2",
                    "reference_1p5b_native_stable_streams": "0/2",
                    "decision_rule": {
                        "2/2": "strong focused scale evidence; consider full 2.9B G7 qualification",
                        "1/2": "partial scale benefit; inspect horizon and margins",
                        "0/2": "no focused native-stability benefit under this adaptation recipe",
                    },
                },
                "claim_boundary": (
                    "Focused 2.9B CPU native-only scale diagnostic. The W2/W3 KVR AdamW adaptation recipe and preregistered "
                    "old_value_echoes/telemetry/inventory/128-512-2048 readout are preserved, while migration-only allocations are removed. "
                    "No paid GPU and no final-backbone freeze."
                ),
            }
            write_report(report)

        del model
        trim_heap()
    except BaseException as exc:
        write_report(
            {
                "status": "VELA_G7_RWKV7_2P9B_NATIVE_SCALE_PROBE_V2_ERROR",
                "source_commit": os.environ.get("GITHUB_SHA"),
                "actual_checked_out_head": os.environ.get("VELA_CHECKED_OUT_HEAD"),
                "wrapper_blob": os.environ.get("VELA_WRAPPER_BLOB"),
                "scientific_failure": False,
                "failure_class": "runtime_before_scientific_result",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "max_rss_gib": rss_gib(),
                "traceback_tail": traceback.format_exc().splitlines()[-120:],
            }
        )
        raise


if __name__ == "__main__":
    run()
