from __future__ import annotations

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

ROB_PATH = BASE / "long-history-native-robustness-v1" / "mamba_long_history_native_robustness.py"
spec2 = importlib.util.spec_from_file_location("vela_mamba_longhist", ROB_PATH)
if spec2 is None or spec2.loader is None:
    raise RuntimeError(f"cannot load {ROB_PATH}")
rob = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(rob)
FIXTURES = rob.FIXTURES


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def score_fixture(model, ns, tok, fx):
    ids = tok.encode("".join(fx["segments"]))
    with torch.no_grad():
        _, state = rw.run_tokens(model, ns, ids, rw.zero_state(model.args, model))
        state = rw.clone_state(state)
    probes = []
    for pid, suffix, candidates, expected in fx["probes"]:
        scores = {c: rw.candidate_score(model, ns, tok, state, suffix, c) for c in candidates}
        chosen = max(scores, key=scores.get)
        probes.append({
            "id": pid,
            "chosen": chosen,
            "expected": expected,
            "correct": chosen == expected,
            "scores": scores,
        })
    acc = sum(int(r["correct"]) for r in probes) / max(len(probes), 1)
    return {
        "history_tokens": len(ids),
        "events": len(fx["segments"]),
        "expected_accuracy": acc,
        "all_correct": acc == 1.0,
        "probe_results": probes,
    }


def summarize(rows):
    by_family = {}
    for row in rows:
        fam = row["family"]
        slot = by_family.setdefault(fam, {"count": 0, "full_success": 0, "failed": []})
        slot["count"] += 1
        ok = row["native"]["all_correct"]
        slot["full_success"] += int(ok)
        if not ok:
            slot["failed"].append({
                "fixture": row["fixture"],
                "history_tokens": row["native"]["history_tokens"],
                "wrong_probes": [p["id"] for p in row["native"]["probe_results"] if not p["correct"]],
            })
    for slot in by_family.values():
        slot["full_success_rate"] = slot["full_success"] / max(slot["count"], 1)
    return by_family


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

            # Recheck the state-continuity property on a real fixture before semantic scoring.
            probe_fx = FIXTURES[0]
            probe_ids = tok.encode("".join(probe_fx["segments"]))
            cut = max(1, len(probe_ids) // 2)
            with torch.no_grad():
                _, checkpoint = rw.run_tokens(model, ns, probe_ids[:cut], rw.zero_state(args, model))
                checkpoint = rw.clone_state(checkpoint)
                _, native_final = rw.run_tokens(model, ns, probe_ids[cut:], rw.clone_state(checkpoint))
                _, restored_final = rw.run_tokens(model, ns, probe_ids[cut:], rw.clone_state(checkpoint))
                _, replay_final = rw.run_tokens(model, ns, probe_ids, rw.zero_state(args, model))
            restore_diff = rw.state_distance(native_final, restored_final)
            replay_diff = rw.state_distance(native_final, replay_final)

            rows = []
            for fx in FIXTURES:
                rows.append({
                    "fixture": fx["id"],
                    "family": rob.family_name(fx["id"]),
                    "native": score_fixture(model, ns, tok, fx),
                })

            by_family = summarize(rows)
            success_count = sum(int(r["native"]["all_correct"]) for r in rows)
            report = {
                "status": "VELA_RWKV7_NATIVE_LONG_HISTORY_V1",
                "source_commit": rw.SOURCE_COMMIT,
                "weight_revision": rw.WEIGHT_REVISION,
                "weight_file": rw.WEIGHT_FILE,
                "device": "cpu",
                "dtype": "float32",
                "design": {
                    "purpose": "Candidate-backbone spec/continuity recheck plus native full-history semantics on the unchanged 38-fixture suite.",
                    "selector_or_migration_used": False,
                    "training_used": False,
                },
                "state_continuity": {
                    "checkpoint_cut_tokens": cut,
                    "restore_max_abs_diff": restore_diff["max_abs"],
                    "full_replay_max_abs_diff": replay_diff["max_abs"],
                    "restore_equivalent": restore_diff["max_abs"] == 0.0,
                    "full_replay_equivalent": replay_diff["max_abs"] == 0.0,
                    "state_numel": restore_diff["numel"],
                },
                "suite": {
                    "fixture_count": len(rows),
                    "full_success_count": success_count,
                    "failure_count": len(rows) - success_count,
                    "by_family": by_family,
                },
                "fixtures": rows,
                "claim_boundary": "Released RWKV-7 0.1B weights, pinned official RNN equations, CPU float32, deterministic synthetic histories. Native candidate diagnostic only; no learned adaptation yet.",
            }
            write_report(report)
    except BaseException as exc:
        write_report({
            "status": "VELA_RWKV7_NATIVE_LONG_HISTORY_V1_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-100:],
        })
        raise


if __name__ == "__main__":
    run()
