from __future__ import annotations

import gc
import importlib.util
import json
import os
import random
import re
import resource
import tempfile
import traceback
import urllib.request
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
CORE_FIXTURE_IDS = (
    "superseded_base",
    "superseded_suffix4",
    "superseded_suffix8",
    "superseded_old_value_echoes",
)
HORIZONS = (128, 512, 2048)
MAX_HORIZON = max(HORIZONS)
FUTURE_STREAMS = (
    {"id": "telemetry", "template": "Neutral telemetry packet {i:04d} was archived. Sensor checksum marker {j:02d} was logged. "},
    {"id": "inventory", "template": "Routine inventory entry {i:04d} was filed. Auxiliary schedule marker {j:02d} was recorded. "},
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


selv3 = load_module(
    "vela_selector_v3_for_2p9b_full",
    BASE / "g7-selector-v3" / "rwkv7_0p4b_g7_selector_v3.py",
)
g7 = selv3.g7
chain = g7.chain

# 2.9B released checkpoint used by the focused rescue probe.
chain.scale.WEIGHT_REPO = "BlinkDL/rwkv7-g1"
chain.scale.WEIGHT_REVISION = "ede85bf8ab2e59aff7d7ca909fbbc73317866d89"
chain.scale.WEIGHT_FILE = "rwkv7-g1i-2.9b-20260805-ctx16384.pth"
chain.scale.N_LAYER = 32
chain.scale.N_EMBD = 2560
chain.scale.HEAD_SIZE = 64
chain.scale.VOCAB_SIZE = 65536


def rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def trim_heap() -> None:
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_future(tok, stream: dict, token_count: int):
    chunks = []
    ids = []
    i = 0
    while len(ids) < token_count:
        chunks.append(stream["template"].format(i=i, j=i % 97))
        ids = tok.encode("".join(chunks))
        i += 1
    return ids[:token_count], {"id": stream["id"], "chunks_generated": i, "tokens_used": token_count}


def expected_accuracy(rows) -> float:
    return sum(int(x["correct"]) for x in rows) / max(len(rows), 1)


def lowpeak_train(model, ns, tok, trainable, rows, lr, seed_offset, stage):
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0, foreach=False)
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
        emit({"heartbeat": stage, "step": n, "steps": len(order), "loss": float(loss.detach()), "max_rss_gib": round(rss_gib(), 3)})
    mean = total / max(len(rows), 1)
    del opt
    trim_heap()
    emit({"stage_complete": stage, "mean_loss": mean, "max_rss_gib": round(rss_gib(), 3)})
    return mean


def fixture_payload(tok, fx):
    ids = tok.encode("".join(fx["segments"]))
    starts, ends = chain.ca.boundaries(tok, fx["segments"])
    positions = sorted(set(starts + ends + [0, len(ids)]))
    return ids, starts, ends, positions


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def save_kvr_sharded(model, args, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in chain.kvr_names(args):
        fn = safe_name(name) + ".pt"
        torch.save(model.z[name].detach(), out_dir / fn)
        manifest[name] = fn
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_kvr_sharded(model, in_dir: Path) -> dict:
    manifest = json.loads((in_dir / "manifest.json").read_text(encoding="utf-8"))
    with torch.no_grad():
        for n, (name, fn) in enumerate(manifest.items(), start=1):
            tensor = torch.load(in_dir / fn, map_location="cpu", weights_only=True)
            model.z[name].copy_(tensor)
            del tensor
            if n % 12 == 0:
                trim_heap()
                emit({"heartbeat": "load_w3_kvr", "loaded": n, "total": len(manifest), "max_rss_gib": round(rss_gib(), 3)})
    return manifest


def configure_2p9b():
    torch.manual_seed(chain.rw.SEED)
    random.seed(chain.rw.SEED)
    torch.set_num_threads(2)


def load_model_and_tokenizer():
    from huggingface_hub import hf_hub_download
    weight_path = hf_hub_download(
        repo_id=chain.scale.WEIGHT_REPO,
        filename=chain.scale.WEIGHT_FILE,
        revision=chain.scale.WEIGHT_REVISION,
    )
    model, args, ns = chain.scale.load_reference_scaled(weight_path)
    td = tempfile.TemporaryDirectory()
    vocab_path = Path(td.name) / "vocab.txt"
    urllib.request.urlretrieve(chain.rw.VOCAB_URL, vocab_path)
    tok = chain.rw.RWKVTokenizer(str(vocab_path))
    return model, args, ns, tok, td


def phase_build(work_dir: Path):
    configure_2p9b()
    model, args, ns, tok, td = load_model_and_tokenizer()
    try:
        emit({"stage": "build_start", "max_rss_gib": round(rss_gib(), 3)})
        fixtures = [fx for fx in chain.FIXTURES if fx["id"] in CORE_FIXTURE_IDS]
        if [fx["id"] for fx in fixtures] != list(CORE_FIXTURE_IDS):
            raise RuntimeError(f"fixture order mismatch: {[fx['id'] for fx in fixtures]}")

        w1_dir = work_dir / "w1"
        w2_dir = work_dir / "w2"
        w1_dir.mkdir(parents=True, exist_ok=True)
        w2_dir.mkdir(parents=True, exist_ok=True)

        # W1 lineages are produced once, serialized, then released from RAM.
        for fx in fixtures:
            ids, starts, ends, positions = fixture_payload(tok, fx)
            states = {}
            for p in positions:
                states[p] = chain.ca.state_at(model, ns, ids[:p], args)
            payload = {
                "fixture": fx["id"], "ids": ids, "starts": starts, "ends": ends,
                "lineage": states, "origins": {p: "W1" for p in positions},
            }
            torch.save(payload, w1_dir / f"{fx['id']}.pt")
            del payload, states
            trim_heap()
            emit({"stage": "w1_lineage_saved", "fixture": fx["id"], "max_rss_gib": round(rss_gib(), 3)})

        trainable = chain.scale.configure_kvr(model, args)
        trainable_numel = int(sum(p.numel() for p in trainable))
        w2_loss = lowpeak_train(model, ns, tok, trainable, chain.W2_ROWS, chain.W2_LR, 101, "W2")
        model.eval()

        # Create actual carried W2 lineages while the model is W2; no W2 KVR snapshot is retained.
        for fx in fixtures:
            p = torch.load(w1_dir / f"{fx['id']}.pt", map_location="cpu", weights_only=False)
            ids, starts, ends = p["ids"], p["starts"], p["ends"]
            specs = fx["probes"]
            w1_lineage, w1_origins = p["lineage"], p["origins"]
            w2_native = chain.ca.state_at(model, ns, ids, args)
            w2_native_rows = chain.ca.score_specs(model, ns, tok, w2_native, specs)
            hop1 = chain.target_free_select(model, ns, tok, ids, starts, ends, w1_lineage, specs, fx["segments"])
            hop1_comp = chain.compare(hop1["rows"], w2_native_rows)
            w2_lineage, w2_origins, _ = chain.rebuild_lineage(
                model, ns, ids, starts, ends, w1_lineage, w1_origins, hop1["chosen_seg"], "W2"
            )
            out = {
                "fixture": fx["id"], "ids": ids, "starts": starts, "ends": ends,
                "lineage": w2_lineage, "origins": w2_origins,
                "hop1": {
                    "chosen_seg": hop1["chosen_seg"],
                    "anchor_pos": hop1["anchor_pos"],
                    "decision_agreement": hop1_comp["decision_agreement"],
                    "expected_accuracy": hop1_comp.get("expected_accuracy"),
                },
            }
            torch.save(out, w2_dir / f"{fx['id']}.pt")
            del p, w1_lineage, w1_origins, w2_lineage, w2_origins, out, w2_native, w2_native_rows, hop1
            (w1_dir / f"{fx['id']}.pt").unlink(missing_ok=True)
            trim_heap()
            emit({"stage": "w2_lineage_saved", "fixture": fx["id"], "max_rss_gib": round(rss_gib(), 3)})

        w3_loss = lowpeak_train(model, ns, tok, trainable, chain.W3_ROWS, chain.W3_LR, 202, "W3")
        model.eval()
        manifest = save_kvr_sharded(model, args, work_dir / "w3_kvr")
        meta = {
            "status": "VELA_G7_RWKV7_2P9B_FULL_SELECTOR_V3_ADAPTATION_BUILD_V1",
            "source_commit": os.environ.get("GITHUB_SHA"),
            "actual_checked_out_head": os.environ.get("VELA_CHECKED_OUT_HEAD"),
            "wrapper_blob": os.environ.get("VELA_WRAPPER_BLOB"),
            "model": {"repo": chain.scale.WEIGHT_REPO, "revision": chain.scale.WEIGHT_REVISION, "file": chain.scale.WEIGHT_FILE},
            "trainable_numel": trainable_numel,
            "w2_mean_loss": w2_loss,
            "w3_mean_loss": w3_loss,
            "w3_kvr_shard_count": len(manifest),
            "fixture_ids": list(CORE_FIXTURE_IDS),
            "max_rss_gib": rss_gib(),
            "adaptation_reuse": "W2/W3 trained exactly once; W2 carried lineages and W3 KVR are serialized for evaluation shards",
        }
        write_json(work_dir / "build_meta.json", meta)
        emit(meta)
    finally:
        td.cleanup()
        del model
        trim_heap()


def eval_fixture(model, args, ns, tok, work_dir: Path, fx):
    p = torch.load(work_dir / "w2" / f"{fx['id']}.pt", map_location="cpu", weights_only=False)
    ids, starts, ends = p["ids"], p["starts"], p["ends"]
    w2_lineage, w2_origins = p["lineage"], p["origins"]
    specs = fx["probes"]

    # Re-register carried-generation provenance after deserialization so chain-v2 and selector-v3 audit it.
    g7.v2._ORIGINS_BY_LINEAGE_ID[id(w2_lineage)] = dict(w2_origins)

    w3_native = chain.ca.state_at(model, ns, ids, args)
    w3_native_rows = chain.ca.score_specs(model, ns, tok, w3_native, specs)
    hop2 = chain.target_free_select(model, ns, tok, ids, starts, ends, w2_lineage, specs, fx["segments"])
    hop2_comp = chain.compare(hop2["rows"], w3_native_rows)
    chosen_seg = int(hop2["chosen_seg"])
    chosen_pos = starts[chosen_seg]
    chosen_origin = w2_origins[chosen_pos]
    _, _, selected_state = chain.rebuild_lineage(
        model, ns, ids, starts, ends, w2_lineage, w2_origins, chosen_seg, "W3"
    )
    selected_rows = chain.ca.score_specs(model, ns, tok, selected_state, specs)
    selected_comp = chain.compare(selected_rows, w3_native_rows)
    initial_ok = (
        expected_accuracy(w3_native_rows) == 1.0
        and p["hop1"]["decision_agreement"] == 1.0
        and hop2_comp["decision_agreement"] == 1.0
        and selected_comp["decision_agreement"] == 1.0
        and chosen_origin == "W2"
    )

    initial = {
        "qualified": initial_ok,
        "w3_native_expected_accuracy": expected_accuracy(w3_native_rows),
        "hop1_decision_agreement": p["hop1"]["decision_agreement"],
        "hop2_decision_agreement": hop2_comp["decision_agreement"],
        "selected_final_decision_agreement": selected_comp["decision_agreement"],
        "selected_anchor_seg": chosen_seg,
        "selected_anchor_pos": chosen_pos,
        "selected_anchor_origin": chosen_origin,
        "selector_v3": hop2.get("selector_v3"),
        "provenance_tiebreak": hop2.get("provenance_tiebreak"),
    }

    cases = []
    for stream in FUTURE_STREAMS:
        future_ids, meta = build_future(tok, stream, MAX_HORIZON)
        roll = g7.roll_path(model, ns, tok, selected_state, w3_native, specs, future_ids)
        native_stable = bool(roll["native_stable"])
        migrated_stable = bool(roll["path_stable_vs_native"])
        cases.append({
            "fixture": fx["id"],
            "future_stream": stream["id"],
            "future_meta": meta,
            "initial": initial,
            "selected": roll,
            "native_stable_case": native_stable,
            "migration_excess_failure": bool(initial_ok and native_stable and not migrated_stable),
        })
        emit({"heartbeat": "fixture_stream_complete", "fixture": fx["id"], "stream": stream["id"], "native_stable": native_stable, "migrated_stable": migrated_stable, "max_rss_gib": round(rss_gib(), 3)})

    del p, w2_lineage, w2_origins, w3_native, selected_state
    trim_heap()
    return {"fixture": fx["id"], "initial_qualified": initial_ok, "cases": cases}


def phase_eval(work_dir: Path, out_path: Path, fixture_ids: list[str]):
    configure_2p9b()
    model, args, ns, tok, td = load_model_and_tokenizer()
    try:
        load_kvr_sharded(model, work_dir / "w3_kvr")
        fixtures = {fx["id"]: fx for fx in chain.FIXTURES}
        rows = []
        for fid in fixture_ids:
            rows.append(eval_fixture(model, args, ns, tok, work_dir, fixtures[fid]))
        report = {
            "status": "VELA_G7_RWKV7_2P9B_FULL_SELECTOR_V3_EVAL_SHARD_V1",
            "source_commit": os.environ.get("GITHUB_SHA"),
            "actual_checked_out_head": os.environ.get("VELA_CHECKED_OUT_HEAD"),
            "wrapper_blob": os.environ.get("VELA_WRAPPER_BLOB"),
            "fixture_ids": fixture_ids,
            "rows": rows,
            "max_rss_gib": rss_gib(),
        }
        write_json(out_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        td.cleanup()
        del model
        trim_heap()


def phase_aggregate(shard_dir: Path, out_path: Path):
    rows = []
    source_commits = set()
    heads = set()
    wrappers = set()
    for p in sorted(shard_dir.glob("**/*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("status") != "VELA_G7_RWKV7_2P9B_FULL_SELECTOR_V3_EVAL_SHARD_V1":
            continue
        source_commits.add(obj.get("source_commit"))
        heads.add(obj.get("actual_checked_out_head"))
        wrappers.add(obj.get("wrapper_blob"))
        rows.extend(obj["rows"])
    if {r["fixture"] for r in rows} != set(CORE_FIXTURE_IDS):
        raise RuntimeError(f"aggregate fixture mismatch: {[r['fixture'] for r in rows]}")
    cases = [c for r in rows for c in r["cases"]]
    immediate_qualified = sum(int(r["initial_qualified"]) for r in rows)
    native_stable_cases = [c for c in cases if c["native_stable_case"]]
    migration_excess = [c for c in cases if c["migration_excess_failure"]]
    per_fixture_native_stable = {
        fid: sum(int(c["native_stable_case"]) for c in cases if c["fixture"] == fid)
        for fid in CORE_FIXTURE_IDS
    }
    native_coverage_gate = len(native_stable_cases) >= 6 and all(v >= 1 for v in per_fixture_native_stable.values())
    migration_excess_gate = len(migration_excess) == 0
    formal_pass_candidate = immediate_qualified == 4 and native_coverage_gate and migration_excess_gate
    report = {
        "status": "VELA_G7_RWKV7_2P9B_FULL_SELECTOR_V3_V1",
        "source_commits": sorted(x for x in source_commits if x),
        "actual_checked_out_heads": sorted(x for x in heads if x),
        "wrapper_blobs": sorted(x for x in wrappers if x),
        "model": {"repo": chain.scale.WEIGHT_REPO, "revision": chain.scale.WEIGHT_REVISION, "file": chain.scale.WEIGHT_FILE},
        "protocol": {
            "fixtures": list(CORE_FIXTURE_IDS),
            "future_streams": [x["id"] for x in FUTURE_STREAMS],
            "horizons": list(HORIZONS),
            "selector": "selector-v3 earliest carried-W2 immediate-functional-equivalence guard",
            "adaptation_reuse": "single W2/W3 adaptation build reused by all fixture evaluations",
            "oracle_usage": "W3-native/future trajectory evaluation-only; not used by selector",
        },
        "rows": rows,
        "summary": {
            "fixture_count": 4,
            "case_count": len(cases),
            "initial_chain_qualified": immediate_qualified,
            "native_stable_case_count": len(native_stable_cases),
            "per_fixture_native_stable": per_fixture_native_stable,
            "migration_only_excess_failure_count": len(migration_excess),
            "native_coverage_gate": native_coverage_gate,
            "migration_excess_gate": migration_excess_gate,
            "formal_g7_2p9b_pass_candidate": formal_pass_candidate,
        },
        "claim_boundary": "Full 4-fixture x 2-stream synthetic G7 qualification on one released RWKV-7 2.9B checkpoint using the preregistered W2/W3 KVR adaptation and selector-v3 guard. Passing is evidence for this checkpoint/protocol and does not freeze final backbone, selector formula, state schema, or production hardware.",
    }
    write_json(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    phase = os.environ.get("VELA_PHASE", "")
    work_dir = Path(os.environ.get("VELA_WORK_DIR", "/tmp/vela-g7-2p9b-full-v1"))
    try:
        if phase == "build":
            phase_build(work_dir)
        elif phase == "eval":
            ids = [x for x in os.environ["VELA_FIXTURE_IDS"].split(",") if x]
            phase_eval(work_dir, Path(os.environ["VELA_RESULT_PATH"]), ids)
        elif phase == "aggregate":
            phase_aggregate(Path(os.environ["VELA_SHARD_DIR"]), Path(os.environ["VELA_RESULT_PATH"]))
        else:
            raise RuntimeError(f"unknown VELA_PHASE={phase!r}")
    except BaseException as exc:
        err = {
            "status": "VELA_G7_RWKV7_2P9B_FULL_SELECTOR_V3_V1_ERROR",
            "phase": phase,
            "source_commit": os.environ.get("GITHUB_SHA"),
            "actual_checked_out_head": os.environ.get("VELA_CHECKED_OUT_HEAD"),
            "wrapper_blob": os.environ.get("VELA_WRAPPER_BLOB"),
            "scientific_failure": False if phase != "aggregate" else None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "max_rss_gib": rss_gib(),
            "traceback_tail": traceback.format_exc().splitlines()[-120:],
        }
        pth = os.environ.get("VELA_RESULT_PATH")
        if pth:
            write_json(Path(pth), err)
        print(json.dumps(err, ensure_ascii=False, indent=2), flush=True)
        raise


if __name__ == "__main__":
    main()
