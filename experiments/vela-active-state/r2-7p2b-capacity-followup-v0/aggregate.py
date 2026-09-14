from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

BASE_AGGREGATE_BLOB = "ac86908be386b0159f52bcf45d4dd0e0786daae4"
BASE_AGGREGATE_PATH = Path(__file__).resolve().parent.parent / "r2-2p9b-full-control-v0" / "aggregate.py"
CONDITIONS = ["B", "C", "G", "M", "X"]
SCALES = ["2.9B", "7.2B"]
PRIMARY_SCALE = "7.2B"
BASELINE_SCALE = "2.9B"

def _gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()

_raw = BASE_AGGREGATE_PATH.read_bytes()
_actual = _gitblob(_raw)
if _actual != BASE_AGGREGATE_BLOB:
    raise RuntimeError(f"frozen P-R2-02 aggregate drift: {_actual}")

_spec = importlib.util.spec_from_file_location("_vela_pr202_aggregate", BASE_AGGREGATE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load frozen P-R2-02 aggregate")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

IDENTITY = _base.IDENTITY
classify = _base.classify
signature = _base.signature
delta = _base.delta

HISTORICAL_PR202 = {
    "scientific_candidate": "a7ca8b9cd429262edf550eccbf406c7cb308138d",
    "workflow_run_id": 34792771868,
    "aggregate_artifact_id": 10327744795,
    "primary_hypothesis_status": "MIXED",
    "model_profile": "g1a-20250924",
    "M_minus_X_correct_count": 0,
    "M_minus_X_margin_sum": 0.5201422113459557,
    "episode5_M_minus_X_margin": 0.21704092272557318,
    "episode6_M_minus_X_margin": 0.30310128862038255,
}

def _scale_summary(shards: dict[str, dict], scale: str) -> dict:
    r = {c: shards[c]["scales"][scale] for c in CONDITIONS}
    sig = signature(r["B"])
    equal = all(signature(r[c]) == sig for c in CONDITIONS[1:])
    m, x = r["M"], r["X"]
    pri = {
        "episode5_security": {"M": m["prior_selected_state_sha256_trace"][4], "X": x["prior_selected_state_sha256_trace"][4]},
        "episode6_runtime": {"M": m["prior_selected_state_sha256_trace"][5], "X": x["prior_selected_state_sha256_trace"][5]},
    }
    ident = pri["episode5_security"]["M"] != pri["episode5_security"]["X"] and pri["episode6_runtime"]["M"] != pri["episode6_runtime"]["X"]
    status = classify(r, equal, ident)
    return {
        "conditions": r,
        "equal_external_workflow": equal,
        "route_identifiability_ok": ident,
        "return_prior_state_digests": pri,
        "hypothesis_status": status,
        "protocol_ok": all(r[c]["protocol_ok"] for c in CONDITIONS) and equal and ident,
        "primary": {
            "M_minus_X_correct_count": int(m["proposal_correct_count"]) - int(x["proposal_correct_count"]),
            "M_minus_X_margin_sum": float(m["expected_margin_sum"]) - float(x["expected_margin_sum"]),
            "security_return_episode5_M_minus_X": delta(r, "M", "X", 4),
            "runtime_return_episode6_M_minus_X": delta(r, "M", "X", 5),
        },
        "secondary": {
            "M_minus_G_correct_count": int(m["proposal_correct_count"]) - int(r["G"]["proposal_correct_count"]),
            "M_minus_G_margin_sum": float(m["expected_margin_sum"]) - float(r["G"]["expected_margin_sum"]),
            "correction_episode4_M_minus_C": delta(r, "M", "C", 3),
            "security_return_episode5_M_minus_G": delta(r, "M", "G", 4),
            "runtime_return_episode6_M_minus_G": delta(r, "M", "G", 5),
        },
    }

def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    shards: dict[str, dict] = {}
    for c in CONDITIONS:
        matches = list(root.rglob(f"condition_result_{c}.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one shard for {c}, got {matches}")
        d = json.loads(matches[0].read_text())
        if d.get("status") != "P-R2-02_CONDITION_RESULT" or d.get("condition") != c or not d.get("scientific_evidence") or d.get("input_identity") != IDENTITY:
            raise RuntimeError(f"invalid shard {c}")
        if set(d.get("scales", {})) != set(SCALES):
            raise RuntimeError(f"scale set mismatch in {c}: {set(d.get('scales', {}))}")
        shards[c] = d
    commits = {d.get("source_commit") for d in shards.values()}
    if len(commits) != 1 or None in commits:
        raise RuntimeError(f"source commit mismatch {commits}")
    runtimes = [d["runtime"] for d in shards.values()]
    if any(x != runtimes[0] for x in runtimes[1:]):
        raise RuntimeError("runtime identity differs across shards")
    scales = {scale: _scale_summary(shards, scale) for scale in SCALES}
    a = scales[BASELINE_SCALE]["primary"]
    b = scales[PRIMARY_SCALE]["primary"]
    out = {
        "status": "P-R2-03_RESULT",
        "scientific_evidence": True,
        "scientific_matrix_run_ordinal": 1,
        "source_commit": next(iter(commits)),
        "runtime": runtimes[0],
        "input_identity": IDENTITY,
        "shard_conditions": CONDITIONS,
        "matched_model_profile": "g1i-20260805",
        "scales": scales,
        "historical_p_r2_02_reference": HISTORICAL_PR202,
        "scale_interaction": {
            "M_minus_X_hard_delta_7p2b_minus_2p9b": int(b["M_minus_X_correct_count"]) - int(a["M_minus_X_correct_count"]),
            "M_minus_X_margin_delta_7p2b_minus_2p9b": float(b["M_minus_X_margin_sum"]) - float(a["M_minus_X_margin_sum"]),
            "ep5_M_minus_X_margin_delta_7p2b_minus_2p9b": float(b["security_return_episode5_M_minus_X"]["margin"]) - float(a["security_return_episode5_M_minus_X"]["margin"]),
            "ep6_M_minus_X_margin_delta_7p2b_minus_2p9b": float(b["runtime_return_episode6_M_minus_X"]["margin"]) - float(a["runtime_return_episode6_M_minus_X"]["margin"]),
        },
        "primary_hypothesis_status": scales[PRIMARY_SCALE]["hypothesis_status"],
        "protocol_ok": bool(scales[BASELINE_SCALE]["protocol_ok"] and scales[PRIMARY_SCALE]["protocol_ok"]),
        "claim_boundary": "P-R2-03 primary classifier applies to matched-profile g1i 7.2B. Matched-profile g1i 2.9B is the strict scale baseline. P-R2-02 g1a 2.9B is historical context only and does not define the scale delta.",
        "inherited_shard_schema": "P-R2-02_CONDITION_RESULT",
        "frozen_classifier_source_blob": BASE_AGGREGATE_BLOB,
    }
    Path("scientific_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
