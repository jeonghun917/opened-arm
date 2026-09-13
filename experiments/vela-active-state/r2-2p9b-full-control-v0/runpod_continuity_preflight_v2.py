#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MODEL_REPO = "BlinkDL/rwkv7-g1"
MODEL_REVISION = "2448c93da77d9958ea89a9faa4c1b34c5c46ccec"
MODEL_FILE = "rwkv7-g1a-2.9b-20250924-ctx4096.pth"
MODEL_SIZE = 5896274949
MODEL_SHA256 = "df5716263b617e7da83590446fb2a98b6663447cfd52e8e9b49e11ce7f4faa3e"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_state(state):
    return [x.detach().clone() for x in state]


def state_meta(state) -> list[dict]:
    return [{
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "stride": list(x.stride()),
        "contiguous": bool(x.is_contiguous()),
    } for x in state]


def compare_tensor(a, b) -> dict:
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"compatible": False, "exact": False, "max_abs": None, "mean_abs": None, "nonzero_count": None}
    af = a.detach().float()
    bf = b.detach().float()
    diff = (af - bf).abs()
    return {
        "compatible": True,
        "exact": bool(torch.equal(a, b)),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "nonzero_count": int(torch.count_nonzero(diff).item()),
        "numel": int(diff.numel()),
    }


def compare_state(a, b) -> dict:
    if len(a) != len(b):
        return {"compatible": False, "exact": False, "tensor_count_a": len(a), "tensor_count_b": len(b)}
    per = [compare_tensor(x, y) for x, y in zip(a, b)]
    return {
        "compatible": all(x["compatible"] for x in per),
        "exact": all(x["exact"] for x in per),
        "tensor_count": len(per),
        "differing_tensor_count": sum(1 for x in per if not x["exact"]),
        "total_nonzero_count": sum(int(x.get("nonzero_count") or 0) for x in per),
        "max_abs": max((float(x.get("max_abs") or 0.0) for x in per), default=0.0),
    }


def pair(logits_a, state_a, logits_b, state_b) -> dict:
    return {"logits": compare_tensor(logits_a, logits_b), "state": compare_state(state_a, state_b)}


def pair_exact(x: dict) -> bool:
    return bool(x["logits"]["exact"] and x["state"]["exact"])


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download

    if torch.__version__ != "2.7.1+cu126":
        raise RuntimeError(f"torch drift: {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    os.environ["RWKV_V7_ON"] = "1"
    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "0"
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    path = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, revision=MODEL_REVISION))
    actual_sha = sha256_file(path)
    if path.stat().st_size != MODEL_SIZE or actual_sha != MODEL_SHA256:
        raise RuntimeError(f"model identity mismatch size={path.stat().st_size} sha={actual_sha}")

    model = RWKV(model=str(path)[:-4], strategy="cuda fp16")
    pipe = PIPELINE(model, "rwkv_vocab_v20230424")
    zero = model.generate_zero_state()

    # Neutral runtime warm-up. These outputs are discarded and never become evidence.
    warm_seed = [int(x) for x in pipe.encode("VELA CONTINUITY PREFLIGHT V2 NEUTRAL WARMUP")]
    warm_cont = [int(x) for x in pipe.encode(" CONTINUATION CHECK")]
    _, warm_state = model.forward(warm_seed, clone_state(zero))
    _, _ = model.forward(warm_cont, clone_state(warm_state))
    torch.cuda.synchronize()

    seed_tokens = [int(x) for x in pipe.encode("VELA RUNPOD STATE ROUNDTRIP NEUTRAL")]
    _, state = model.forward(seed_tokens, clone_state(zero))
    torch.cuda.synchronize()

    out_dir = Path(os.environ.get("VELA_CONTINUITY_PREFLIGHT_DIR", "/workspace/vela"))
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = out_dir / "runpod-continuity-preflight-v2-state.pt"
    torch.save(state, snapshot)
    restored = torch.load(snapshot, weights_only=False)

    serialization = compare_state(state, restored)
    metadata_equal = state_meta(state) == state_meta(restored)
    continuation = [int(x) for x in pipe.encode(" CONTINUATION CHECK")]

    def run_once(src_state):
        logits, out_state = model.forward(continuation, clone_state(src_state))
        torch.cuda.synchronize()
        return logits.detach().clone(), clone_state(out_state)

    # Symmetric sacrificial calls. Both are discarded before measured comparisons.
    run_once(state)
    run_once(restored)

    # Measured ABBA order controls residual call-order effects.
    original_a_logits, original_a_state = run_once(state)
    restored_a_logits, restored_a_state = run_once(restored)
    restored_b_logits, restored_b_state = run_once(restored)
    original_b_logits, original_b_state = run_once(state)

    comparisons = {
        "original_repeat": pair(original_a_logits, original_a_state, original_b_logits, original_b_state),
        "restored_repeat": pair(restored_a_logits, restored_a_state, restored_b_logits, restored_b_state),
        "cross_first": pair(original_a_logits, original_a_state, restored_a_logits, restored_a_state),
        "cross_second": pair(original_b_logits, original_b_state, restored_b_logits, restored_b_state),
    }
    all_exact = all(pair_exact(v) for v in comparisons.values())
    passed = bool(serialization.get("exact") and metadata_equal and all_exact)

    report = {
        "schema": "vela-runpod-continuity-preflight-result:v2",
        "status": "PASS" if passed else "FAIL",
        "classification": "POST_WARMUP_ORDER_CONTROL_EXACT" if passed else "POST_WARMUP_ORDER_CONTROL_NONEXACT",
        "scientific_evidence": False,
        "source_commit": os.environ.get("VELA_SOURCE_COMMIT"),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "strategy": "cuda fp16",
            "rwkv_cuda_kernel": False,
            "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "model": {"file": MODEL_FILE, "size": path.stat().st_size, "sha256": actual_sha},
        "warmup": {
            "neutral_runtime_warmup": True,
            "sacrificial_order": ["original", "restored"],
            "measured_order": ["original_a", "restored_a", "restored_b", "original_b"],
        },
        "snapshot": {
            "bytes": snapshot.stat().st_size,
            "sha256": sha256_file(snapshot),
            "serialization_exact": bool(serialization.get("exact")),
            "metadata_equal": metadata_equal,
            "tensor_count": len(state),
            "state_bytes": int(sum(x.numel() * x.element_size() for x in state)),
        },
        "comparisons": comparisons,
    }
    (out_dir / "runpod-continuity-preflight-v2.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    summary = {
        "schema": report["schema"],
        "status": report["status"],
        "classification": report["classification"],
        "scientific_evidence": False,
        "source_commit": report["source_commit"],
        "device": report["runtime"]["device"],
        "torch": report["runtime"]["torch"],
        "model_sha256": actual_sha,
        "serialization_exact": report["snapshot"]["serialization_exact"],
        "metadata_equal": metadata_equal,
        "original_repeat_exact": pair_exact(comparisons["original_repeat"]),
        "restored_repeat_exact": pair_exact(comparisons["restored_repeat"]),
        "cross_first_exact": pair_exact(comparisons["cross_first"]),
        "cross_second_exact": pair_exact(comparisons["cross_second"]),
        "max_abs": {k: max(float(v["logits"].get("max_abs") or 0.0), float(v["state"].get("max_abs") or 0.0)) for k, v in comparisons.items()},
    }
    print("VELA_CONTINUITY_PREFLIGHT_V2_SUMMARY=" + json.dumps(summary, separators=(",", ":"), sort_keys=True), flush=True)
    if not passed:
        raise SystemExit("continuity preflight v2 failed")


if __name__ == "__main__":
    main()
