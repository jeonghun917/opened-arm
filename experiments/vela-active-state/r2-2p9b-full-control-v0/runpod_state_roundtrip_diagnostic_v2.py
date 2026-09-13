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


def tensor_meta(x) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "stride": list(x.stride()),
        "contiguous": bool(x.is_contiguous()),
    }


def compare_tensor(a, b) -> dict:
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"compatible": False, "exact": False}
    af = a.detach().float()
    bf = b.detach().float()
    diff = (af - bf).abs()
    denom = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-12)
    rel = diff / denom
    nonzero = int(torch.count_nonzero(diff).item())
    return {
        "compatible": True,
        "exact": bool(torch.equal(a, b)),
        "nonzero_count": nonzero,
        "numel": int(a.numel()),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
    }


def compare_state(a, b) -> dict:
    if len(a) != len(b):
        return {"compatible": False, "exact": False, "tensor_count_a": len(a), "tensor_count_b": len(b)}
    per = [compare_tensor(x, y) for x, y in zip(a, b)]
    return {
        "compatible": all(x.get("compatible") is True for x in per),
        "exact": all(x.get("exact") is True for x in per),
        "tensor_count": len(per),
        "differing_tensor_count": sum(1 for x in per if x.get("exact") is not True),
        "total_nonzero_count": sum(int(x.get("nonzero_count") or 0) for x in per),
        "max_abs": max((float(x.get("max_abs") or 0.0) for x in per), default=0.0),
        "max_rel": max((float(x.get("max_rel") or 0.0) for x in per), default=0.0),
        "per_tensor": per,
    }


def compact_state_comparison(x: dict) -> dict:
    return {k: v for k, v in x.items() if k != "per_tensor"}


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
    seed_tokens = [int(x) for x in pipe.encode("VELA RUNPOD STATE ROUNDTRIP NEUTRAL")]
    _, state = model.forward(seed_tokens, clone_state(zero))
    torch.cuda.synchronize()

    out_dir = Path(os.environ.get("VELA_DIAGNOSTIC_DIR", "/workspace/vela"))
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = out_dir / "runpod-state-roundtrip-diagnostic-v2.pt"
    torch.save(state, snapshot)
    restored = torch.load(snapshot, weights_only=False)

    serialization = compare_state(state, restored)
    metadata_equal = len(state) == len(restored) and all(tensor_meta(a) == tensor_meta(b) for a, b in zip(state, restored))
    continuation = [int(x) for x in pipe.encode(" CONTINUATION CHECK")]

    def run_once(src_state):
        logits, out_state = model.forward(continuation, clone_state(src_state))
        torch.cuda.synchronize()
        return logits.detach().clone(), clone_state(out_state)

    original_1_logits, original_1_state = run_once(state)
    original_2_logits, original_2_state = run_once(state)
    restored_1_logits, restored_1_state = run_once(restored)
    restored_2_logits, restored_2_state = run_once(restored)

    comparisons = {
        "original_repeat": {
            "logits": compare_tensor(original_1_logits, original_2_logits),
            "state": compare_state(original_1_state, original_2_state),
        },
        "restored_repeat": {
            "logits": compare_tensor(restored_1_logits, restored_2_logits),
            "state": compare_state(restored_1_state, restored_2_state),
        },
        "original_vs_restored_first": {
            "logits": compare_tensor(original_1_logits, restored_1_logits),
            "state": compare_state(original_1_state, restored_1_state),
        },
        "original_vs_restored_second": {
            "logits": compare_tensor(original_2_logits, restored_2_logits),
            "state": compare_state(original_2_state, restored_2_state),
        },
        "original_second_vs_restored_first": {
            "logits": compare_tensor(original_2_logits, restored_1_logits),
            "state": compare_state(original_2_state, restored_1_state),
        },
    }

    def exact_pair(name: str) -> bool:
        c = comparisons[name]
        return c["logits"]["exact"] and c["state"]["exact"]

    original_repeat_exact = exact_pair("original_repeat")
    restored_repeat_exact = exact_pair("restored_repeat")
    cross_first_exact = exact_pair("original_vs_restored_first")
    cross_second_exact = exact_pair("original_vs_restored_second")
    post_first_cross_exact = exact_pair("original_second_vs_restored_first")

    if serialization.get("exact") is not True:
        classification = "SERIALIZATION_DEFECT"
    elif (not original_repeat_exact and restored_repeat_exact and cross_second_exact and post_first_cross_exact):
        classification = "FIRST_CONTINUATION_WARMUP_OR_RUNTIME_ORDER_EFFECT"
    elif not original_repeat_exact:
        classification = "BASE_RUNTIME_NONDETERMINISM_OR_MODEL_MUTATION"
    elif not restored_repeat_exact:
        classification = "RESTORED_RUNTIME_NONDETERMINISM"
    elif cross_first_exact:
        classification = "NO_DIVERGENCE_REPRODUCED"
    else:
        classification = "RESTORE_SPECIFIC_EXECUTION_DIVERGENCE"

    runtime = {
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
    }
    model_info = {"file": MODEL_FILE, "size": path.stat().st_size, "sha256": actual_sha}
    snapshot_info = {
        "bytes": snapshot.stat().st_size,
        "sha256": sha256_file(snapshot),
        "serialization": serialization,
        "metadata_equal": metadata_equal,
        "original_meta": [tensor_meta(x) for x in state],
        "restored_meta": [tensor_meta(x) for x in restored],
    }
    report = {
        "status": "DIAGNOSTIC_COMPLETE",
        "scientific_evidence": False,
        "purpose": "result-blind diagnosis of RunPod recurrent-state continuation non-exactness",
        "source_commit": os.environ.get("VELA_SOURCE_COMMIT"),
        "classification": classification,
        "runtime": runtime,
        "model": model_info,
        "snapshot": snapshot_info,
        "comparisons": comparisons,
    }
    out = out_dir / "runpod-state-roundtrip-diagnostic-v2.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    summary = {
        "status": report["status"],
        "scientific_evidence": False,
        "source_commit": report["source_commit"],
        "classification": classification,
        "runtime": runtime,
        "model": model_info,
        "snapshot": {
            "bytes": snapshot_info["bytes"],
            "sha256": snapshot_info["sha256"],
            "metadata_equal": metadata_equal,
            "serialization": compact_state_comparison(serialization),
        },
        "comparisons": {
            name: {
                "logits": value["logits"],
                "state": compact_state_comparison(value["state"]),
            }
            for name, value in comparisons.items()
        },
    }
    print("VELA_STATE_DIAGNOSTIC_V2_SUMMARY=" + json.dumps(summary, separators=(",", ":"), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
