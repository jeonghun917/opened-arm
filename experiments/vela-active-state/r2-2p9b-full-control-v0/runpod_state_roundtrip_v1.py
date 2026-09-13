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


def state_bytes(state) -> int:
    return int(sum(x.numel() * x.element_size() for x in state))


def clone_state(state):
    return [x.detach().clone() for x in state]


def states_equal(a, b) -> bool:
    import torch
    if len(a) != len(b):
        return False
    return all(x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y) for x, y in zip(a, b))


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
    if path.stat().st_size != MODEL_SIZE:
        raise RuntimeError("2.9B model size mismatch")
    actual_sha = sha256_file(path)
    if actual_sha != MODEL_SHA256:
        raise RuntimeError(f"2.9B model sha mismatch: {actual_sha}")

    model = RWKV(model=str(path)[:-4], strategy="cuda fp16")
    pipe = PIPELINE(model, "rwkv_vocab_v20230424")
    zero = model.generate_zero_state()
    seed_tokens = [int(x) for x in pipe.encode("VELA RUNPOD STATE ROUNDTRIP NEUTRAL")]
    logits0, state = model.forward(seed_tokens, clone_state(zero))
    torch.cuda.synchronize()

    out_dir = Path(os.environ.get("VELA_ROUNDTRIP_DIR", "/workspace/vela"))
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = out_dir / "runpod-state-roundtrip.pt"
    torch.save(state, snapshot)
    restored = torch.load(snapshot, weights_only=False)
    state_exact = states_equal(state, restored)

    continuation = [int(x) for x in pipe.encode(" CONTINUATION CHECK")]
    logits_a, state_a = model.forward(continuation, clone_state(state))
    logits_b, state_b = model.forward(continuation, clone_state(restored))
    torch.cuda.synchronize()
    continuation_logits_exact = torch.equal(logits_a.detach().cpu(), logits_b.detach().cpu())
    continuation_state_exact = states_equal(state_a, state_b)

    report = {
        "status": "PASS" if state_exact and continuation_logits_exact and continuation_state_exact else "FAIL",
        "scientific_evidence": False,
        "purpose": "result-blind exact recurrent-state serialization roundtrip for P-R2-02",
        "source_commit": os.environ.get("VELA_SOURCE_COMMIT"),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "strategy": "cuda fp16",
            "rwkv_cuda_kernel": False
        },
        "model": {
            "file": MODEL_FILE,
            "size": path.stat().st_size,
            "sha256": actual_sha
        },
        "state": {
            "tensor_count": len(state),
            "state_bytes": state_bytes(state),
            "snapshot_bytes": snapshot.stat().st_size,
            "snapshot_sha256": sha256_file(snapshot),
            "serialize_restore_exact": state_exact,
            "continuation_logits_exact": continuation_logits_exact,
            "continuation_state_exact": continuation_state_exact
        }
    }
    print("VELA_STATE_ROUNDTRIP_JSON=" + json.dumps(report, separators=(",", ":"), sort_keys=True), flush=True)
    if report["status"] != "PASS":
        raise SystemExit("state roundtrip failed")


if __name__ == "__main__":
    main()
