from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

MODEL_REPO = "BlinkDL/rwkv7-g1"
MODEL_REVISION = "2448c93da77d9958ea89a9faa4c1b34c5c46ccec"
MODEL_FILE = "rwkv7-g1a-2.9b-20250924-ctx4096.pth"
STRATEGY = "cpu fp32"
TOKENIZER = "rwkv_vocab_v20230424"
EXPECTED = {
    "rwkv": "0.8.32",
    "torch": "2.7.1+cpu",
    "tokenizers": "0.21.4",
    "numpy": "2.4.6",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_state(state):
    return [x.detach().clone() for x in state]


def state_equal(a, b) -> bool:
    return len(a) == len(b) and all(x.dtype == y.dtype and tuple(x.shape) == tuple(y.shape) and torch.equal(x, y) for x, y in zip(a, b))


def state_bytes(state) -> int:
    return int(sum(x.numel() * x.element_size() for x in state))


def write_result(obj: dict) -> None:
    path = Path(os.environ.get("VELA_RESULT_PATH", "/tmp/vela-r2-2p9b-preflight.json"))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(obj, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    versions = {
        "rwkv": importlib.metadata.version("rwkv"),
        "torch": torch.__version__,
        "tokenizers": importlib.metadata.version("tokenizers"),
        "numpy": np.__version__,
    }
    if versions != EXPECTED:
        raise RuntimeError(f"runtime identity drift: expected={EXPECTED} actual={versions}")

    t0 = time.monotonic()
    model_path = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, revision=MODEL_REVISION))
    download_seconds = time.monotonic() - t0
    size = model_path.stat().st_size
    digest = sha256_file(model_path)

    os.environ["RWKV_V7_ON"] = "1"
    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "0"
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    t1 = time.monotonic()
    model = RWKV(model=str(model_path)[:-4], strategy=STRATEGY)
    pipeline = PIPELINE(model, TOKENIZER)
    load_seconds = time.monotonic() - t1

    zero = model.generate_zero_state()
    zero_clone = clone_state(zero)
    if not state_equal(zero, zero_clone):
        raise RuntimeError("zero-state clone mismatch")

    probe_text = "VELA 2.9B infrastructure preflight only."
    probe_tokens = [int(x) for x in pipeline.encode(probe_text)]
    if not probe_tokens:
        raise RuntimeError("preflight probe encoded to zero tokens")
    t2 = time.monotonic()
    logits, state1 = model.forward(probe_tokens, clone_state(zero))
    forward_seconds = time.monotonic() - t2

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "state.pt"
        torch.save([x.detach().cpu() for x in state1], p)
        restored = torch.load(p, map_location="cpu", weights_only=True)
        roundtrip_exact = state_equal([x.detach().cpu() for x in state1], restored)
        if not roundtrip_exact:
            raise RuntimeError("state serialization round-trip mismatch")

        continuation = [int(x) for x in pipeline.encode(" OK")]
        if not continuation:
            raise RuntimeError("continuation encoded to zero tokens")
        out_a, next_a = model.forward([continuation[0]], clone_state(state1))
        out_b, next_b = model.forward([continuation[0]], clone_state(restored))
        continuation_exact = torch.equal(out_a, out_b) and state_equal(next_a, next_b)
        if not continuation_exact:
            raise RuntimeError("restored-state continuation mismatch")

    write_result({
        "status": "PASS",
        "scientific_evidence": False,
        "purpose": "result-blind model identity/load/state-roundtrip preflight only",
        "source_commit": os.environ.get("GITHUB_SHA"),
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "file": MODEL_FILE,
            "size": size,
            "sha256": digest,
            "strategy": STRATEGY,
            "tokenizer": TOKENIZER,
        },
        "runtime": versions,
        "state": {
            "tensor_count": len(zero),
            "zero_state_bytes": state_bytes(zero),
            "post_probe_state_bytes": state_bytes(state1),
            "roundtrip_exact": roundtrip_exact,
            "continuation_exact": continuation_exact,
        },
        "probe": {
            "token_count": len(probe_tokens),
            "download_seconds": download_seconds,
            "model_load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "logits_shape": list(logits.shape),
        },
    })


if __name__ == "__main__":
    main()
