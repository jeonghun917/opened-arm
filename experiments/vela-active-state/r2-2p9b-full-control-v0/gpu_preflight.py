from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


EXPECTED_TORCH = "2.7.1+cu126"


def ensure_runtime() -> None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--extra-index-url", "https://download.pytorch.org/whl/cu126",
        f"torch=={EXPECTED_TORCH}",
        "rwkv==0.8.32",
        "tokenizers==0.21.4",
        "numpy==2.4.6",
        "huggingface_hub==0.36.0",
    ])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_bytes(state) -> int:
    return int(sum(x.numel() * x.element_size() for x in state))


def write_report(report: dict) -> None:
    Path("gpu_preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    ensure_runtime()
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download

    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(f"torch version drift: expected {EXPECTED_TORCH}, got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU preflight requires CUDA")
    arches = list(torch.cuda.get_arch_list())
    capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))
    if capability == (6, 0) and "sm_60" not in arches:
        raise RuntimeError(f"P100 requires sm_60 but wheel arches are {arches}")

    os.environ["RWKV_V7_ON"] = "1"
    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "0"
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    models = [
        {
            "scale": "0.4B",
            "repo": "BlinkDL/rwkv7-g1",
            "revision": "c4d4f4748a8233e9a2f1ee7f38f4c4e3780e750d",
            "file": "rwkv7-g1a-0.4b-20250905-ctx4096.pth",
            "size": 901776757,
            "sha256": "d852e99ef6c95726109660c64e7c51a8df30c53b0832a68645bfcd15253b3109",
        },
        {
            "scale": "2.9B",
            "repo": "BlinkDL/rwkv7-g1",
            "revision": "2448c93da77d9958ea89a9faa4c1b34c5c46ccec",
            "file": "rwkv7-g1a-2.9b-20250924-ctx4096.pth",
            "size": 5896274949,
            "sha256": "df5716263b617e7da83590446fb2a98b6663447cfd52e8e9b49e11ce7f4faa3e",
        },
    ]

    results = []
    for spec in models:
        t0 = time.monotonic()
        path = Path(hf_hub_download(repo_id=spec["repo"], filename=spec["file"], revision=spec["revision"]))
        dl = time.monotonic() - t0
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != spec["size"] or actual_sha != spec["sha256"]:
            raise RuntimeError(f"model identity mismatch for {spec['scale']}: size={actual_size} sha={actual_sha}")

        torch.cuda.reset_peak_memory_stats()
        t1 = time.monotonic()
        model = RWKV(model=str(path)[:-4], strategy="cuda fp16")
        pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
        load_s = time.monotonic() - t1
        zero = model.generate_zero_state()
        tokens = [int(x) for x in pipeline.encode("VELA GPU preflight")]
        if not tokens:
            raise RuntimeError("empty probe tokens")
        t2 = time.monotonic()
        logits, state = model.forward(tokens, [x.detach().clone() for x in zero])
        torch.cuda.synchronize()
        fwd_s = time.monotonic() - t2
        results.append({
            **spec,
            "actual_size": actual_size,
            "actual_sha256": actual_sha,
            "download_seconds": dl,
            "load_seconds": load_s,
            "probe_tokens": len(tokens),
            "forward_seconds": fwd_s,
            "zero_state_bytes": state_bytes(zero),
            "post_state_bytes": state_bytes(state),
            "tensor_count": len(zero),
            "logits_shape": list(logits.shape),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        })
        del logits, state, zero, pipeline, model
        gc.collect()
        torch.cuda.empty_cache()

    write_report({
        "status": "PASS",
        "scientific_evidence": False,
        "purpose": "result-blind matched-GPU backend preflight for P-R2-02",
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_arch_list": arches,
            "rwkv": importlib.metadata.version("rwkv"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(capability),
            "strategy": "cuda fp16",
            "rwkv_cuda_kernel": False,
        },
        "models": results,
    })


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        runtime = {"python": sys.version.split()[0]}
        try:
            import torch
            runtime.update({
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            })
        except BaseException:
            pass
        write_report({
            "status": "PREFLIGHT_ERROR",
            "scientific_evidence": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runtime": runtime,
            "traceback_tail": traceback.format_exc().splitlines()[-80:],
        })
        raise
