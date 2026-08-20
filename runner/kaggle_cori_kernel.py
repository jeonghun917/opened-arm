from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MATCHA_REPO = "https://github.com/shivammehta25/Matcha-TTS.git"
MATCHA_COMMIT = "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
MATCHA = WORK / "Matcha-TTS"
OUTPUT = WORK / "c3-cori-kaggle-runs"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def discover_private_input() -> Path:
    hits = []
    if INPUT.is_dir():
        for root in INPUT.iterdir():
            if not root.is_dir():
                continue
            if (
                (root / "c3-cori-handoff").is_dir()
                and (root / "c3-cori-dataset").is_dir()
                and (root / "c3-cori-e280" / "checkpoint_epoch=279.ckpt").is_file()
            ):
                hits.append(root)
    if len(hits) != 1:
        visible = sorted(p.name for p in INPUT.iterdir()) if INPUT.is_dir() else []
        raise RuntimeError(
            f"expected exactly one attached private C3 input dataset; hits={len(hits)} visible_roots={visible}"
        )
    return hits[0]


def install_environment() -> None:
    # Match the established continuation runtime as closely as practical rather than
    # relying on Kaggle's moving default torch image.
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "git", "build-essential", "espeak-ng", "ffmpeg", "libsndfile1"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "--force-reinstall",
            "torch==2.5.1",
            "torchaudio==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )
    if MATCHA.exists():
        shutil.rmtree(MATCHA)
    run(["git", "clone", "--quiet", MATCHA_REPO, str(MATCHA)])
    run(["git", "checkout", "--detach", MATCHA_COMMIT], cwd=MATCHA)
    run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(MATCHA)])
    run([sys.executable, "-m", "pip", "install", "--quiet", "lightning==2.6.5"])


def environment_report() -> dict:
    import lightning
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle T4 is not CUDA-visible after pinned environment setup")
    # Force a real CUDA kernel launch, not just a driver-visibility check.
    x = torch.ones(1, device="cuda")
    x.add_(1)
    torch.cuda.synchronize()
    del x
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "lightning": lightning.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 3),
        "matcha_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MATCHA, text=True).strip(),
    }


def main() -> None:
    started = time.monotonic()
    source = discover_private_input()
    print("C3_KAGGLE_PRIVATE_INPUT_DISCOVERED", source.name, flush=True)
    install_environment()
    env = environment_report()
    if env["gpu_count"] != 1 or "T4" not in str(env["gpu"]).upper():
        raise RuntimeError(f"first benchmark requires exactly one T4; environment={env}")
    if env["matcha_commit"] != MATCHA_COMMIT:
        raise RuntimeError("Matcha commit mismatch after environment setup")
    print("C3_KAGGLE_ENVIRONMENT", json.dumps(env, ensure_ascii=False), flush=True)

    here = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(here / "kaggle_cori_segment_entry.py"),
        "--handoff-dir",
        str(source / "c3-cori-handoff"),
        "--dataset-root",
        str(source / "c3-cori-dataset"),
        "--resume-checkpoint",
        str(source / "c3-cori-e280" / "checkpoint_epoch=279.ckpt"),
        "--output-root",
        str(OUTPUT),
        "--matcha-checkout",
        str(MATCHA),
        "--target-max-epochs",
        "290",
    ]
    run(cmd)

    elapsed = time.monotonic() - started
    report = {
        "schema": "c3-cori-kaggle-t4-benchmark-wrapper-v1",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "wall_seconds_including_environment_setup": elapsed,
        "environment": env,
        "private_input_root_name": source.name,
        "target_epoch": 290,
        "batch_size": 16,
    }
    (WORK / "C3_KAGGLE_T4_BENCHMARK_WRAPPER_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("C3_KAGGLE_T4_WRAPPER_PASS")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
