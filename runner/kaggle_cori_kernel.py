from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MATCHA_REPO = "https://github.com/shivammehta25/Matcha-TTS.git"
MATCHA_COMMIT = "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
MATCHA = WORK / "Matcha-TTS"
OUTPUT = WORK / "c3-cori-kaggle-runs"
PRIVATE_STAGE = WORK / "c3-cori-private-input"
E280_SHA256 = "081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_private_root(root: Path) -> bool:
    return (
        (root / "c3-cori-handoff").is_dir()
        and (root / "c3-cori-dataset").is_dir()
        and (root / "c3-cori-e280" / "checkpoint_epoch=279.ckpt").is_file()
    )


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            target = (destination / info.filename).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"unsafe archive member in {archive.name}: {info.filename}")
        zf.extractall(destination)


def discover_private_input() -> Path:
    # Kaggle has used more than one input mount layout over time. Newer script
    # sessions can expose an extra /kaggle/input/datasets/... hierarchy, so do
    # not assume the Dataset is a direct child of /kaggle/input.
    direct_hits: list[Path] = []
    if INPUT.is_dir():
        for checkpoint in INPUT.rglob("checkpoint_epoch=279.ckpt"):
            root = checkpoint.parent.parent
            if is_private_root(root):
                direct_hits.append(root)
    direct_hits = sorted(set(direct_hits))
    if len(direct_hits) == 1:
        checkpoint = direct_hits[0] / "c3-cori-e280" / "checkpoint_epoch=279.ckpt"
        if sha256_file(checkpoint) != E280_SHA256:
            raise RuntimeError("attached E280 checkpoint SHA mismatch")
        print("C3_KAGGLE_PRIVATE_INPUT_LAYOUT=expanded", flush=True)
        return direct_hits[0]
    if len(direct_hits) > 1:
        raise RuntimeError(f"multiple expanded private C3 inputs found: {len(direct_hits)}")

    # The private Dataset was intentionally uploaded as two archives. Kaggle may
    # mount those archives verbatim rather than unpacking them. Locate them at
    # any nested input depth, require a unique same-directory pair, then expand
    # only into ephemeral /kaggle/working storage.
    base_zips = sorted(INPUT.rglob("c3-cori-base.zip")) if INPUT.is_dir() else []
    e280_zips = sorted(INPUT.rglob("c3-cori-e280.zip")) if INPUT.is_dir() else []
    pairs = [(base, e280) for base in base_zips for e280 in e280_zips if base.parent == e280.parent]
    if len(pairs) != 1:
        visible = sorted(p.name for p in INPUT.iterdir()) if INPUT.is_dir() else []
        raise RuntimeError(
            "expected exactly one attached private C3 input; "
            f"expanded_hits=0 archive_pairs={len(pairs)} base_zips={len(base_zips)} "
            f"e280_zips={len(e280_zips)} visible_roots={visible}"
        )

    base_zip, e280_zip = pairs[0]
    if PRIVATE_STAGE.exists():
        shutil.rmtree(PRIVATE_STAGE)
    PRIVATE_STAGE.mkdir(parents=True, exist_ok=True)
    print("C3_KAGGLE_PRIVATE_INPUT_LAYOUT=archived", flush=True)
    print("C3_KAGGLE_PRIVATE_INPUT_EXTRACT_BASE_BEGIN", flush=True)
    safe_extract_zip(base_zip, PRIVATE_STAGE)
    print("C3_KAGGLE_PRIVATE_INPUT_EXTRACT_E280_BEGIN", flush=True)
    safe_extract_zip(e280_zip, PRIVATE_STAGE)

    if not is_private_root(PRIVATE_STAGE):
        raise RuntimeError("private C3 archives extracted but required root structure is incomplete")
    checkpoint = PRIVATE_STAGE / "c3-cori-e280" / "checkpoint_epoch=279.ckpt"
    actual_sha = sha256_file(checkpoint)
    if actual_sha != E280_SHA256:
        raise RuntimeError(f"extracted E280 SHA mismatch: {actual_sha}")
    print("C3_KAGGLE_PRIVATE_INPUT_EXTRACT_PASS", flush=True)
    return PRIVATE_STAGE


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
        "schema": "c3-cori-kaggle-t4-benchmark-wrapper-v2",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "wall_seconds_including_environment_setup": elapsed,
        "environment": env,
        "private_input_root_name": source.name,
        "target_epoch": 290,
        "batch_size": 16,
        "e280_sha256": E280_SHA256,
    }
    (WORK / "C3_KAGGLE_T4_BENCHMARK_WRAPPER_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("C3_KAGGLE_T4_WRAPPER_PASS")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
