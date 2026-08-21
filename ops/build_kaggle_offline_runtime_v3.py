from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import build_kaggle_offline_runtime as base


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest_v3() -> None:
    """Hash transport-stable payloads; Matcha source is identity-gated by git SHA later.

    Kaggle can re-materialize nested archives when creating a Dataset version. The
    previous CPU probe proved that an exact byte hash on the nested Matcha tarball
    is therefore a false transport gate. We still verify every wheel/deb byte, and
    the validator independently verifies the extracted Matcha git commit plus the
    patched cleaner SHA before any training is allowed.
    """
    rows = []
    for path in sorted(base.RUNTIME.rglob("*")):
        if not path.is_file() or path.name in {"runtime-manifest.json", "matcha-source.tar.gz"}:
            continue
        rows.append(
            {
                "path": str(path.relative_to(base.RUNTIME)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema": base.RUNTIME_SCHEMA,
        "matcha_commit": base.MATCHA_COMMIT,
        "network_required_at_kaggle_runtime": False,
        "torch_vendored": False,
        "transport_integrity_policy": (
            "SHA-256 all vendored wheels/debs; verify Matcha source semantically by exact git commit "
            "and patched cleaner SHA after extraction"
        ),
        "files": rows,
    }
    (base.RUNTIME / "runtime-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("C3_OFFLINE_MANIFEST_V3_PASS", flush=True)


def wait_for_exact_runtime_archive(runtime_id: str) -> None:
    """Do not launch a validator against a stale/asynchronously processed Dataset version."""
    local_archive = base.UPLOAD / "c3-cori-offline-runtime.tar.gz"
    expected = sha256_file(local_archive)
    verify_root = base.ROOT / "runtime-transport-verify"

    for attempt in range(1, 31):
        shutil.rmtree(verify_root, ignore_errors=True)
        verify_root.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                runtime_id,
                "-f",
                local_archive.name,
                "-p",
                str(verify_root),
                "--unzip",
                "-o",
                "-q",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        hits = list(verify_root.rglob(local_archive.name)) if proc.returncode == 0 else []
        if len(hits) == 1:
            actual = sha256_file(hits[0])
            if actual == expected:
                print(f"C3_OFFLINE_RUNTIME_TRANSPORT_SHA_PASS attempt={attempt}", flush=True)
                return
        print(f"C3_OFFLINE_RUNTIME_TRANSPORT_WAIT attempt={attempt}/30", flush=True)
        time.sleep(10)

    raise RuntimeError("Kaggle runtime Dataset never exposed the exact uploaded archive bytes")


def publish_runtime_v3(owner: str) -> str:
    runtime_id = base.publish_runtime(owner)
    wait_for_exact_runtime_archive(runtime_id)
    return runtime_id


def resolve_training_dataset(owner: str) -> str:
    raw = os.environ.get("KAGGLE_DATASET_ID", "").strip()
    if raw.count("/") != 1:
        raise RuntimeError("KAGGLE_DATASET_ID missing or malformed")
    slug = raw.split("/", 1)[1]
    candidates: list[str] = []
    for item in (f"{owner}/{slug}", raw):
        if item not in candidates:
            candidates.append(item)

    for item in candidates:
        proc = subprocess.run(
            ["kaggle", "datasets", "status", item],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0:
            print("C3_OFFLINE_PRIVATE_DATASET_HANDLE_PASS", flush=True)
            return item
    raise RuntimeError("no valid private training Dataset handle for authenticated Kaggle account")


def build_validation_kernel_v3(owner: str, runtime_id: str) -> str:
    c3_dataset_id = resolve_training_dataset(owner)
    validator = Path("ops/kaggle_offline_cpu_validate.py")
    if not validator.is_file():
        raise RuntimeError("offline CPU validator script missing from repository")
    shutil.copy2(validator, base.KERNEL / "validate.py")
    kernel_id = f"{owner}/c3-cori-offline-runtime-cpu-validate"
    meta = {
        "id": kernel_id,
        "title": "C3 Cori Offline Runtime CPU Validate",
        "code_file": "validate.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [c3_dataset_id, runtime_id],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (base.KERNEL / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("C3_OFFLINE_VALIDATION_KERNEL_METADATA_PASS", flush=True)
    return kernel_id


def main() -> int:
    base.write_manifest = write_manifest_v3
    base.publish_runtime = publish_runtime_v3
    base.build_validation_kernel = build_validation_kernel_v3
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
