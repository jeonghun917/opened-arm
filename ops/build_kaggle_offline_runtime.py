from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

MATCHA_COMMIT = "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
RUNTIME_SCHEMA = "c3-kaggle-offline-runtime-v2"
ROOT = Path(".c3-kaggle-offline-build")
RUNTIME = ROOT / "runtime"
UPLOAD = ROOT / "upload"
KERNEL = ROOT / "kernel"
OUTPUT = ROOT / "output"
REPORT = Path("ops/kaggle-offline-runtime-validation-latest.txt")

WHEEL_PACKAGES = [
    "lightning==2.6.5",
    "lightning-utilities",
    "torchmetrics",
    "hydra-core==1.3.2",
    "hydra-colorlog==1.2.0",
    "omegaconf==2.3.0",
    "antlr4-python3-runtime==4.9.3",
    "phonemizer==3.3.0",
    "segments",
    "dlinfo",
    "joblib",
    "attrs",
    "rootutils",
    "rich",
    "einops",
    "inflect",
    "more-itertools",
    "typeguard",
    "Unidecode",
    "conformer==0.3.2",
    "diffusers",
    "huggingface-hub",
    "safetensors",
    "Cython",
    "packaging",
]
DEB_PACKAGES = ["espeak-ng", "espeak-ng-data", "libespeak-ng1", "libpcaudio0", "libsonic0"]


def run(cmd: list[str | Path], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    rendered = [str(x) for x in cmd]
    print("+", " ".join(rendered), flush=True)
    return subprocess.run(
        rendered,
        cwd=str(cwd) if cwd else None,
        text=True,
        check=check,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_owner() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN missing")
    req = urllib.request.Request(
        "https://www.kaggle.com/api/v1/oauth2/introspect",
        data=urllib.parse.urlencode({"token": token}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.load(response)
    owner = str(data.get("username") or "").strip()
    if not data.get("active") or not owner:
        raise RuntimeError("Kaggle token is inactive or has no username")
    print("C3_OFFLINE_OWNER_RESOLVED", flush=True)
    return owner


def build_matcha_source() -> None:
    repo = ROOT / "Matcha-TTS"
    run(["git", "clone", "--quiet", "https://github.com/shivammehta25/Matcha-TTS.git", repo])
    run(["git", "checkout", "--detach", MATCHA_COMMIT], cwd=repo)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != MATCHA_COMMIT:
        raise RuntimeError(f"Matcha checkout mismatch: {actual}")
    # Remove network remote. The continuation runner's `git fetch --all` then becomes
    # a local no-op while `git checkout <exact SHA>` still works from packed objects.
    run(["git", "remote", "remove", "origin"], cwd=repo)
    source_tar = RUNTIME / "matcha-source.tar.gz"
    with tarfile.open(source_tar, "w:gz") as tf:
        for path in sorted(repo.rglob("*")):
            tf.add(path, arcname=str(path.relative_to(repo)), recursive=False)
    print("C3_OFFLINE_MATCHA_SOURCE_BUNDLED", flush=True)


def build_wheelhouse() -> None:
    wheelhouse = RUNTIME / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    for package in WHEEL_PACKAGES:
        run([
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--quiet",
            "--wheel-dir",
            wheelhouse,
            "--no-deps",
            package,
        ])
    if not list(wheelhouse.glob("*.whl")):
        raise RuntimeError("wheelhouse build produced no wheels")
    print("C3_OFFLINE_WHEELHOUSE_PASS", flush=True)


def build_debs() -> None:
    debs = RUNTIME / "debs"
    debs.mkdir(parents=True, exist_ok=True)
    run(["sudo", "apt-get", "update", "-qq"])
    for package in DEB_PACKAGES:
        run(["apt-get", "download", package], cwd=debs)
    if not list(debs.glob("*.deb")):
        raise RuntimeError("espeak-ng deb build produced no packages")
    print("C3_OFFLINE_DEB_BUNDLE_PASS", flush=True)


def write_manifest() -> None:
    rows = []
    for path in sorted(RUNTIME.rglob("*")):
        if not path.is_file() or path.name == "runtime-manifest.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(RUNTIME)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema": RUNTIME_SCHEMA,
        "matcha_commit": MATCHA_COMMIT,
        "builder_python": platform.python_version(),
        "builder_platform": platform.platform(),
        "network_required_at_kaggle_runtime": False,
        "torch_vendored": False,
        "scientific_stack_policy": "reuse Kaggle CUDA-matched torch/numpy/scipy; vendor only Matcha/source and missing control/text dependencies",
        "files": rows,
    }
    (RUNTIME / "runtime-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def package_runtime() -> Path:
    archive = UPLOAD / "c3-cori-offline-runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for path in sorted(RUNTIME.rglob("*")):
            tf.add(path, arcname=str(path.relative_to(RUNTIME)), recursive=False)
    (UPLOAD / "archive.sha256").write_text(
        sha256_file(archive) + "  " + archive.name + "\n", encoding="utf-8"
    )
    return archive


def publish_runtime(owner: str) -> str:
    runtime_id = f"{owner}/c3-cori-offline-runtime"
    metadata = {
        "title": "C3 Cori Offline Runtime",
        "id": runtime_id,
        "licenses": [{"name": "other"}],
        "description": (
            "Private operational cache of public/open-source dependencies for network-free "
            "C3 Cori Kaggle execution. No voice audio or private model weights."
        ),
    }
    (UPLOAD / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    exists = run(["kaggle", "datasets", "status", runtime_id], check=False).returncode == 0
    if exists:
        run(["kaggle", "datasets", "version", "-p", UPLOAD, "-m", "Refresh offline runtime", "-q", "-t"])
    else:
        run(["kaggle", "datasets", "create", "-p", UPLOAD, "-q", "-t"])
    run(["kaggle", "datasets", "status", runtime_id])
    print("C3_OFFLINE_RUNTIME_PUBLISH_PASS", flush=True)
    return runtime_id


def build_validation_kernel(owner: str, runtime_id: str) -> str:
    c3_dataset_id = os.environ.get("KAGGLE_DATASET_ID", "").strip()
    if c3_dataset_id.count("/") != 1:
        raise RuntimeError("KAGGLE_DATASET_ID missing or malformed")
    validator = Path("ops/kaggle_offline_cpu_validate.py")
    if not validator.is_file():
        raise RuntimeError("offline CPU validator script missing from repository")
    shutil.copy2(validator, KERNEL / "validate.py")
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
    (KERNEL / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return kernel_id


def parse_kaggle_logs() -> list[str]:
    lines: list[str] = []
    for path in sorted(OUTPUT.rglob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            rows = json.loads(text)
        except Exception:
            rows = None
        raw: list[str] = []
        if isinstance(rows, list):
            for row in rows:
                raw.extend(str(row.get("data", "")).splitlines())
        else:
            raw.extend(text.splitlines())
        for line in raw:
            if any(
                token in line
                for token in (
                    "C3_OFFLINE_",
                    "Traceback",
                    "RuntimeError",
                    "ModuleNotFoundError",
                    "ImportError",
                    "CalledProcessError",
                    "ERROR",
                    "Error",
                )
            ):
                lines.append(line[:3000])
    return lines[-500:]


def validate_cpu(kernel_id: str) -> bool:
    run(["kaggle", "kernels", "push", "-p", KERNEL, "-t", "1200"])
    status = "UNKNOWN"
    for _ in range(120):
        proc = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            text=True,
            capture_output=True,
            check=False,
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        if "KernelWorkerStatus.COMPLETE" in text:
            status = "COMPLETE"
            break
        if any(x in text for x in ("KernelWorkerStatus.ERROR", "KernelWorkerStatus.FAILED", "KernelWorkerStatus.CANCELLED", "KernelWorkerStatus.CANCELED")):
            status = "ERROR"
            break
        time.sleep(10)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    run(["kaggle", "kernels", "output", kernel_id, "-p", OUTPUT, "--force"], check=False)
    lines = parse_kaggle_logs()
    passed = status == "COMPLETE" and any("C3_OFFLINE_CPU_VALIDATION_PASS" in x for x in lines)
    body = [
        "# Kaggle offline runtime CPU validation",
        "",
        "Private CPU-only validation; no credentials or model/audio bytes are recorded here.",
        "",
        f"terminal_status: {status}",
        f"validation_pass: {str(passed).lower()}",
        "",
        "```text",
        *lines,
        "```",
        "",
    ]
    REPORT.write_text("\n".join(body), encoding="utf-8")
    print("\n".join(lines[-100:]), flush=True)
    return passed


def main() -> int:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    RUNTIME.mkdir(parents=True)
    UPLOAD.mkdir(parents=True)
    KERNEL.mkdir(parents=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    owner = resolve_owner()
    build_matcha_source()
    build_wheelhouse()
    build_debs()
    write_manifest()
    archive = package_runtime()
    print(f"C3_OFFLINE_RUNTIME_ARCHIVE_BYTES={archive.stat().st_size}", flush=True)
    runtime_id = publish_runtime(owner)
    kernel_id = build_validation_kernel(owner, runtime_id)
    passed = validate_cpu(kernel_id)
    if passed:
        print("C3_OFFLINE_RUNTIME_END_TO_END_PASS", flush=True)
        return 0
    print("C3_OFFLINE_RUNTIME_END_TO_END_FAIL", flush=True)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
