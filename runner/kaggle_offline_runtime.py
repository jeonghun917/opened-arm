from __future__ import annotations

import ctypes.util
import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

MATCHA_COMMIT = "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
RUNTIME_STAGE = WORK / "c3-cori-offline-runtime"

FORCED_PYTHON_PACKAGES = [
    "lightning==2.6.5",
    "hydra-core==1.3.2",
    "hydra-colorlog==1.2.0",
    "omegaconf==2.3.0",
    "phonemizer==3.3.0",
    "conformer==0.3.2",
]
OPTIONAL_IF_MISSING = [
    "lightning-utilities",
    "torchmetrics",
    "antlr4-python3-runtime",
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
    "diffusers",
    "huggingface-hub",
    "safetensors",
    "Cython",
    "packaging",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str | Path], *, cwd: Path | None = None) -> None:
    rendered = [str(x) for x in cmd]
    print("C3_OFFLINE_CMD", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=str(cwd) if cwd else None, check=True)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"unsafe runtime archive member: {member.name}")
        tf.extractall(destination)


def discover_runtime() -> Path:
    manifests = list(INPUT.rglob("runtime-manifest.json"))
    archives = list(INPUT.rglob("c3-cori-offline-runtime.tar.gz"))

    if len(manifests) == 1:
        root = manifests[0].parent
    elif len(archives) == 1:
        if RUNTIME_STAGE.exists():
            shutil.rmtree(RUNTIME_STAGE)
        RUNTIME_STAGE.mkdir(parents=True)
        _safe_extract_tar(archives[0], RUNTIME_STAGE)
        root = RUNTIME_STAGE
    else:
        raise RuntimeError(
            "offline runtime discovery failed: "
            f"manifests={len(manifests)} archives={len(archives)}"
        )

    manifest_path = root / "runtime-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("offline runtime manifest missing after discovery")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3-kaggle-offline-runtime-v1":
        raise RuntimeError(f"unexpected offline runtime schema: {manifest.get('schema')!r}")
    if manifest.get("matcha_commit") != MATCHA_COMMIT:
        raise RuntimeError("offline runtime Matcha commit mismatch")
    if manifest.get("network_required_at_kaggle_runtime") is not False:
        raise RuntimeError("offline runtime manifest does not assert network-free execution")

    for row in manifest.get("files", []):
        rel = Path(str(row["path"]))
        path = root / rel
        if not path.is_file():
            raise RuntimeError(f"offline runtime file missing: {rel}")
        actual = sha256_file(path)
        if actual != row.get("sha256"):
            raise RuntimeError(f"offline runtime SHA mismatch: {rel}")

    print("C3_KAGGLE_OFFLINE_RUNTIME_SHA_PASS", flush=True)
    return root


def ensure_espeak_ng(runtime: Path) -> None:
    if shutil.which("espeak-ng") and ctypes.util.find_library("espeak-ng"):
        print("C3_KAGGLE_OFFLINE_ESPEAK_PREINSTALLED=true", flush=True)
        return

    debs = sorted((runtime / "debs").glob("*.deb"))
    if not debs:
        raise RuntimeError("offline runtime contains no espeak-ng deb packages")
    if os.geteuid() != 0:
        raise RuntimeError("offline espeak-ng dpkg install requires root in Kaggle container")
    run(["dpkg", "-i", *debs])
    run(["ldconfig"])
    if not shutil.which("espeak-ng"):
        raise RuntimeError("espeak-ng binary missing after offline dpkg install")
    if ctypes.util.find_library("espeak-ng") is None:
        raise RuntimeError("libespeak-ng not discoverable after offline dpkg install")
    print("C3_KAGGLE_OFFLINE_ESPEAK_PASS", flush=True)


def _dist_name(requirement: str) -> str:
    return requirement.split("==", 1)[0]


def ensure_python_dependencies(runtime: Path) -> None:
    wheelhouse = runtime / "wheelhouse"
    if not wheelhouse.is_dir() or not list(wheelhouse.glob("*.whl")):
        raise RuntimeError("offline runtime wheelhouse missing")

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--find-links",
            wheelhouse,
            *FORCED_PYTHON_PACKAGES,
        ]
    )
    for requirement in OPTIONAL_IF_MISSING:
        dist = _dist_name(requirement)
        try:
            metadata.version(dist)
        except metadata.PackageNotFoundError:
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--find-links",
                    wheelhouse,
                    requirement,
                ]
            )

    # Imports here intentionally happen before any GPU work. Missing transitive
    # dependencies must fail the environment gate rather than fall back to network.
    import lightning  # noqa: F401
    import hydra  # noqa: F401
    import phonemizer  # noqa: F401
    import torchmetrics  # noqa: F401

    print("C3_KAGGLE_OFFLINE_PY_DEPS_PASS", flush=True)


def prepare_matcha_source(runtime: Path, destination: Path) -> None:
    source_archive = runtime / "matcha-source.tar.gz"
    if not source_archive.is_file():
        raise RuntimeError("offline Matcha source archive missing")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    _safe_extract_tar(source_archive, destination)

    # Compile the Cython monotonic-alignment extension against the Kaggle image's
    # existing NumPy/Python ABI. No PEP-517 isolation and no package download.
    run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=destination)
    marker = destination / ".c3_matcha_commit"
    marker.write_text(MATCHA_COMMIT + "\n", encoding="utf-8")

    core_hits = list((destination / "matcha" / "utils" / "monotonic_align").glob("core*.so"))
    if not core_hits:
        raise RuntimeError("Matcha monotonic_align extension was not built")
    print("C3_KAGGLE_OFFLINE_MATCHA_BUILD_PASS", flush=True)


def prepare_offline_environment(matcha_destination: Path) -> dict:
    # Hard-disable accidental package-index access even if a later command invokes pip.
    os.environ["PIP_NO_INDEX"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    runtime = discover_runtime()
    ensure_espeak_ng(runtime)
    ensure_python_dependencies(runtime)
    prepare_matcha_source(runtime, matcha_destination)

    manifest = json.loads((runtime / "runtime-manifest.json").read_text(encoding="utf-8"))
    report = {
        "schema": "c3-kaggle-offline-environment-v1",
        "runtime_root_name": runtime.name,
        "matcha_commit": MATCHA_COMMIT,
        "builder_python": manifest.get("builder_python"),
        "network_required": False,
        "espeak_ng": shutil.which("espeak-ng"),
    }
    print("C3_KAGGLE_OFFLINE_ENVIRONMENT_PASS", json.dumps(report, ensure_ascii=False), flush=True)
    return report
