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
import textwrap
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/c3-offline-validation")
MATCHA_COMMIT = "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
E280_SHA256 = "081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9"

FORCED = [
    "lightning==2.6.5",
    "hydra-core==1.3.2",
    "hydra-colorlog==1.2.0",
    "omegaconf==2.3.0",
    "phonemizer==3.3.0",
    "conformer==0.3.2",
]
OPTIONAL = [
    "lightning-utilities", "torchmetrics", "antlr4-python3-runtime", "segments",
    "dlinfo", "joblib", "attrs", "rootutils", "rich", "einops", "inflect",
    "more-itertools", "typeguard", "Unidecode", "diffusers", "huggingface-hub",
    "safetensors", "Cython", "packaging",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str | Path], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    rendered = [str(x) for x in cmd]
    print("C3_OFFLINE_CMD", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=str(cwd) if cwd else None, env=env, check=True)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tf.extractall(destination)


def discover_runtime() -> Path:
    archives = list(INPUT.rglob("c3-cori-offline-runtime.tar.gz"))
    manifests = list(INPUT.rglob("runtime-manifest.json"))
    runtime = WORK / "runtime"
    if len(archives) == 1:
        if runtime.exists():
            shutil.rmtree(runtime)
        runtime.mkdir(parents=True)
        safe_extract(archives[0], runtime)
    elif len(manifests) == 1:
        runtime = manifests[0].parent
    else:
        raise RuntimeError(
            f"runtime discovery failed archives={len(archives)} manifests={len(manifests)}"
        )
    manifest = json.loads((runtime / "runtime-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3-kaggle-offline-runtime-v2":
        raise RuntimeError(f"runtime schema mismatch: {manifest.get('schema')!r}")
    if manifest.get("matcha_commit") != MATCHA_COMMIT:
        raise RuntimeError("runtime Matcha commit mismatch")
    for row in manifest.get("files", []):
        path = runtime / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"runtime integrity mismatch: {row['path']}")
    print("C3_OFFLINE_RUNTIME_SHA_PASS", flush=True)
    return runtime


def install_system(runtime: Path) -> None:
    if not (shutil.which("espeak-ng") and ctypes.util.find_library("espeak-ng")):
        if os.geteuid() != 0:
            raise RuntimeError("Kaggle container is not root; cannot offline-install espeak-ng debs")
        debs = sorted((runtime / "debs").glob("*.deb"))
        if not debs:
            raise RuntimeError("espeak-ng deb bundle missing")
        run(["dpkg", "-i", *debs])
        run(["ldconfig"])
    if not shutil.which("espeak-ng") or ctypes.util.find_library("espeak-ng") is None:
        raise RuntimeError("espeak-ng unavailable after offline system bootstrap")
    print("C3_OFFLINE_ESPEAK_PASS", flush=True)


def install_python(runtime: Path) -> None:
    wheelhouse = runtime / "wheelhouse"
    os.environ["PIP_NO_INDEX"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    run([
        sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
        "--find-links", wheelhouse, *FORCED,
    ])
    for req in OPTIONAL:
        dist = req.split("==", 1)[0]
        try:
            metadata.version(dist)
        except metadata.PackageNotFoundError:
            run([
                sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                "--find-links", wheelhouse, req,
            ])
    import lightning  # noqa: F401
    import hydra  # noqa: F401
    import phonemizer  # noqa: F401
    import torchmetrics  # noqa: F401
    print("C3_OFFLINE_PY_DEPS_PASS", flush=True)


def prepare_matcha(runtime: Path) -> Path:
    matcha = WORK / "Matcha-TTS"
    if matcha.exists():
        shutil.rmtree(matcha)
    matcha.mkdir(parents=True)
    safe_extract(runtime / "matcha-source.tar.gz", matcha)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=matcha, text=True).strip()
    if actual != MATCHA_COMMIT:
        raise RuntimeError(f"offline Matcha git identity mismatch: {actual}")
    # Exact continuation runner executes these commands. Prove they are network-free here.
    run(["git", "fetch", "--all", "--tags"], cwd=matcha)
    run(["git", "checkout", "--detach", MATCHA_COMMIT], cwd=matcha)
    run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=matcha)
    if not list((matcha / "matcha/utils/monotonic_align").glob("core*.so")):
        raise RuntimeError("monotonic_align extension was not built")
    print("C3_OFFLINE_MATCHA_BUILD_PASS", flush=True)
    return matcha


def patch_cleaner_and_verify(matcha: Path) -> None:
    cleaners = matcha / "matcha/text/cleaners.py"
    source = cleaners.read_text(encoding="utf-8")
    marker = "# C3_MATCHA_TEXT_PATCH_V1"
    if marker not in source:
        if "import unicodedata\n" not in source:
            source = source.replace(
                "import logging\nimport re\n",
                "import logging\nimport re\nimport unicodedata\n",
                1,
            )
        needle = "    phonemes = collapse_whitespace(phonemes)\n    return phonemes\n"
        replacement = """    phonemes = collapse_whitespace(phonemes)\n    # C3_MATCHA_TEXT_PATCH_V1\n    from matcha.text.symbols import symbols as _c3_symbols\n    _c3_allowed = set(_c3_symbols)\n    _c3_unknown_noncombining = sorted(\n        {ch for ch in phonemes if ch not in _c3_allowed and not unicodedata.combining(ch)}\n    )\n    if _c3_unknown_noncombining:\n        raise ValueError(\n            f\"unsupported non-combining Matcha symbols: {_c3_unknown_noncombining}\"\n        )\n    phonemes = \"\".join(\n        ch for ch in phonemes if ch in _c3_allowed or not unicodedata.combining(ch)\n    )\n    return phonemes\n"""
        if needle not in source:
            raise RuntimeError("Matcha cleaner patch anchor not found")
        cleaners.write_text(source.replace(needle, replacement, 1), encoding="utf-8")

    stats_hits = list(INPUT.rglob("STATS.json"))
    if len(stats_hits) != 1:
        raise RuntimeError(f"expected one STATS.json, got {len(stats_hits)}")
    stats = json.loads(stats_hits[0].read_text(encoding="utf-8"))
    expected = str(stats.get("matcha_patched_cleaners_sha256", ""))
    actual = sha256_file(cleaners)
    if not expected or actual != expected:
        raise RuntimeError(f"cleaner SHA mismatch actual={actual} expected={expected}")
    print("C3_OFFLINE_CLEANER_SHA_PASS", flush=True)


def verify_imports_and_phonemizer(matcha: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(matcha) + os.pathsep + env.get("PYTHONPATH", "")
    code = textwrap.dedent(
        """
        import hashlib, json, sys, torch
        import lightning
        import matcha.train
        from matcha.text.cleaners import english_cleaners2
        text = "The sixth street shuttle stopped beside three freshly painted shops."
        phonemes = english_cleaners2(text)
        print("C3_OFFLINE_IMPORT_PASS")
        print("C3_OFFLINE_PHONEME_SHA=" + hashlib.sha256(phonemes.encode()).hexdigest())
        print("C3_OFFLINE_VERSIONS=" + json.dumps({
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "lightning": lightning.__version__,
        }))
        """
    )
    run([sys.executable, "-c", code], cwd=matcha, env=env)


def verify_e280() -> None:
    hits = []
    for path in INPUT.rglob("checkpoint_epoch=279.ckpt"):
        try:
            if sha256_file(path) == E280_SHA256:
                hits.append(path)
        except OSError:
            continue
    if len(hits) != 1:
        raise RuntimeError(f"expected one exact E280 checkpoint, got {len(hits)}")
    import torch
    payload = torch.load(hits[0], map_location="cpu", weights_only=False)
    epoch = int(payload.get("epoch", -1)) + 1
    step = int(payload.get("global_step", -1))
    if (epoch, step) != (280, 140560):
        raise RuntimeError(f"E280 metadata mismatch epoch={epoch} step={step}")
    print("C3_OFFLINE_E280_LOAD_PASS", flush=True)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    runtime = discover_runtime()
    install_system(runtime)
    install_python(runtime)
    matcha = prepare_matcha(runtime)
    patch_cleaner_and_verify(matcha)
    verify_imports_and_phonemizer(matcha)
    verify_e280()
    print("C3_OFFLINE_CPU_VALIDATION_PASS", flush=True)


if __name__ == "__main__":
    main()
