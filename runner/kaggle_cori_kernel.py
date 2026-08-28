from __future__ import annotations

import ctypes.util
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
TARGET_NAMES = {"checkpoint_epoch=279.ckpt", "c3-cori-base.zip", "c3-cori-e280.zip"}


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_retry(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    attempts: int = 4,
    delay_seconds: int = 15,
) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            run(cmd, cwd=cwd)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"C3_KAGGLE_RETRY attempt={attempt}/{attempts} rc={exc.returncode} "
                f"sleep={delay_seconds}s",
                flush=True,
            )
            time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


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


def walk_input_followlinks() -> tuple[list[Path], list[Path], list[Path], list[str]]:
    checkpoints: list[Path] = []
    base_zips: list[Path] = []
    e280_zips: list[Path] = []
    tree: list[str] = []
    if not INPUT.is_dir():
        return checkpoints, base_zips, e280_zips, tree

    seen_real_dirs: set[str] = set()
    for base, dirs, files in os.walk(str(INPUT), followlinks=True):
        base_path = Path(base)
        try:
            real = str(base_path.resolve())
        except OSError:
            real = str(base_path)
        if real in seen_real_dirs:
            dirs[:] = []
            continue
        seen_real_dirs.add(real)

        if len(tree) < 80:
            try:
                rel = str(base_path.relative_to(INPUT)) or "."
            except ValueError:
                rel = str(base_path)
            # Never print source-audio filenames into public orchestration logs.
            targets = [name for name in files if name in TARGET_NAMES]
            tree.append(
                f"{rel}:dirs={dirs[:12]}:file_count={len(files)}:target_files={targets}:islink={base_path.is_symlink()}"
            )

        for name in files:
            p = base_path / name
            if name == "checkpoint_epoch=279.ckpt":
                checkpoints.append(p)
            elif name == "c3-cori-base.zip":
                base_zips.append(p)
            elif name == "c3-cori-e280.zip":
                e280_zips.append(p)

    return sorted(set(checkpoints)), sorted(set(base_zips)), sorted(set(e280_zips)), tree


def stage_split_expanded_mount(checkpoints: list[Path]) -> Path | None:
    """Normalize Kaggle's auto-expanded two-archive mount without copying data.

    Kaggle currently expands `c3-cori-base.zip` under a `c3-cori-base/` directory
    and `c3-cori-e280.zip` under a separate `c3-cori-e280/` directory. Build the
    unified layout expected by the frozen continuation code using ephemeral
    symlinks in /kaggle/working.
    """
    candidates: list[tuple[Path, Path]] = []
    for checkpoint in checkpoints:
        if checkpoint.name != "checkpoint_epoch=279.ckpt":
            continue
        e280_dir = checkpoint.parent
        for ancestor in checkpoint.parents:
            if ancestor == INPUT.parent:
                break
            base_dir = ancestor / "c3-cori-base"
            if (
                (base_dir / "c3-cori-handoff").is_dir()
                and (base_dir / "c3-cori-dataset").is_dir()
            ):
                candidates.append((base_dir, e280_dir))
                break

    # Deduplicate by resolved paths.
    unique: dict[tuple[str, str], tuple[Path, Path]] = {}
    for base_dir, e280_dir in candidates:
        key = (str(base_dir.resolve()), str(e280_dir.resolve()))
        unique[key] = (base_dir, e280_dir)
    pairs = list(unique.values())
    if not pairs:
        return None
    if len(pairs) != 1:
        raise RuntimeError(f"multiple split-expanded private C3 mounts found: {len(pairs)}")

    base_dir, e280_dir = pairs[0]
    checkpoint = e280_dir / "checkpoint_epoch=279.ckpt"
    actual_sha = sha256_file(checkpoint)
    if actual_sha != E280_SHA256:
        raise RuntimeError(f"split-expanded E280 SHA mismatch: {actual_sha}")

    if PRIVATE_STAGE.exists() or PRIVATE_STAGE.is_symlink():
        if PRIVATE_STAGE.is_symlink() or PRIVATE_STAGE.is_file():
            PRIVATE_STAGE.unlink()
        else:
            shutil.rmtree(PRIVATE_STAGE)
    PRIVATE_STAGE.mkdir(parents=True, exist_ok=True)
    (PRIVATE_STAGE / "c3-cori-handoff").symlink_to(
        (base_dir / "c3-cori-handoff").resolve(), target_is_directory=True
    )
    (PRIVATE_STAGE / "c3-cori-dataset").symlink_to(
        (base_dir / "c3-cori-dataset").resolve(), target_is_directory=True
    )
    (PRIVATE_STAGE / "c3-cori-e280").symlink_to(e280_dir.resolve(), target_is_directory=True)

    if not is_private_root(PRIVATE_STAGE):
        raise RuntimeError("split-expanded mount staging did not produce required unified structure")
    print("C3_KAGGLE_PRIVATE_INPUT_LAYOUT=split_expanded", flush=True)
    print("C3_KAGGLE_PRIVATE_INPUT_SHA_PASS", flush=True)
    return PRIVATE_STAGE


def discover_private_input() -> Path:
    checkpoints, base_zips, e280_zips, tree = walk_input_followlinks()

    # Legacy/unified expanded layout.
    direct_hits: list[Path] = []
    for checkpoint in checkpoints:
        root = checkpoint.parent.parent
        if is_private_root(root):
            direct_hits.append(root)
    direct_hits = sorted(set(direct_hits))
    if len(direct_hits) == 1:
        checkpoint = direct_hits[0] / "c3-cori-e280" / "checkpoint_epoch=279.ckpt"
        if sha256_file(checkpoint) != E280_SHA256:
            raise RuntimeError("attached E280 checkpoint SHA mismatch")
        print("C3_KAGGLE_PRIVATE_INPUT_LAYOUT=expanded", flush=True)
        print("C3_KAGGLE_PRIVATE_INPUT_SHA_PASS", flush=True)
        return direct_hits[0]
    if len(direct_hits) > 1:
        raise RuntimeError(f"multiple expanded private C3 inputs found: {len(direct_hits)}")

    # Current Kaggle behavior: each uploaded ZIP is auto-expanded into its own
    # sibling directory under the private Dataset mount.
    split = stage_split_expanded_mount(checkpoints)
    if split is not None:
        return split

    # Fallback for providers that mount the ZIP files verbatim.
    pairs = [(base, e280) for base in base_zips for e280 in e280_zips if base.parent == e280.parent]
    if len(pairs) != 1:
        print("C3_KAGGLE_INPUT_TREE_BEGIN", flush=True)
        for row in tree:
            print("C3_KAGGLE_INPUT_TREE", row, flush=True)
        print("C3_KAGGLE_INPUT_TREE_END", flush=True)
        raise RuntimeError(
            "expected exactly one attached private C3 input; "
            f"expanded_hits=0 archive_pairs={len(pairs)} base_zips={len(base_zips)} e280_zips={len(e280_zips)}"
        )

    base_zip, e280_zip = pairs[0]
    if PRIVATE_STAGE.exists():
        shutil.rmtree(PRIVATE_STAGE)
    PRIVATE_STAGE.mkdir(parents=True, exist_ok=True)
    print("C3_KAGGLE_PRIVATE_INPUT_LAYOUT=archived", flush=True)
    safe_extract_zip(base_zip, PRIVATE_STAGE)
    safe_extract_zip(e280_zip, PRIVATE_STAGE)

    if not is_private_root(PRIVATE_STAGE):
        raise RuntimeError("private C3 archives extracted but required root structure is incomplete")
    checkpoint = PRIVATE_STAGE / "c3-cori-e280" / "checkpoint_epoch=279.ckpt"
    actual_sha = sha256_file(checkpoint)
    if actual_sha != E280_SHA256:
        raise RuntimeError(f"extracted E280 SHA mismatch: {actual_sha}")
    print("C3_KAGGLE_PRIVATE_INPUT_EXTRACT_PASS", flush=True)
    print("C3_KAGGLE_PRIVATE_INPUT_SHA_PASS", flush=True)
    return PRIVATE_STAGE


def missing_system_packages() -> list[str]:
    package_commands = {
        "git": ["git"],
        "build-essential": ["gcc", "g++", "make"],
        "espeak-ng": ["espeak-ng"],
        "ffmpeg": ["ffmpeg"],
    }
    missing: list[str] = []
    for package, commands in package_commands.items():
        if any(shutil.which(command) is None for command in commands):
            missing.append(package)
    if ctypes.util.find_library("sndfile") is None:
        missing.append("libsndfile1")
    return sorted(set(missing))


def ensure_system_dependencies() -> None:
    missing = missing_system_packages()
    if not missing:
        print("C3_KAGGLE_SYSTEM_DEPS_PREINSTALLED=true", flush=True)
        return

    print("C3_KAGGLE_SYSTEM_DEPS_MISSING=" + ",".join(missing), flush=True)
    apt_base = [
        "apt-get",
        "-o",
        "Acquire::Retries=3",
        "-o",
        "Acquire::http::Timeout=20",
        "-o",
        "Acquire::https::Timeout=20",
    ]
    run_retry(apt_base + ["update", "-qq"], attempts=3, delay_seconds=20)
    run_retry(
        apt_base + ["install", "-y", "-qq", "--no-install-recommends", *missing],
        attempts=3,
        delay_seconds=20,
    )
    remaining = missing_system_packages()
    if remaining:
        raise RuntimeError("system dependencies still missing after apt install: " + ",".join(remaining))
    print("C3_KAGGLE_SYSTEM_DEPS_READY=true", flush=True)


def install_environment() -> None:
    # Do not spend GPU time hitting Ubuntu mirrors unless the current Kaggle image
    # is actually missing a required system dependency. Previous runs died only
    # because archive.ubuntu.com DNS resolution failed during unconditional apt.
    ensure_system_dependencies()
    run_retry(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--retries",
            "8",
            "--timeout",
            "30",
            "--upgrade",
            "--force-reinstall",
            "torch==2.5.1",
            "torchaudio==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        attempts=3,
        delay_seconds=20,
    )
    if MATCHA.exists():
        shutil.rmtree(MATCHA)
    run_retry(["git", "clone", "--quiet", MATCHA_REPO, str(MATCHA)], attempts=3, delay_seconds=20)
    run(["git", "checkout", "--detach", MATCHA_COMMIT], cwd=MATCHA)
    run_retry(
        [sys.executable, "-m", "pip", "install", "--quiet", "--retries", "8", "--timeout", "30", "-e", str(MATCHA)],
        attempts=3,
        delay_seconds=20,
    )
    run_retry(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--retries",
            "8",
            "--timeout",
            "30",
            "lightning==2.6.5",
        ],
        attempts=3,
        delay_seconds=20,
    )


def environment_report() -> dict:
    import lightning
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle T4 is not CUDA-visible after pinned environment setup")
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
        "schema": "c3-cori-kaggle-t4-benchmark-wrapper-v5",
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
