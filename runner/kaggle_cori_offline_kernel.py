from __future__ import annotations

import json
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import kaggle_cori_kernel as legacy
import kaggle_offline_runtime as offline

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
MATCHA = WORK / "Matcha-TTS"
OUTPUT = WORK / "c3-cori-kaggle-runs"
RUNTIME_STAGE = WORK / "c3-cori-offline-runtime-v2"


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"unsafe offline runtime archive member: {member.name}")
        tf.extractall(destination)


def prepare_offline_v2() -> dict:
    manifests = list(INPUT.rglob("runtime-manifest.json"))
    archives = list(INPUT.rglob("c3-cori-offline-runtime.tar.gz"))
    if len(manifests) == 1:
        runtime = manifests[0].parent
    elif len(archives) == 1:
        if RUNTIME_STAGE.exists():
            shutil.rmtree(RUNTIME_STAGE)
        RUNTIME_STAGE.mkdir(parents=True)
        _safe_extract(archives[0], RUNTIME_STAGE)
        runtime = RUNTIME_STAGE
    else:
        raise RuntimeError(
            f"offline runtime discovery failed manifests={len(manifests)} archives={len(archives)}"
        )

    manifest = json.loads((runtime / "runtime-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3-kaggle-offline-runtime-v2":
        raise RuntimeError(f"offline runtime schema mismatch: {manifest.get('schema')!r}")
    if manifest.get("matcha_commit") != legacy.MATCHA_COMMIT:
        raise RuntimeError("offline runtime Matcha commit mismatch")
    if manifest.get("network_required_at_kaggle_runtime") is not False:
        raise RuntimeError("offline runtime does not assert network-free execution")

    # Kaggle-only transport exception: Kaggle may re-materialize archive-like files
    # (.whl/.deb) and change byte identity. Keep those hashes as provenance, but
    # validate them operationally by offline installation/import below. Any non-
    # archive payload mismatch remains fatal. E280 checkpoint validation is separate
    # and remains exact-SHA-gated by the continuation path.
    kaggle_archive_exemptions = 0
    for row in manifest.get("files", []):
        rel = Path(str(row["path"]))
        path = runtime / rel
        if not path.is_file():
            raise RuntimeError(f"offline runtime file missing: {rel}")
        if offline.sha256_file(path) != row.get("sha256"):
            if rel.parts and rel.parts[0] in {"wheelhouse", "debs"}:
                kaggle_archive_exemptions += 1
                print(f"C3_KAGGLE_ARCHIVE_SHA_EXEMPT {rel}", flush=True)
                continue
            raise RuntimeError(f"offline runtime integrity mismatch: {rel}")
    print(
        f"C3_KAGGLE_OFFLINE_RUNTIME_INTEGRITY_PASS kaggle_archive_exemptions={kaggle_archive_exemptions}",
        flush=True,
    )

    offline.ensure_espeak_ng(runtime)
    offline.ensure_python_dependencies(runtime)
    offline.prepare_matcha_source(runtime, MATCHA)

    return {
        "schema": manifest.get("schema"),
        "matcha_commit": manifest.get("matcha_commit"),
        "builder_python": manifest.get("builder_python"),
        "network_required": False,
        "torch_vendored": manifest.get("torch_vendored"),
        "kaggle_archive_sha_exemptions": kaggle_archive_exemptions,
    }


def main() -> None:
    started = time.monotonic()

    # Private C3 assets are discovered and SHA-gated by the already-tested mount
    # normalization path. No source audio or checkpoint bytes are logged.
    source = legacy.discover_private_input()
    print("C3_KAGGLE_PRIVATE_INPUT_DISCOVERED", source.name, flush=True)

    # No apt/pip/git network access. Public Matcha source, eSpeak-NG packages and
    # small Python dependency wheels come from the attached runtime Dataset.
    offline_runtime = prepare_offline_v2()

    env = legacy.environment_report()
    if env["gpu_count"] != 1 or "T4" not in str(env["gpu"]).upper():
        raise RuntimeError(f"first benchmark requires exactly one T4; environment={env}")
    if env["matcha_commit"] != legacy.MATCHA_COMMIT:
        raise RuntimeError("Matcha commit mismatch after offline environment setup")
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
    legacy.run(cmd)

    elapsed = time.monotonic() - started
    report = {
        "schema": "c3-cori-kaggle-t4-offline-wrapper-v2",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "wall_seconds_including_offline_environment_setup": elapsed,
        "environment": env,
        "offline_runtime": offline_runtime,
        "target_epoch": 290,
        "batch_size": 16,
        "e280_sha256": legacy.E280_SHA256,
        "network_enabled": False,
    }
    (WORK / "C3_KAGGLE_T4_BENCHMARK_WRAPPER_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("C3_KAGGLE_T4_OFFLINE_WRAPPER_PASS")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
