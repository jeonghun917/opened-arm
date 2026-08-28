from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import build_kaggle_offline_runtime as base

# Keep stable references before monkey-patching base.main() hooks. Without this,
# publish_runtime_v3() recursively calls itself after base.publish_runtime is replaced.
BASE_PUBLISH_RUNTIME = base.publish_runtime


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest_v3() -> None:
    """Hash only payloads whose path and bytes survive Kaggle Dataset transport.

    Kaggle can re-materialize wheel/deb/archive-like files when creating a Dataset
    version, including changing the mounted filename. Those payloads therefore cannot
    participate in a path+SHA transport invariant. Their provenance remains in the
    builder output and their runtime validity is enforced by network-free pip/dpkg plus
    import/version checks. This relaxation is Kaggle-only: non-archive payload hashes,
    exact Matcha identity, cleaner SHA, E280 SHA and checkpoint metadata stay strict.
    """
    rows = []
    archive_payloads = []
    for path in sorted(base.RUNTIME.rglob("*")):
        if not path.is_file() or path.name in {"runtime-manifest.json", "matcha-source.tar.gz"}:
            continue
        rel = path.relative_to(base.RUNTIME)
        if rel.parts and rel.parts[0] in {"wheelhouse", "debs"}:
            archive_payloads.append(
                {
                    "path": str(rel),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            continue
        rows.append(
            {
                "path": str(rel),
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
            "Strict path+SHA for transport-stable files; wheel/deb path+SHA retained "
            "as provenance only because Kaggle may re-materialize them; their usability "
            "is verified by network-free install/import gates"
        ),
        "files": rows,
        "archive_payload_provenance": archive_payloads,
    }
    (base.RUNTIME / "runtime-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"C3_OFFLINE_MANIFEST_V3_PASS strict_files={len(rows)} "
        f"archive_provenance={len(archive_payloads)}",
        flush=True,
    )


def _read_runtime_manifest_from_download(root: Path) -> dict | None:
    """Read the logical runtime manifest regardless of Kaggle's archive materialization."""
    manifests = list(root.rglob("runtime-manifest.json"))
    if len(manifests) == 1:
        try:
            return json.loads(manifests[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    archives = list(root.rglob("c3-cori-offline-runtime.tar.gz"))
    if len(archives) != 1:
        return None
    try:
        with tarfile.open(archives[0], "r:gz") as tf:
            member = tf.getmember("runtime-manifest.json")
            src = tf.extractfile(member)
            if src is None:
                return None
            return json.loads(src.read().decode("utf-8"))
    except (OSError, KeyError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def wait_for_kaggle_runtime_semantics(runtime_id: str) -> None:
    """Kaggle-only readiness gate based on logical identity, not archive bytes."""
    verify_root = base.ROOT / "runtime-kaggle-semantic-verify"

    for attempt in range(1, 31):
        shutil.rmtree(verify_root, ignore_errors=True)
        verify_root.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                runtime_id,
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
        manifest = _read_runtime_manifest_from_download(verify_root) if proc.returncode == 0 else None
        if (
            isinstance(manifest, dict)
            and manifest.get("schema") == base.RUNTIME_SCHEMA
            and manifest.get("matcha_commit") == base.MATCHA_COMMIT
            and manifest.get("network_required_at_kaggle_runtime") is False
        ):
            print(f"C3_OFFLINE_KAGGLE_LOGICAL_IDENTITY_PASS attempt={attempt}", flush=True)
            return
        print(f"C3_OFFLINE_KAGGLE_LOGICAL_IDENTITY_WAIT attempt={attempt}/30", flush=True)
        time.sleep(10)

    raise RuntimeError("Kaggle runtime Dataset never exposed the expected logical runtime identity")


def publish_runtime_v3(owner: str) -> str:
    runtime_id = BASE_PUBLISH_RUNTIME(owner)
    # Kaggle can re-materialize archive-like files. Do not compare archive bytes here.
    # This exemption is not used outside the Kaggle runtime transport path.
    wait_for_kaggle_runtime_semantics(runtime_id)
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


def write_kaggle_cpu_validator(source_path: Path, destination: Path) -> None:
    """Patch only the Kaggle copy of the validator.

    Kaggle has now changed byte identity for .tar.gz and .whl payloads in real runs.
    The v3 manifest excludes wheel/deb payloads from strict path+SHA verification, so
    this compatibility patch is now only a defense for older runtime manifests. The
    next stages must still successfully install archives offline and import the required
    packages. Non-archive payload hashes remain strict, including E280.

    The Kaggle-only copy also emits bounded dpkg diagnostics. This is deliberately a
    CPU gate: it exposes package-manager failure reasons before any T4 allocation.
    """
    source = source_path.read_text(encoding="utf-8")
    old = '''    for row in manifest.get("files", []):\n        path = runtime / row["path"]\n        if not path.is_file() or sha256_file(path) != row["sha256"]:\n            raise RuntimeError(f"runtime integrity mismatch: {row['path']}")\n    print("C3_OFFLINE_RUNTIME_SHA_PASS", flush=True)\n'''
    new = '''    kaggle_archive_exemptions = 0\n    for row in manifest.get("files", []):\n        rel = Path(str(row["path"]))\n        path = runtime / rel\n        if not path.is_file():\n            raise RuntimeError(f"runtime file missing: {rel}")\n        if sha256_file(path) != row["sha256"]:\n            if rel.parts and rel.parts[0] in {"wheelhouse", "debs"}:\n                kaggle_archive_exemptions += 1\n                print(f"C3_OFFLINE_KAGGLE_ARCHIVE_SHA_EXEMPT {rel}", flush=True)\n                continue\n            raise RuntimeError(f"runtime integrity mismatch: {rel}")\n    print(\n        f"C3_OFFLINE_RUNTIME_INTEGRITY_PASS kaggle_archive_exemptions={kaggle_archive_exemptions}",\n        flush=True,\n    )\n'''
    if old not in source:
        raise RuntimeError("Kaggle validator integrity patch anchor not found")
    patched = source.replace(old, new, 1)

    old_dpkg = '''        run(["dpkg", "-i", *debs])\n        run(["ldconfig"])\n'''
    new_dpkg = '''        for pkg in ["espeak-ng", "espeak-ng-data", "libespeak-ng1", "libpcaudio0", "libsonic0"]:\n            probe = subprocess.run(\n                ["dpkg-query", "-W", "-f=${Status} ${Version}", pkg],\n                text=True, capture_output=True, check=False,\n            )\n            state = (probe.stdout or probe.stderr).strip().replace("\\n", " | ")[:1000]\n            print(f"C3_OFFLINE_DPKG_BEFORE package={pkg} rc={probe.returncode} state={state}", flush=True)\n        proc = subprocess.run(\n            ["dpkg", "-i", *[str(path) for path in debs]],\n            text=True, capture_output=True, check=False,\n        )\n        if proc.returncode != 0:\n            stdout = (proc.stdout or "").replace("\\n", " | ")[:6000]\n            stderr = (proc.stderr or "").replace("\\n", " | ")[:6000]\n            print(f"C3_OFFLINE_DPKG_STDOUT rc={proc.returncode} {stdout}", flush=True)\n            print(f"C3_OFFLINE_DPKG_STDERR rc={proc.returncode} {stderr}", flush=True)\n            audit = subprocess.run(["dpkg", "--audit"], text=True, capture_output=True, check=False)\n            audit_text = ((audit.stdout or "") + " " + (audit.stderr or "")).replace("\\n", " | ")[:4000]\n            print(f"C3_OFFLINE_DPKG_AUDIT rc={audit.returncode} {audit_text}", flush=True)\n            raise RuntimeError(f"offline dpkg install failed rc={proc.returncode}")\n        run(["ldconfig"])\n'''
    if old_dpkg not in patched:
        raise RuntimeError("Kaggle validator dpkg patch anchor not found")
    patched = patched.replace(old_dpkg, new_dpkg, 1)

    destination.write_text(patched, encoding="utf-8")
    print("C3_OFFLINE_KAGGLE_VALIDATOR_PATCH_PASS", flush=True)


def build_validation_kernel_v3(owner: str, runtime_id: str) -> str:
    c3_dataset_id = resolve_training_dataset(owner)
    validator = Path("ops/kaggle_offline_cpu_validate.py")
    if not validator.is_file():
        raise RuntimeError("offline CPU validator script missing from repository")
    write_kaggle_cpu_validator(validator, base.KERNEL / "validate.py")
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
