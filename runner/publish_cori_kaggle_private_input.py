from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import textwrap
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from lightning_sdk import Machine, Studio

ORG = "jeonghun917-org"
TEAMSPACE = "default-project"
SOURCE_STUDIO_NAME = "c3-cori-e100-e200"
VAULT_STUDIO_NAME = "c3-asset-vault"

E280_SHA256 = "081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9"
VAULT_E280 = (
    "C3_ASSET_VAULT/cori/matcha/E280/"
    "checkpoint_epoch=279__sha256_081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9.ckpt"
)

LOCAL_ROOT = Path("cori_kaggle_private_stage")
LOCAL_BASE_ZIP = LOCAL_ROOT / "c3-cori-base.zip"
LOCAL_E280 = LOCAL_ROOT / "checkpoint_epoch=279.ckpt"
LOCAL_E280_ZIP = LOCAL_ROOT / "c3-cori-e280.zip"
UPLOAD_ROOT = LOCAL_ROOT / "upload"
REMOTE_BASE_ZIP = "C3_KAGGLE_STAGING/c3-cori-base.zip"
REMOTE_MANIFEST = "C3_KAGGLE_STAGING/c3-cori-base-manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_cpu_ready(studio: Studio) -> None:
    if str(studio.status).lower() not in {"running", "started"}:
        studio.start(Machine.CPU)
    last = None
    for _ in range(60):
        try:
            out = str(studio.run("echo C3_KAGGLE_SOURCE_STUDIO_READY"))
            if "C3_KAGGLE_SOURCE_STUDIO_READY" in out:
                return
        except Exception as exc:  # provider transition states
            last = exc
        time.sleep(5)
    raise RuntimeError(f"source Studio never became shell-ready: {last!r}")


def stop_studio(studio: Studio) -> None:
    try:
        if str(studio.status).lower() not in {"stopped", "stopping"}:
            studio.stop()
    except Exception as exc:
        print("C3_KAGGLE_SOURCE_STUDIO_STOP_WARNING", type(exc).__name__, flush=True)


def remote_build_base_bundle(studio: Studio) -> None:
    remote = r'''
from __future__ import annotations
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HOME = Path('/teamspace/studios/this_studio')
ROOT = HOME / 'c3-migration' / 'c3-stage-src'
HANDOFF = ROOT / 'handoff'
STAGING = HOME / 'C3_KAGGLE_STAGING'
OUT = STAGING / 'c3-cori-base.zip'
MANIFEST = STAGING / 'c3-cori-base-manifest.json'
STAGING.mkdir(parents=True, exist_ok=True)

required = [
    HANDOFF / 'cori_matcha_epoch100.ckpt',
    HANDOFF / 'HANDOFF_MANIFEST.json',
    HANDOFF / 'metadata' / 'FREEZE.json',
    HANDOFF / 'metadata' / 'PREPARED.json',
    HANDOFF / 'metadata' / 'STATS.json',
    HANDOFF / 'filelists' / 'train.txt',
    HANDOFF / 'filelists' / 'valid.txt',
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise RuntimeError('handoff incomplete: ' + ', '.join(missing))

freeze_hits = list((ROOT / 'cori_dataset').glob('**/metadata/FREEZE.json'))
if len(freeze_hits) != 1:
    raise RuntimeError(f'expected exactly one frozen dataset root; FREEZE hits={len(freeze_hits)}')
DATASET = freeze_hits[0].parent.parent

# Keep the archive deterministic enough for operational verification: sorted paths,
# no compression, no transient caches. Source audio and manifests remain private.
def iter_files(root: Path):
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        if '__pycache__' in p.parts or p.name.endswith(('.pyc', '.tmp')):
            continue
        yield p

rows = []
with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for prefix, root in [('c3-cori-handoff', HANDOFF), ('c3-cori-dataset', DATASET)]:
        for p in iter_files(root):
            rel = Path(prefix) / p.relative_to(root)
            zf.write(p, arcname=str(rel))
            rows.append({'path': str(rel), 'bytes': p.stat().st_size})

h = hashlib.sha256()
with OUT.open('rb') as f:
    for b in iter(lambda: f.read(1024 * 1024), b''):
        h.update(b)
manifest = {
    'schema': 'c3-cori-kaggle-private-base-v1',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'dataset_root': str(DATASET),
    'file_count': len(rows),
    'payload_bytes': sum(x['bytes'] for x in rows),
    'archive_bytes': OUT.stat().st_size,
    'archive_sha256': h.hexdigest(),
}
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('C3_KAGGLE_BASE_BUNDLE_PASS')
print(json.dumps(manifest, ensure_ascii=False))
'''
    command = "python - <<'PY'\n" + textwrap.dedent(remote) + "\nPY"
    out, rc = studio.run_with_exit_code(command)
    print(out, flush=True)
    if rc != 0 or "C3_KAGGLE_BASE_BUNDLE_PASS" not in str(out):
        raise RuntimeError(f"remote base bundle failed with exit code {rc}")


def download_file(studio: Studio, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        local.unlink()
    studio.download_file(remote, file_path=str(local))
    if not local.is_file():
        raise RuntimeError(f"Lightning download did not produce {local} from {remote}")


def build_e280_zip() -> None:
    if sha256_file(LOCAL_E280) != E280_SHA256:
        raise RuntimeError("E280 SHA mismatch before Kaggle staging")
    with zipfile.ZipFile(LOCAL_E280_ZIP, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.write(LOCAL_E280, arcname="c3-cori-e280/checkpoint_epoch=279.ckpt")
        zf.writestr(
            "c3-cori-e280/IDENTITY.json",
            json.dumps(
                {
                    "semantic_epoch": 280,
                    "global_step": 140560,
                    "sha256": E280_SHA256,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )


def kaggle_auth_present() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def publish_private_dataset(dataset_id: str) -> str:
    if not kaggle_auth_present():
        raise RuntimeError("Kaggle authentication is missing")
    if dataset_id.count("/") != 1:
        raise RuntimeError("KAGGLE_DATASET_ID must be owner/private-dataset-slug")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LOCAL_BASE_ZIP, UPLOAD_ROOT / LOCAL_BASE_ZIP.name)
    shutil.copy2(LOCAL_E280_ZIP, UPLOAD_ROOT / LOCAL_E280_ZIP.name)
    metadata = {
        "title": "C3 Cori Private Training Input",
        "id": dataset_id,
        "licenses": [{"name": "other"}],
        "description": (
            "Private research-compute staging bundle for the C3 Cori Matcha-TTS continuation experiment. "
            "Underlying source-data and model licenses remain controlling; this private staging copy does not grant a new license."
        ),
    }
    (UPLOAD_ROOT / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    subprocess.run(["kaggle", "datasets", "list", "-m", "--page", "1"], stdout=subprocess.DEVNULL, check=True)
    exists = subprocess.run(
        ["kaggle", "datasets", "status", dataset_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if exists:
        subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(UPLOAD_ROOT), "-m", "Refresh frozen E280 training input", "-q", "-t"],
            check=True,
        )
        action = "versioned"
    else:
        # Kaggle datasets are private by default; do not pass --public.
        subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(UPLOAD_ROOT), "-q", "-t"],
            check=True,
        )
        action = "created_private"
    subprocess.run(["kaggle", "datasets", "status", dataset_id], check=True)
    return action


def main() -> None:
    required = ["LIGHTNING_USER_ID", "LIGHTNING_API_KEY", "KAGGLE_DATASET_ID"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing required environment values: " + ", ".join(missing))
    if not kaggle_auth_present():
        raise RuntimeError("missing Kaggle auth: set KAGGLE_API_TOKEN or KAGGLE_USERNAME + KAGGLE_KEY")

    dataset_id = os.environ["KAGGLE_DATASET_ID"].strip()
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    source = Studio(name=SOURCE_STUDIO_NAME, teamspace=TEAMSPACE, org=ORG, create_ok=False)
    vault = Studio(name=VAULT_STUDIO_NAME, teamspace=TEAMSPACE, org=ORG, create_ok=False)
    try:
        # E280 can be transferred from persistent vault storage without starting GPU compute.
        download_file(vault, VAULT_E280, LOCAL_E280)
        if sha256_file(LOCAL_E280) != E280_SHA256:
            raise RuntimeError("downloaded vault E280 SHA mismatch")
        build_e280_zip()

        # Only the source Studio needs CPU briefly to package the frozen handoff+audio tree.
        ensure_cpu_ready(source)
        remote_build_base_bundle(source)
        download_file(source, REMOTE_BASE_ZIP, LOCAL_BASE_ZIP)

        action = publish_private_dataset(dataset_id)
        report = {
            "schema": "c3-cori-kaggle-private-publish-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "ok": True,
            "gpu_allocated": False,
            "dataset_id": dataset_id,
            "kaggle_action": action,
            "e280_sha256": E280_SHA256,
            "base_archive_bytes": LOCAL_BASE_ZIP.stat().st_size,
            "base_archive_sha256": sha256_file(LOCAL_BASE_ZIP),
            "e280_archive_bytes": LOCAL_E280_ZIP.stat().st_size,
            "e280_archive_sha256": sha256_file(LOCAL_E280_ZIP),
        }
        Path("cori-kaggle-private-publish-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("C3_KAGGLE_PRIVATE_PUBLISH_PASS")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        stop_studio(source)
        # Sensitive local staging is ephemeral and must not become an Actions artifact.
        shutil.rmtree(LOCAL_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
