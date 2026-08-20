from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from lightning_sdk import Studio

ORG = "jeonghun917-org"
TEAMSPACE = "default-project"
STUDIO_NAME = "c3-asset-vault"
MODAL_VOLUME = "c3-speech-en-v1"

ADAPT_RUN = "training/cori/vocoder_adaptation/bigvgan_base_cori_22k80/20260817T022729Z"
MODAL_GENERATOR = f"{ADAPT_RUN}/generator_final.pt"
MODAL_CONFIG = f"{ADAPT_RUN}/checkpoints/config.json"

LOCAL_ROOT = Path("cori_exact_assets")
LOCAL_GENERATOR = LOCAL_ROOT / "generator_final.pt"
LOCAL_CONFIG = LOCAL_ROOT / "checkpoints" / "config.json"
LOCAL_MANIFEST = LOCAL_ROOT / "MIRROR_MANIFEST.json"
LOCAL_E280 = Path("cori_e280_checkpoint_epoch_279.ckpt")

VAULT_BIGVGAN_ROOT = "C3_ASSET_VAULT/cori/bigvgan/adapted_20260817T022729Z"
VAULT_GENERATOR = f"{VAULT_BIGVGAN_ROOT}/generator_final.pt"
VAULT_CONFIG = f"{VAULT_BIGVGAN_ROOT}/checkpoints/config.json"
VAULT_MANIFEST = f"{VAULT_BIGVGAN_ROOT}/MIRROR_MANIFEST.json"

E280_SHA256 = "081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9"
VAULT_E280 = (
    "C3_ASSET_VAULT/cori/matcha/E280/"
    "checkpoint_epoch=279__sha256_081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9.ckpt"
)
MODAL_E280 = "training/cori/lightning_e280/checkpoint_epoch=279.ckpt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def modal_get(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        local.unlink()
    subprocess.run(
        ["modal", "volume", "get", "--force", MODAL_VOLUME, remote, str(local)],
        check=True,
    )
    if not local.is_file():
        raise RuntimeError(f"Modal download did not produce {local} from {remote}")


def main() -> None:
    required = (
        "LIGHTNING_USER_ID",
        "LIGHTNING_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required credentials: {', '.join(missing)}")

    # Pull only the exact frozen vocoder assets required by the established evaluation path.
    modal_get(MODAL_GENERATOR, LOCAL_GENERATOR)
    modal_get(MODAL_CONFIG, LOCAL_CONFIG)

    generator_sha = sha256_file(LOCAL_GENERATOR)
    config_sha = sha256_file(LOCAL_CONFIG)
    manifest = {
        "schema": "c3-cori-bigvgan-mirror-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_provider": "Modal",
        "source_volume": MODAL_VOLUME,
        "source_run": f"/vol/{ADAPT_RUN}",
        "files": [
            {
                "source": f"/vol/{MODAL_GENERATOR}",
                "vault": VAULT_GENERATOR,
                "bytes": LOCAL_GENERATOR.stat().st_size,
                "sha256": generator_sha,
            },
            {
                "source": f"/vol/{MODAL_CONFIG}",
                "vault": VAULT_CONFIG,
                "bytes": LOCAL_CONFIG.stat().st_size,
                "sha256": config_sha,
            },
        ],
    }
    LOCAL_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, org=ORG, create_ok=False)
    print(f"Starting Lightning CPU Studio {STUDIO_NAME!r} for verified asset transfer...", flush=True)
    studio.start()
    try:
        studio.upload_file(str(LOCAL_GENERATOR), VAULT_GENERATOR, progress_bar=False)
        studio.upload_file(str(LOCAL_CONFIG), VAULT_CONFIG, progress_bar=False)
        studio.upload_file(str(LOCAL_MANIFEST), VAULT_MANIFEST, progress_bar=False)

        verify_py = f"""
import hashlib, json
from pathlib import Path

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

checks={{
  'generator': ('{VAULT_GENERATOR}', '{generator_sha}'),
  'config': ('{VAULT_CONFIG}', '{config_sha}'),
  'e280': ('{VAULT_E280}', '{E280_SHA256}'),
}}
result={{}}
for name,(rel,expected) in checks.items():
    p=Path.home()/rel
    if not p.is_file(): raise SystemExit(f'missing vault asset: {{p}}')
    actual=sha256(p)
    if actual != expected: raise SystemExit(f'SHA mismatch {{name}}: {{actual}} != {{expected}}')
    result[name]={{'path':str(p),'bytes':p.stat().st_size,'sha256':actual}}
print(json.dumps(result, indent=2))
"""
        remote_output = studio.run("python - <<'PY'\n" + verify_py + "\nPY")
        print("C3_LIGHTNING_VAULT_VERIFY_BEGIN", flush=True)
        print(remote_output, flush=True)
        print("C3_LIGHTNING_VAULT_VERIFY_END", flush=True)

        if LOCAL_E280.exists():
            LOCAL_E280.unlink()
        studio.download_file(VAULT_E280, file_path=str(LOCAL_E280), progress_bar=False)
    finally:
        print(f"Stopping Lightning CPU Studio {STUDIO_NAME!r}...", flush=True)
        studio.stop()

    if not LOCAL_E280.is_file():
        raise RuntimeError("Lightning Studio download did not produce E280")
    e280_sha = sha256_file(LOCAL_E280)
    if e280_sha != E280_SHA256:
        raise RuntimeError(f"local E280 SHA mismatch: {e280_sha} != {E280_SHA256}")

    subprocess.run(
        ["modal", "volume", "put", "--force", MODAL_VOLUME, str(LOCAL_E280), MODAL_E280],
        check=True,
    )

    report = {
        "ok": True,
        "gpu_allocated": False,
        "bigvgan_generator_sha256": generator_sha,
        "bigvgan_config_sha256": config_sha,
        "e280_sha256": e280_sha,
        "lightning_vault_root": VAULT_BIGVGAN_ROOT,
        "modal_e280": f"/vol/{MODAL_E280}",
    }
    Path("cori-asset-transfer-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("C3_CORI_ASSET_TRANSFER_PASS")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
