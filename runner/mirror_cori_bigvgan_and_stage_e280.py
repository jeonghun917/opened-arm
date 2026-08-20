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
VERIFY_GENERATOR = LOCAL_ROOT / "_verify" / "generator_final.pt"
VERIFY_CONFIG = LOCAL_ROOT / "_verify" / "config.json"
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


def studio_download(studio: Studio, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        local.unlink()
    studio.download_file(remote, file_path=str(local))
    if not local.is_file():
        raise RuntimeError(f"Lightning Studio download did not produce {local} from {remote}")


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

    # Studio file transfer works against persistent Studio storage without starting compute.
    studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, org=ORG, create_ok=False)
    print("Mirroring exact BigVGAN assets into persistent Lightning Studio storage...", flush=True)
    studio.upload_file(str(LOCAL_GENERATOR), VAULT_GENERATOR, progress_bar=False)
    studio.upload_file(str(LOCAL_CONFIG), VAULT_CONFIG, progress_bar=False)
    studio.upload_file(str(LOCAL_MANIFEST), VAULT_MANIFEST, progress_bar=False)

    # Read the mirrored bytes back out and verify them independently.
    studio_download(studio, VAULT_GENERATOR, VERIFY_GENERATOR)
    studio_download(studio, VAULT_CONFIG, VERIFY_CONFIG)
    verified_generator_sha = sha256_file(VERIFY_GENERATOR)
    verified_config_sha = sha256_file(VERIFY_CONFIG)
    if verified_generator_sha != generator_sha:
        raise RuntimeError(
            f"post-mirror generator SHA mismatch: {verified_generator_sha} != {generator_sha}"
        )
    if verified_config_sha != config_sha:
        raise RuntimeError(
            f"post-mirror config SHA mismatch: {verified_config_sha} != {config_sha}"
        )

    studio_download(studio, VAULT_E280, LOCAL_E280)
    e280_sha = sha256_file(LOCAL_E280)
    if e280_sha != E280_SHA256:
        raise RuntimeError(f"local E280 SHA mismatch: {e280_sha} != {E280_SHA256}")

    # Stage the SHA-verified E280 acoustic checkpoint beside the original Modal vocoder.
    subprocess.run(
        ["modal", "volume", "put", "--force", MODAL_VOLUME, str(LOCAL_E280), MODAL_E280],
        check=True,
    )

    report = {
        "ok": True,
        "gpu_allocated": False,
        "lightning_compute_started": False,
        "bigvgan_generator_sha256": generator_sha,
        "bigvgan_generator_bytes": LOCAL_GENERATOR.stat().st_size,
        "bigvgan_config_sha256": config_sha,
        "bigvgan_config_bytes": LOCAL_CONFIG.stat().st_size,
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
