from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from lightning_sdk import Teamspace

ORG = "jeonghun917-org"
TEAMSPACE = "default-project"
EXPECTED_SHA256 = "081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9"
SOURCE_RELATIVE = (
    "jobs/c3-cori-e270-e280-b16-oa018/artifacts/"
    "c3-cori-lightning-runs/cori-e100-to-e550-b16/checkpoints/checkpoint_epoch=279.ckpt"
)
LOCAL = Path("cori_e280_checkpoint_epoch_279.ckpt")
MODAL_VOLUME = "c3-speech-en-v1"
MODAL_REMOTE = "training/cori/lightning_e280/checkpoint_epoch=279.ckpt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not os.environ.get("LIGHTNING_USER_ID") or not os.environ.get("LIGHTNING_API_KEY"):
        raise RuntimeError("LIGHTNING_USER_ID / LIGHTNING_API_KEY are required")
    if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
        raise RuntimeError("MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are required")

    teamspace = Teamspace(name=TEAMSPACE, org=ORG)
    if LOCAL.exists():
        LOCAL.unlink()
    teamspace.download_file(SOURCE_RELATIVE, file_path=str(LOCAL))
    if not LOCAL.is_file():
        raise RuntimeError("Teamspace download returned without the E280 checkpoint")

    actual = sha256_file(LOCAL)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"E280 SHA mismatch: {actual} != {EXPECTED_SHA256}")

    subprocess.run(
        ["modal", "volume", "put", "--force", MODAL_VOLUME, str(LOCAL), MODAL_REMOTE],
        check=True,
    )
    print("C3_E280_DIRECT_STAGE_PASS", MODAL_REMOTE, EXPECTED_SHA256, flush=True)


if __name__ == "__main__":
    main()
