from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from lightning_sdk import Studio

ORG = "jeonghun917-org"
TEAMSPACE = "default-project"
STUDIO_NAME = "c3-asset-vault"
MODAL_VOLUME = "c3-speech-en-v1"

CHECKPOINTS = {
    "E200": {
        "vault": "C3_ASSET_VAULT/cori/matcha/E200/checkpoint_epoch=199__sha256_b3235e8bff23c6241119add85e57dccfa1e88ed2cf2ed51bed8a3c305dee5c54.ckpt",
        "modal": "training/cori/diagnostics/d01_seed_sweep/E200/checkpoint_epoch=199.ckpt",
        "sha256": "b3235e8bff23c6241119add85e57dccfa1e88ed2cf2ed51bed8a3c305dee5c54",
        "epoch": 200,
        "global_step": 100400,
    },
    "E280": {
        "vault": "C3_ASSET_VAULT/cori/matcha/E280/checkpoint_epoch=279__sha256_081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9.ckpt",
        "modal": "training/cori/diagnostics/d01_seed_sweep/E280/checkpoint_epoch=279.ckpt",
        "sha256": "081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9",
        "epoch": 280,
        "global_step": 140560,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    required = ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required credentials: {', '.join(missing)}")

    studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, org=ORG, create_ok=False)
    report = {
        "ok": True,
        "gpu_allocated": False,
        "lightning_compute_started": False,
        "checkpoints": {},
    }

    local_root = Path("cori_d01_stage")
    local_root.mkdir(parents=True, exist_ok=True)

    for label, meta in CHECKPOINTS.items():
        local = local_root / f"{label}.ckpt"
        if local.exists():
            local.unlink()
        studio.download_file(meta["vault"], file_path=str(local))
        if not local.is_file():
            raise RuntimeError(f"missing staged {label} checkpoint after Lightning download")
        actual = sha256_file(local)
        if actual != meta["sha256"]:
            raise RuntimeError(f"{label} SHA mismatch: {actual} != {meta['sha256']}")
        subprocess.run(
            ["modal", "volume", "put", "--force", MODAL_VOLUME, str(local), meta["modal"]],
            check=True,
        )
        report["checkpoints"][label] = {
            "epoch": meta["epoch"],
            "global_step": meta["global_step"],
            "sha256": actual,
            "modal_path": f"/vol/{meta['modal']}",
        }

    Path("cori-d01-stage-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("C3_D01_STAGE_PASS")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
