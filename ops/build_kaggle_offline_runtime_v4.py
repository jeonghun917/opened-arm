from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import build_kaggle_offline_runtime as base
import build_kaggle_offline_runtime_v3 as v3

V3_WRITE_MANIFEST = v3.write_manifest_v3

JAMMY_SYSTEM_PACKAGES = [
    "espeak-ng",
    "espeak-ng-data",
    "libespeak-ng1",
    "libpcaudio0",
    "libsonic0",
    "libasound2",
    "libasound2-data",
]


def _apt_options(source_list: Path, lists_dir: Path) -> list[str]:
    return [
        "-o", f"Dir::Etc::sourcelist={source_list.resolve()}",
        "-o", "Dir::Etc::sourceparts=-",
        "-o", f"Dir::State::lists={lists_dir.resolve()}",
        "-o", "APT::Get::List-Cleanup=0",
        "-o", "Acquire::Languages=none",
    ]


def build_jammy_debs() -> None:
    """Build the system bundle against Kaggle's observed Ubuntu 22.04 ABI.

    The GitHub builder runs Ubuntu 24.04, but the Kaggle image observed by the CPU
    gate uses glibc 2.35. Noble espeak dependencies therefore cannot be installed
    there (notably libasound2t64 requires a newer libc). Fetch Jammy packages into
    the private offline runtime instead; Kaggle itself remains network-disabled.
    """
    debs = base.RUNTIME / "debs"
    debs.mkdir(parents=True, exist_ok=True)

    apt_root = base.ROOT / "apt-jammy"
    lists_dir = apt_root / "lists"
    (lists_dir / "partial").mkdir(parents=True, exist_ok=True)
    source_list = apt_root / "sources.list"
    keyring = "/usr/share/keyrings/ubuntu-archive-keyring.gpg"
    source_list.write_text(
        "\n".join(
            [
                f"deb [arch=amd64 signed-by={keyring}] http://archive.ubuntu.com/ubuntu jammy main universe",
                f"deb [arch=amd64 signed-by={keyring}] http://archive.ubuntu.com/ubuntu jammy-updates main universe",
                f"deb [arch=amd64 signed-by={keyring}] http://security.ubuntu.com/ubuntu jammy-security main universe",
                "",
            ]
        ),
        encoding="utf-8",
    )
    opts = _apt_options(source_list, lists_dir)
    base.run(["apt-get", *opts, "update", "-qq"])

    for package in JAMMY_SYSTEM_PACKAGES:
        base.run(["apt-get", *opts, "download", package], cwd=debs)

    downloaded = sorted(debs.glob("*.deb"))
    if len(downloaded) < len(JAMMY_SYSTEM_PACKAGES):
        raise RuntimeError(
            f"Jammy system bundle incomplete packages={len(JAMMY_SYSTEM_PACKAGES)} debs={len(downloaded)}"
        )

    metadata = []
    for path in downloaded:
        proc = subprocess.run(
            ["dpkg-deb", "-f", str(path), "Package", "Version", "Depends"],
            text=True,
            capture_output=True,
            check=True,
        )
        fields = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        metadata.append({"file": path.name, "fields": fields})
        joined = " ".join(fields)
        if "libc6 (>= 2.38" in joined:
            raise RuntimeError(f"Jammy bundle unexpectedly requires glibc 2.38+: {path.name}")

    (base.RUNTIME / "jammy-system-bundle.json").write_text(
        json.dumps(
            {
                "schema": "c3-kaggle-jammy-system-bundle-v1",
                "suite": "ubuntu-jammy",
                "architecture": "amd64",
                "packages": JAMMY_SYSTEM_PACKAGES,
                "artifacts": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"C3_OFFLINE_JAMMY_DEB_BUNDLE_PASS packages={len(JAMMY_SYSTEM_PACKAGES)} "
        f"debs={len(downloaded)}",
        flush=True,
    )


def write_manifest_v4() -> None:
    V3_WRITE_MANIFEST()
    path = base.RUNTIME / "runtime-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["system_bundle_suite"] = "ubuntu-jammy"
    manifest["system_bundle_architecture"] = "amd64"
    manifest["system_bundle_packages"] = JAMMY_SYSTEM_PACKAGES
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("C3_OFFLINE_MANIFEST_V4_JAMMY_PASS", flush=True)


def main() -> int:
    # Patch only the builder side. Kaggle runtime stays network-free and validates
    # the resulting packages by dpkg + ldconfig + phonemizer/import gates.
    base.build_debs = build_jammy_debs
    v3.write_manifest_v3 = write_manifest_v4
    return v3.main()


if __name__ == "__main__":
    raise SystemExit(main())
