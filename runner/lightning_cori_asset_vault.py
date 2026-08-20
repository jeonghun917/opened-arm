from __future__ import annotations

import json
import textwrap
from pathlib import Path

from lightning_sdk import Studio

ORG = "jeonghun917-org"
TEAMSPACE = "default-project"
STUDIO_NAME = "c3-asset-vault"
EXPECTED_E280_SHA = "081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9"

REMOTE = r'''
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

VAULT = Path('/teamspace/studios/c3-asset-vault/C3_ASSET_VAULT')
VAULT.mkdir(parents=True, exist_ok=True)
(VAULT / 'cori' / 'matcha').mkdir(parents=True, exist_ok=True)
(VAULT / 'cori' / 'bigvgan').mkdir(parents=True, exist_ok=True)

EXPECTED_E280_SHA = '081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9'
E280_ROOT = Path('/teamspace/jobs/c3-cori-e270-e280-b16-oa018')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_verified(src: Path, dst: Path, expected_sha: str | None = None) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    actual = sha256(src)
    if expected_sha and actual != expected_sha:
        raise RuntimeError(f'SHA mismatch for {src}: {actual} != {expected_sha}')
    if dst.exists():
        dst_sha = sha256(dst)
        if dst_sha != actual:
            raise RuntimeError(f'vault collision at {dst}: {dst_sha} != {actual}')
    else:
        shutil.copy2(src, dst)
    final = sha256(dst)
    if final != actual:
        raise RuntimeError(f'post-copy SHA mismatch for {dst}')
    return {
        'source': str(src),
        'vault': str(dst),
        'bytes': src.stat().st_size,
        'sha256': actual,
    }


def find_named(root: Path, name: str, limit: int = 100) -> list[str]:
    if not root.exists():
        return []
    out = []
    for p in root.rglob(name):
        out.append(str(p))
        if len(out) >= limit:
            break
    return out


report = {
    'schema': 'c3-cori-asset-vault-v1',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'studio': 'c3-asset-vault',
    'vault_root': str(VAULT),
    'e280': {'status': 'missing'},
    'adapted_bigvgan': {'status': 'not_found'},
    'discovery': {},
}

# 1) Recover the accepted E280 checkpoint from the Studio-visible managed-job mount.
e280_candidates = find_named(E280_ROOT, 'checkpoint_epoch=279.ckpt', limit=20)
report['discovery']['e280_candidates'] = e280_candidates
for raw in e280_candidates:
    p = Path(raw)
    try:
        if sha256(p) == EXPECTED_E280_SHA:
            dst = VAULT / 'cori' / 'matcha' / 'E280' / f'checkpoint_epoch=279__sha256_{EXPECTED_E280_SHA}.ckpt'
            report['e280'] = {'status': 'archived', **copy_verified(p, dst, EXPECTED_E280_SHA)}
            break
    except OSError as exc:
        report.setdefault('errors', []).append(f'e280 candidate {p}: {exc!r}')

# 2) Search Lightning shared storage for an exact Cori-adapted BigVGAN run copy.
search_roots = [Path('/teamspace/studios'), Path('/teamspace/jobs')]
markers = []
for root in search_roots:
    if not root.exists():
        continue
    for p in root.rglob('*'):
        s = str(p).lower()
        if ('bigvgan_base_cori_22k80' in s or 'vocoder_adaptation' in s) and 'c3_asset_vault' not in s:
            markers.append(str(p))
            if len(markers) >= 200:
                break
    if len(markers) >= 200:
        break
report['discovery']['bigvgan_markers'] = markers

# Prefer the exact historical run marker when present.
run_roots = []
for raw in markers:
    p = Path(raw)
    s = str(p)
    if 'bigvgan_base_cori_22k80' in s and '20260817T022729Z' in s:
        cur = p if p.is_dir() else p.parent
        while cur.name and cur.name != '20260817T022729Z':
            cur = cur.parent
        if cur.name == '20260817T022729Z' and cur not in run_roots:
            run_roots.append(cur)

report['discovery']['exact_bigvgan_run_roots'] = [str(x) for x in run_roots]

if run_roots:
    src_root = run_roots[0]
    dst_root = VAULT / 'cori' / 'bigvgan' / 'adapted_20260817T022729Z'
    files = []
    # Archive model/config/metadata files only; no source audio or dataset material.
    preferred_names = {
        'generator_final.pt', 'generator.pt', 'config.json', 'args.json',
        'hparams.json', 'training_args.json', 'README.md', 'STATS.json',
    }
    for p in src_root.rglob('*'):
        if not p.is_file():
            continue
        low = p.name.lower()
        if p.name in preferred_names or low.endswith(('.json', '.yaml', '.yml')) or 'generator' in low:
            rel = p.relative_to(src_root)
            files.append(copy_verified(p, dst_root / rel))
    report['adapted_bigvgan'] = {
        'status': 'archived' if files else 'run_found_no_selected_files',
        'source_root': str(src_root),
        'vault_root': str(dst_root),
        'files': files,
    }

# 3) Record anchor-checkpoint discovery without hashing/copying every historical file.
anchors = {}
for name in ('checkpoint_epoch=099.ckpt', 'checkpoint_epoch=199.ckpt'):
    found = []
    for root in search_roots:
        found.extend(find_named(root, name, limit=max(0, 20 - len(found))))
        if len(found) >= 20:
            break
    anchors[name] = found
report['discovery']['anchor_checkpoint_candidates'] = anchors

manifest_path = VAULT / 'C3_ASSET_VAULT_MANIFEST.json'
manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('C3_ASSET_VAULT_REPORT_BEGIN')
print(json.dumps(report, ensure_ascii=False, indent=2))
print('C3_ASSET_VAULT_REPORT_END')
'''


def main() -> None:
    studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, org=ORG, create_ok=True)
    print(f"Starting Lightning Studio {STUDIO_NAME!r} on the SDK default CPU machine...", flush=True)
    studio.start()
    try:
        cmd = "python - <<'PY'\n" + textwrap.dedent(REMOTE) + "\nPY"
        output = studio.run(cmd)
        print(output, flush=True)
        Path('lightning-cori-asset-vault-report.txt').write_text(output + '\n', encoding='utf-8')
    finally:
        print(f"Stopping Lightning Studio {STUDIO_NAME!r}...", flush=True)
        studio.stop()


if __name__ == '__main__':
    main()
