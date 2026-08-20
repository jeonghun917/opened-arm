from __future__ import annotations

import textwrap
from pathlib import Path

from lightning_sdk import Studio

ORG = "jeonghun917-org"
TEAMSPACE = "default-project"
STUDIO_NAME = "c3-asset-vault"

REMOTE = r'''
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

STUDIO_HOME = Path('/teamspace/studios/this_studio')
VAULT = STUDIO_HOME / 'C3_ASSET_VAULT'
VAULT.mkdir(parents=True, exist_ok=True)
(VAULT / 'cori' / 'matcha').mkdir(parents=True, exist_ok=True)
(VAULT / 'cori' / 'bigvgan').mkdir(parents=True, exist_ok=True)

EXPECTED_E280_SHA = '081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9'
E280_EXACT = Path('/teamspace/jobs/c3-cori-e270-e280-b16-oa018/artifacts/c3-cori-lightning-runs/cori-e100-to-e550-b16/checkpoints/checkpoint_epoch=279.ckpt')


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
    if sha256(dst) != actual:
        raise RuntimeError(f'post-copy SHA mismatch for {dst}')
    return {'source': str(src), 'vault': str(dst), 'bytes': src.stat().st_size, 'sha256': actual}


def bounded_find(root: str, *, kind: str, name: str, maxdepth: int = 8, timeout: int = 30) -> list[str]:
    cmd = ['find', root, '-maxdepth', str(maxdepth), '-type', kind, '-name', name, '-print']
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        lines = [x for x in cp.stdout.splitlines() if x.strip()]
        if cp.returncode not in (0, 1):
            lines.append(f'__FIND_RC_{cp.returncode}__:{cp.stderr[-500:]}')
        return lines[:100]
    except subprocess.TimeoutExpired:
        return [f'__FIND_TIMEOUT__:{root}:{name}']


report = {
    'schema': 'c3-cori-asset-vault-v1',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'studio': 'c3-asset-vault',
    'studio_home': str(STUDIO_HOME),
    'vault_root': str(VAULT),
    'e280': {'status': 'missing'},
    'adapted_bigvgan': {'status': 'not_found'},
    'discovery': {},
}

# Directly use the exact managed-job mount path previously accepted by the controller.
report['discovery']['e280_exact_exists'] = E280_EXACT.exists()
if E280_EXACT.exists():
    dst = VAULT / 'cori' / 'matcha' / 'E280' / f'checkpoint_epoch=279__sha256_{EXPECTED_E280_SHA}.ckpt'
    report['e280'] = {'status': 'archived', **copy_verified(E280_EXACT, dst, EXPECTED_E280_SHA)}

# Keep the expensive search bounded. We only need to know whether the historical adapted run was copied to Lightning.
studio_children = []
try:
    studio_children = sorted(p.name for p in Path('/teamspace/studios').iterdir())
except Exception as exc:
    report.setdefault('errors', []).append(f'list /teamspace/studios: {exc!r}')
report['discovery']['studio_children'] = studio_children

bigvgan_dirs = []
for root in ('/teamspace/studios', '/teamspace/jobs'):
    bigvgan_dirs.extend(bounded_find(root, kind='d', name='bigvgan_base_cori_22k80', maxdepth=8, timeout=25))
report['discovery']['bigvgan_base_dirs'] = bigvgan_dirs

run_roots = []
for raw in bigvgan_dirs:
    if raw.startswith('__'):
        continue
    base = Path(raw)
    exact = base / '20260817T022729Z'
    if exact.is_dir():
        run_roots.append(exact)
report['discovery']['exact_bigvgan_run_roots'] = [str(x) for x in run_roots]

if run_roots:
    src_root = run_roots[0]
    dst_root = VAULT / 'cori' / 'bigvgan' / 'adapted_20260817T022729Z'
    files = []
    preferred_names = {'generator_final.pt', 'generator.pt', 'config.json', 'args.json', 'hparams.json', 'training_args.json', 'README.md', 'STATS.json'}
    for p in src_root.rglob('*'):
        if not p.is_file():
            continue
        low = p.name.lower()
        if p.name in preferred_names or low.endswith(('.json', '.yaml', '.yml')) or 'generator' in low:
            files.append(copy_verified(p, dst_root / p.relative_to(src_root)))
    report['adapted_bigvgan'] = {
        'status': 'archived' if files else 'run_found_no_selected_files',
        'source_root': str(src_root),
        'vault_root': str(dst_root),
        'files': files,
    }

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
        output, exit_code = studio.run_with_exit_code(cmd)
        print(output, flush=True)
        Path('lightning-cori-asset-vault-report.txt').write_text(output + '\n', encoding='utf-8')
        if exit_code != 0:
            raise RuntimeError(f'remote inspection failed with exit code {exit_code}')
    finally:
        print(f"Stopping Lightning Studio {STUDIO_NAME!r}...", flush=True)
        studio.stop()


if __name__ == '__main__':
    main()
