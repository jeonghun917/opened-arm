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

EXPECTED = {
    'E100': 'f4409103780820e356b609ec79c425cb1cffd3059fed163e1f60bfe926438273',
    'E200': 'b3235e8bff23c6241119add85e57dccfa1e88ed2cf2ed51bed8a3c305dee5c54',
    'E280': '081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_verified(src: Path, dst: Path, expected_sha: str) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    actual = sha256(src)
    if actual != expected_sha:
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


def bounded_find(root: str, args: list[str], timeout: int = 25, limit: int = 100) -> list[str]:
    cmd = ['find', root] + args + ['-print']
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        lines = [x for x in cp.stdout.splitlines() if x.strip()]
        if cp.returncode not in (0, 1):
            lines.append(f'__FIND_RC_{cp.returncode}__:{cp.stderr[-500:]}')
        return lines[:limit]
    except subprocess.TimeoutExpired:
        return [f'__FIND_TIMEOUT__:{root}:{" ".join(args)}']


def archive_first_matching(label: str, candidates: list[str]) -> dict:
    expected = EXPECTED[label]
    checked = []
    for raw in candidates:
        if raw.startswith('__'):
            continue
        p = Path(raw)
        if not p.is_file():
            continue
        try:
            actual = sha256(p)
            checked.append({'path': str(p), 'sha256': actual})
            if actual == expected:
                filename = f'{p.name}__sha256_{expected}.ckpt' if not p.name.endswith('.ckpt') else f'{p.stem}__sha256_{expected}.ckpt'
                dst = VAULT / 'cori' / 'matcha' / label / filename
                return {'status': 'archived', **copy_verified(p, dst, expected), 'checked': checked}
        except OSError as exc:
            checked.append({'path': str(p), 'error': repr(exc)})
    return {'status': 'not_found', 'expected_sha256': expected, 'checked': checked}


report = {
    'schema': 'c3-cori-asset-vault-v2',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'studio': 'c3-asset-vault',
    'studio_home': str(STUDIO_HOME),
    'vault_root': str(VAULT),
    'anchors': {},
    'adapted_bigvgan': {'status': 'not_found'},
    'discovery': {},
}

# E280: exact accepted managed-job path.
e280_exact = '/teamspace/jobs/c3-cori-e270-e280-b16-oa018/artifacts/c3-cori-lightning-runs/cori-e100-to-e550-b16/checkpoints/checkpoint_epoch=279.ckpt'
report['anchors']['E280'] = archive_first_matching('E280', [e280_exact])

# E200: bounded search only inside the known accepted E190->E200 job.
e200_candidates = bounded_find(
    '/teamspace/jobs/c3-cori-e190-e200-b16-sh010',
    ['-maxdepth', '10', '-type', 'f', '-name', 'checkpoint_epoch=199.ckpt'],
    timeout=20,
)
report['discovery']['e200_candidates'] = e200_candidates
report['anchors']['E200'] = archive_first_matching('E200', e200_candidates)

# E100: historical handoff landed in the c3-cori-e100-e200 Studio. Check only likely checkpoint names there.
e100_candidates = []
for name in ('checkpoint_epoch=099.ckpt', 'cori_matcha_epoch100.ckpt'):
    e100_candidates += bounded_find(
        '/teamspace/studios/c3-cori-e100-e200',
        ['-maxdepth', '10', '-type', 'f', '-name', name],
        timeout=20,
    )
report['discovery']['e100_candidates'] = e100_candidates
report['anchors']['E100'] = archive_first_matching('E100', e100_candidates)

# Targeted vocoder inspection: the historical handoff Studio and the accepted E200 job only.
vocoder_candidates = []
for root in ('/teamspace/studios/c3-cori-e100-e200', '/teamspace/jobs/c3-cori-e190-e200-b16-sh010'):
    vocoder_candidates += bounded_find(
        root,
        ['-maxdepth', '10', '(', '-iname', '*bigvgan*', '-o', '-iname', '*vocoder*', ')'],
        timeout=20,
        limit=100,
    )
report['discovery']['vocoder_candidates'] = vocoder_candidates

# Archive only if the exact historical adapted run appears. Never substitute ONNX/runtime exports silently.
run_roots = []
for raw in vocoder_candidates:
    if raw.startswith('__'):
        continue
    p = Path(raw)
    if 'bigvgan_base_cori_22k80' in str(p) and '20260817T022729Z' in str(p):
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
    for p in src_root.rglob('*'):
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.endswith(('.json', '.yaml', '.yml')) or 'generator' in low or p.name in {'README.md', 'STATS.json'}:
            actual = sha256(p)
            dst = dst_root / p.relative_to(src_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(p, dst)
            if sha256(dst) != actual:
                raise RuntimeError(f'BigVGAN post-copy SHA mismatch: {dst}')
            files.append({'source': str(p), 'vault': str(dst), 'bytes': p.stat().st_size, 'sha256': actual})
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
