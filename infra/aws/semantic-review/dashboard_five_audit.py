#!/usr/bin/env python3
import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

SOURCE_REPO = 'jeonghun917/dashboard-control-center'
MAX_CODE_BYTES = 36_000
MAX_PART_BYTES = 30_000
MAX_CHUNKS = 4


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True)


def split_patch(path: str, patch: str):
    header = f'=== {path} ===\n'
    whole = header + patch
    if len(whole.encode()) <= MAX_PART_BYTES:
        return [(path, whole)]
    lines = patch.splitlines(True)
    parts = []
    current = []
    size = len(header.encode()) + 80
    for line in lines:
        line_bytes = len(line.encode())
        if line_bytes > MAX_PART_BYTES - 500:
            raise SystemExit(f'single_line_too_large:{path}')
        if current and size + line_bytes > MAX_PART_BYTES:
            parts.append(''.join(current))
            current = []
            size = len(header.encode()) + 80
        current.append(line)
        size += line_bytes
    if current:
        parts.append(''.join(current))
    return [
        (path, f'=== {path} [part {index}/{len(parts)}] ===\n{part}')
        for index, part in enumerate(parts, 1)
    ]


def chunk_blocks(blocks):
    chunks = []
    texts = []
    paths = []
    size = 0
    for path, block in blocks:
        block_bytes = len(block.encode()) + (2 if texts else 0)
        if block_bytes > MAX_CODE_BYTES:
            raise SystemExit(f'block_too_large:{path}')
        if texts and size + block_bytes > MAX_CODE_BYTES:
            chunks.append({
                'code': '\n\n'.join(texts),
                'paths': list(dict.fromkeys(paths)),
                'bytes': size,
            })
            texts = []
            paths = []
            size = 0
            block_bytes = len(block.encode())
        texts.append(block)
        paths.append(path)
        size += block_bytes
    if texts:
        chunks.append({
            'code': '\n\n'.join(texts),
            'paths': list(dict.fromkeys(paths)),
            'bytes': size,
        })
    if not chunks or len(chunks) > MAX_CHUNKS:
        raise SystemExit(f'chunk_count_invalid:{len(chunks)}')
    return chunks


def requirements(base: str, target: str, files: list[str]) -> str:
    value = '\n'.join([
        'Audit mode: Dashboard Project AI dispatch-control implementation. Findings are HYPOTHESIS_ONLY until source-verified.',
        f'Exact source range: {SOURCE_REPO}@{base}...{target}',
        f'Full changed paths: {", ".join(files)}',
        'Review concrete correctness/security boundaries, not style:',
        '- Dashboard authentication and same-origin write markers must remain mandatory.',
        '- Project/Workstream/TaskContract identity and exact candidate SHA must fail closed.',
        '- Target-repository read credentials must not gain runner/workflow write authority.',
        '- Runner workflow credentials must not be reused as generic target-repository credentials.',
        '- Paid AI must require explicit human approval; fallback and automatic paid retry remain disabled.',
        '- GitHub workflow_dispatch transport/input bounds must be enforced before provider invocation.',
        '- Request persistence/readback must not equate dispatch acceptance with completed review.',
        '- Semantic results are hypothesis-only and coding results proposal-only; no auto-apply/merge/deploy/completion.',
        '- projectAi.bundle must match the typed Continuity capability and exact execution-plane route.',
        '- Failure handling must preserve evidence without silently widening authority or repeating paid work.',
        '- UI must not expose arbitrary repository/workflow/ref/model/provider dispatch fields.',
        '- Regression tests must cover denial paths, approval, exact route, no fallback/retry, and capability registration.',
        '- During this implementation an accidental create/delete temporary-file pair reached Dashboard main; only report a code defect if the supplied implementation enables or relies on unsafe direct-main mutation.',
        'Do not report speculative missing context or style preferences. Report only concrete defects supported by the supplied diff.',
    ])
    if len(value.encode()) > 12_000:
        raise SystemExit('requirements_too_large')
    return value


def build(repo: Path, base: str, target: str):
    files = [value for value in git(repo, 'diff', '--name-only', base, target, '--').splitlines() if value.strip()]
    if not files or len(files) > 30:
        raise SystemExit(f'changed_file_count_invalid:{len(files)}')
    blocks = []
    for path in files:
        patch = git(repo, 'diff', '--no-ext-diff', '--unified=25', base, target, '--', path)
        if not patch.strip():
            raise SystemExit(f'empty_patch:{path}')
        if 'Binary files ' in patch or 'GIT binary patch' in patch:
            raise SystemExit(f'binary_patch_forbidden:{path}')
        blocks.extend(split_patch(path, patch))
    chunks = chunk_blocks(blocks)
    req = requirements(base, target, files)
    out = Path('audit-inputs')
    out.mkdir(exist_ok=True)
    manifest = {
        'schema': 'dashboard-5plus0-audit-manifest-v0',
        'baseSha': base,
        'targetSha': target,
        'changedPaths': files,
        'chunks': [],
        'semanticReviewersPerChunk': 5,
        'codingReasonersPerChunk': 0,
        'automaticRetry': False,
    }
    for index, chunk in enumerate(chunks, 1):
        payload = {
            'task_id': f'dashboard-5plus0:{target[:12]}:chunk-{index:02d}',
            'language': 'ko',
            'requirements': req + f'\nThis chunk paths: {", ".join(chunk["paths"])}',
            'code': chunk['code'],
        }
        raw = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()
        if len(payload['code'].encode()) > 48_000 or len(payload['requirements'].encode()) > 12_000 or len(raw) > 65_536:
            raise SystemExit(f'chunk_payload_too_large:{index}')
        path = out / f'chunk-{index:02d}.json'
        path.write_bytes(raw)
        manifest['chunks'].append({
            'index': index,
            'input': str(path),
            'paths': chunk['paths'],
            'codeBytes': len(payload['code'].encode()),
            'payloadBytes': len(raw),
        })
    Path('audit-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({
        'changedFiles': len(files),
        'chunks': len(chunks),
        'expectedSemanticCalls': len(chunks) * 5,
        'codingCalls': 0,
    }))


def summarize():
    manifest_path = Path('audit-manifest.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    results = []
    completed = 0
    for path in sorted(glob.glob('audit-results/*.json')):
        data = json.loads(Path(path).read_text())
        completed += int(data.get('completed_reviews', 0))
        results.append({
            'file': path,
            'reviewBudget': data.get('review_budget'),
            'completedReviews': data.get('completed_reviews'),
            'findingCount': data.get('finding_count'),
            'resultState': data.get('result_state'),
            'aggregatedFindings': data.get('aggregated_findings', []),
            'usage': data.get('usage', {}),
        })
    chunks = len(manifest.get('chunks', []))
    expected = chunks * 5
    summary = {
        'schema': 'dashboard-5plus0-audit-summary-v0',
        'authority': 'HYPOTHESIS_ONLY',
        'productionPassFailAuthority': False,
        'baseSha': manifest.get('baseSha'),
        'targetSha': manifest.get('targetSha'),
        'changedPaths': manifest.get('changedPaths', []),
        'chunkCount': chunks,
        'semanticReviewerCallsExpected': expected,
        'semanticReviewerCallsCompleted': completed,
        'codingReasonerCalls': 0,
        'automaticRetry': False,
        'results': results,
        'nextGate': 'deterministic_source_verification_required',
    }
    Path('dashboard-5plus0-audit-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({'chunks': chunks, 'expectedSemanticCalls': expected, 'completedSemanticCalls': completed, 'codingCalls': 0}))
    if chunks and completed != expected:
        raise SystemExit('five_plus_zero_incomplete')


def self_test():
    blocks = [
        ('a.ts', '=== a.ts ===\n' + 'a' * 10_000),
        ('b.ts', '=== b.ts ===\n' + 'b' * 30_000),
        ('c.ts', '=== c.ts ===\n' + 'c' * 5_000),
    ]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 2
    assert chunks[0]['paths'] == ['a.ts']
    assert chunks[1]['paths'] == ['b.ts', 'c.ts']
    assert all(chunk['bytes'] <= MAX_CODE_BYTES for chunk in chunks)
    print('dashboard 5+0 chunker self-test: PASS')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--repo', default='dashboard')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.build:
        build(Path(args.repo), os.environ['BASE_SHA'], os.environ['TARGET_SHA'])
        return
    if args.summarize:
        summarize()
        return
    raise SystemExit('mode_required')


if __name__ == '__main__':
    main()
