#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

MAX_CODE_BYTES = 40_000
MAX_PART_BYTES = 32_000
MAX_REQUIREMENTS_BYTES = 12_000
MAX_PAYLOAD_BYTES = 65_536
MAX_TOTAL_CHUNKS_HARD = 120
ALLOWED_TARGETS = {
    'pharos-orbis': 'jeonghun917/pharos-orbis',
    'vela-development': 'jeonghun917/Ars-Mentis',
}
EXPECTED_REVIEWERS = ['A', 'B', 'C', 'D', 'E']


def fail(message: str):
    raise SystemExit(message)


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        fail(f'config_invalid_json:{exc}')
    if not isinstance(config, dict):
        fail('config_root_must_be_object')
    validate_config(config)
    return config


def clean_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 300:
        fail(f'{field}_invalid')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or value.startswith('/'):
        fail(f'{field}_unsafe')
    return path.as_posix().rstrip('/')


def validate_config(config: dict) -> None:
    if config.get('schema') != 'product-full-bundle-5plus0-target-v0':
        fail('target_schema_invalid')
    project_id = config.get('projectId')
    repository = config.get('repository')
    if ALLOWED_TARGETS.get(project_id) != repository:
        fail('target_project_repository_not_allowed')
    sha = config.get('targetSha')
    if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha):
        fail('target_sha_invalid')
    approved = config.get('approvedBy')
    if not isinstance(approved, str) or not approved.startswith('user-directive:'):
        fail('human_approval_missing')

    plan = config.get('reviewPlan')
    if not isinstance(plan, dict):
        fail('review_plan_missing')
    if plan.get('semanticReviewersPerChunk') != 5:
        fail('review_count_must_be_five')
    if plan.get('reviewerIds') != EXPECTED_REVIEWERS:
        fail('reviewer_ids_invalid')
    if plan.get('codingReasoners') != 0:
        fail('coding_reasoners_must_be_zero')
    if plan.get('automaticRetry') is not False:
        fail('automatic_retry_must_be_false')
    max_chunks = plan.get('maxTotalChunks')
    if not isinstance(max_chunks, int) or max_chunks < 1 or max_chunks > MAX_TOTAL_CHUNKS_HARD:
        fail('max_total_chunks_invalid')

    roots = config.get('sourceRoots')
    if not isinstance(roots, list) or not roots:
        fail('source_roots_invalid')
    normalized_roots = [clean_relative_path(item, 'source_root') for item in roots]
    if len(set(normalized_roots)) != len(normalized_roots):
        fail('source_roots_duplicate')

    extensions = config.get('codeExtensions')
    if not isinstance(extensions, list) or not extensions:
        fail('code_extensions_invalid')
    for ext in extensions:
        if not isinstance(ext, str) or not re.fullmatch(r'\.[a-z0-9]+', ext):
            fail('code_extension_invalid')

    global_requirements = config.get('globalRequirements', [])
    if not isinstance(global_requirements, list) or not all(isinstance(x, str) and x.strip() for x in global_requirements):
        fail('global_requirements_invalid')

    bundles = config.get('bundles')
    if not isinstance(bundles, list) or not bundles:
        fail('bundles_invalid')
    ids = set()
    all_paths = set()
    for index, bundle in enumerate(bundles, 1):
        if not isinstance(bundle, dict):
            fail(f'bundle_{index}_invalid')
        bundle_id = bundle.get('id')
        if not isinstance(bundle_id, str) or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,99}', bundle_id):
            fail(f'bundle_{index}_id_invalid')
        if bundle_id in ids:
            fail(f'bundle_id_duplicate:{bundle_id}')
        ids.add(bundle_id)
        name = bundle.get('name')
        if not isinstance(name, str) or not name.strip() or len(name) > 180:
            fail(f'bundle_{bundle_id}_name_invalid')
        paths = bundle.get('paths')
        if not isinstance(paths, list) or not paths:
            fail(f'bundle_{bundle_id}_paths_invalid')
        normalized = []
        for raw in paths:
            path = clean_relative_path(raw, f'bundle_{bundle_id}_path')
            if path in all_paths:
                fail(f'code_file_multiply_assigned:{path}')
            all_paths.add(path)
            normalized.append(path)
        if len(set(normalized)) != len(normalized):
            fail(f'bundle_{bundle_id}_path_duplicate')
        req = bundle.get('requirements', [])
        if not isinstance(req, list) or not req or not all(isinstance(x, str) and x.strip() for x in req):
            fail(f'bundle_{bundle_id}_requirements_invalid')


def scan_code_files(repo_root: Path, roots: list[str], extensions: set[str]) -> list[str]:
    files = []
    for root in roots:
        root_path = repo_root / root
        if not root_path.exists() or not root_path.is_dir():
            fail(f'source_root_missing:{root}')
        for path in sorted(root_path.rglob('*')):
            if path.is_file() and path.suffix.lower() in extensions:
                files.append(path.relative_to(repo_root).as_posix())
    return sorted(set(files))


def split_text(path: str, text: str) -> list[str]:
    header = f'=== FILE: {path} ===\n'
    if len((header + text).encode()) <= MAX_PART_BYTES:
        return [header + text]
    lines = text.splitlines(True)
    parts: list[str] = []
    current: list[str] = []
    size = len(header.encode()) + 96
    for line in lines:
        line_bytes = len(line.encode())
        if line_bytes > MAX_PART_BYTES - 600:
            fail(f'single_line_too_large:{path}:{line_bytes}')
        if current and size + line_bytes > MAX_PART_BYTES:
            parts.append(''.join(current))
            current = []
            size = len(header.encode()) + 96
        current.append(line)
        size += line_bytes
    if current:
        parts.append(''.join(current))
    return [
        f'=== FILE: {path} [part {index}/{len(parts)}] ===\n{part}'
        for index, part in enumerate(parts, 1)
    ]


def chunk_blocks(blocks: list[tuple[str, str]]) -> list[dict]:
    chunks: list[dict] = []
    texts: list[str] = []
    paths: list[str] = []
    size = 0
    for path, block in blocks:
        block_bytes = len(block.encode()) + (2 if texts else 0)
        if block_bytes > MAX_CODE_BYTES:
            fail(f'block_too_large:{path}:{block_bytes}')
        if texts and size + block_bytes > MAX_CODE_BYTES:
            chunks.append({'code': '\n\n'.join(texts), 'paths': list(dict.fromkeys(paths)), 'codeBytes': size})
            texts, paths, size = [], [], 0
            block_bytes = len(block.encode())
        texts.append(block)
        paths.append(path)
        size += block_bytes
    if texts:
        chunks.append({'code': '\n\n'.join(texts), 'paths': list(dict.fromkeys(paths)), 'codeBytes': size})
    if not chunks:
        fail('empty_bundle_chunks')
    return chunks


def make_requirements(config: dict, bundle: dict, chunk: dict, chunk_index: int, chunk_count: int) -> str:
    lines = [
        'Audit mode: exact-current-source product functional-bundle 5+0 semantic review.',
        'All model findings are HYPOTHESIS_ONLY until source and design verification.',
        f"Exact source: {config['repository']}@{config['targetSha']}",
        f"Project: {config['projectId']}",
        f"Functional bundle: {bundle['id']} — {bundle['name']}",
        f'Bundle chunk: {chunk_index}/{chunk_count}',
        f"This chunk paths: {', '.join(chunk['paths'])}",
        '',
        'Design intent and audit rules:',
        *[f'- {item}' for item in config.get('globalRequirements', [])],
        *[f'- {item}' for item in bundle.get('requirements', [])],
        '',
        'Review concrete behavior against the supplied design intent, not coding style.',
        'Do not treat a deliberate guardrail, conservative rejection, append-only rule, exact binding, or fail-closed path as a defect merely because it blocks an operation.',
        'Do report an implementation that mutates state/authority/data outside the stated design, or that has the intended shape but fails to implement the intended behavior correctly.',
        'Do not claim production PASS/FAIL, do not propose automatic mutation, and do not infer authority from historical/reference documentation.',
        'Do not quote or reproduce private source text in findings; identify the relevant file/line and paraphrase the behavior.',
        'Use an empty findings array when no concrete defect is supported by this chunk.',
        '',
        'After the run, each raw finding will be source-verified into exactly one human classification:',
        '1) UNINTENDED_MUTATION — the code mutates state/data/authority in a way the design does not intend.',
        '2) INTENDED_CORRECT — the alleged issue is actually intended behavior/guardrail and works as designed.',
        '3) INTENDED_SHAPE_BROKEN_BEHAVIOR — the structure is consistent with the intended design and is not an unintended mutation, but runtime behavior/edge handling is wrong or incomplete.',
    ]
    value = '\n'.join(lines)
    if len(value.encode()) > MAX_REQUIREMENTS_BYTES:
        fail(f'requirements_too_large:{bundle["id"]}:{chunk_index}')
    return value


def plan(config_path: Path, repo_root: Path, out_dir: Path) -> dict:
    config = load_config(config_path)
    roots = [clean_relative_path(item, 'source_root') for item in config['sourceRoots']]
    extensions = set(config['codeExtensions'])
    scanned = scan_code_files(repo_root, roots, extensions)
    assigned = [path for bundle in config['bundles'] for path in bundle['paths']]
    scanned_set = set(scanned)
    assigned_set = set(assigned)
    missing = sorted(scanned_set - assigned_set)
    extra = sorted(assigned_set - scanned_set)
    if missing:
        fail('unassigned_code_files:' + ','.join(missing))
    if extra:
        fail('assigned_files_not_in_scanned_scope:' + ','.join(extra))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': 'product-full-bundle-5plus0-manifest-v0',
        'projectId': config['projectId'],
        'repository': config['repository'],
        'targetSha': config['targetSha'],
        'authority': 'HYPOTHESIS_ONLY',
        'productionPassFailAuthority': False,
        'semanticReviewersPerChunk': 5,
        'reviewerIds': EXPECTED_REVIEWERS,
        'codingReasoners': 0,
        'automaticRetry': False,
        'classificationPolicy': [
            'UNINTENDED_MUTATION',
            'INTENDED_CORRECT',
            'INTENDED_SHAPE_BROKEN_BEHAVIOR',
        ],
        'codeFileCount': len(scanned),
        'bundleCount': len(config['bundles']),
        'bundles': [],
    }
    total_chunks = 0
    source_bytes = 0
    for bundle_index, bundle in enumerate(config['bundles'], 1):
        blocks: list[tuple[str, str]] = []
        bundle_bytes = 0
        for path in bundle['paths']:
            text = (repo_root / path).read_text(encoding='utf-8', errors='replace')
            bundle_bytes += len(text.encode())
            for part in split_text(path, text):
                blocks.append((path, part))
        chunks = chunk_blocks(blocks)
        total_chunks += len(chunks)
        source_bytes += bundle_bytes
        bundle_dir = out_dir / f'bundle-{bundle_index:02d}-{bundle["id"]}'
        bundle_dir.mkdir(exist_ok=True)
        chunk_meta = []
        for chunk_index, chunk in enumerate(chunks, 1):
            payload = {
                'task_id': f"product-5plus0:{config['projectId']}:{config['targetSha'][:12]}:{bundle['id']}:chunk-{chunk_index:02d}",
                'language': 'ko',
                'requirements': make_requirements(config, bundle, chunk, chunk_index, len(chunks)),
                'code': chunk['code'],
            }
            raw = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()
            if len(payload['code'].encode()) > MAX_CODE_BYTES:
                fail(f'code_payload_too_large:{bundle["id"]}:{chunk_index}')
            if len(raw) > MAX_PAYLOAD_BYTES:
                fail(f'payload_too_large:{bundle["id"]}:{chunk_index}:{len(raw)}')
            input_path = bundle_dir / f'chunk-{chunk_index:02d}.json'
            input_path.write_bytes(raw)
            chunk_meta.append({
                'index': chunk_index,
                'input': input_path.as_posix(),
                'paths': chunk['paths'],
                'codeBytes': chunk['codeBytes'],
                'payloadBytes': len(raw),
            })
        manifest['bundles'].append({
            'index': bundle_index,
            'id': bundle['id'],
            'name': bundle['name'],
            'fileCount': len(bundle['paths']),
            'sourceBytes': bundle_bytes,
            'paths': bundle['paths'],
            'chunkCount': len(chunks),
            'chunks': chunk_meta,
        })

    max_chunks = config['reviewPlan']['maxTotalChunks']
    if total_chunks > max_chunks:
        fail(f'total_chunk_count_out_of_bounds:{total_chunks}:{max_chunks}')
    manifest['sourceBytes'] = source_bytes
    manifest['totalChunks'] = total_chunks
    manifest['expectedSemanticReviewerCalls'] = total_chunks * 5
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({
        'projectId': manifest['projectId'],
        'targetSha': manifest['targetSha'],
        'codeFileCount': manifest['codeFileCount'],
        'bundleCount': manifest['bundleCount'],
        'totalChunks': total_chunks,
        'expectedSemanticReviewerCalls': total_chunks * 5,
        'codingReasoners': 0,
        'automaticRetry': False,
    }, ensure_ascii=False, separators=(',', ':')))
    return manifest


def summarize(plan_dir: Path, result_dir: Path, output_path: Path) -> dict:
    manifest = json.loads((plan_dir / 'manifest.json').read_text(encoding='utf-8'))
    completed_calls = 0
    raw_findings = 0
    bundles_out = []
    errors = []
    for bundle in manifest['bundles']:
        bundle_reviews = []
        bundle_completed = 0
        bundle_raw = 0
        for chunk in bundle['chunks']:
            stem = f"bundle-{bundle['index']:02d}-{bundle['id']}-chunk-{chunk['index']:02d}"
            for expected_ids, suffix in [(['A', 'B', 'C'], 'abc'), (['D', 'E'], 'de')]:
                path = result_dir / f'{stem}-{suffix}.json'
                if not path.exists():
                    errors.append(f'missing_result:{path.name}')
                    continue
                try:
                    data = json.loads(path.read_text(encoding='utf-8'))
                except Exception as exc:
                    errors.append(f'invalid_result:{path.name}:{exc}')
                    continue
                reviews = data.get('reviews', [])
                ids = [item.get('reviewer_id') for item in reviews if isinstance(item, dict)]
                if data.get('authority') != 'HYPOTHESIS_ONLY' or data.get('automatic_retry') is not False:
                    errors.append(f'authority_contract_failed:{path.name}')
                if data.get('completed_reviews') != len(expected_ids) or ids != expected_ids:
                    errors.append(f'reviewer_contract_failed:{path.name}')
                completed = int(data.get('completed_reviews', 0) or 0)
                findings = int(data.get('finding_count', 0) or 0)
                completed_calls += completed
                bundle_completed += completed
                raw_findings += findings
                bundle_raw += findings
                bundle_reviews.append({
                    'chunk': chunk['index'],
                    'group': suffix.upper(),
                    'completedReviews': completed,
                    'findingCount': findings,
                    'resultState': data.get('result_state'),
                    'aggregatedFindings': data.get('aggregated_findings', []),
                    'reviews': reviews,
                    'usage': data.get('usage', {}),
                })
        bundles_out.append({
            'index': bundle['index'],
            'id': bundle['id'],
            'name': bundle['name'],
            'completedReviewerCalls': bundle_completed,
            'rawFindingCount': bundle_raw,
            'results': bundle_reviews,
        })

    expected = int(manifest['expectedSemanticReviewerCalls'])
    summary = {
        'schema': 'product-full-bundle-5plus0-summary-v0',
        'projectId': manifest['projectId'],
        'repository': manifest['repository'],
        'targetSha': manifest['targetSha'],
        'authority': 'HYPOTHESIS_ONLY',
        'productionPassFailAuthority': False,
        'bundleCount': manifest['bundleCount'],
        'codeFileCount': manifest['codeFileCount'],
        'totalChunks': manifest['totalChunks'],
        'semanticReviewerCallsExpected': expected,
        'semanticReviewerCallsCompleted': completed_calls,
        'codingReasonerCalls': 0,
        'automaticRetry': False,
        'rawFindingCount': raw_findings,
        'classificationPolicy': manifest['classificationPolicy'],
        'classificationStatus': 'SOURCE_VERIFICATION_REQUIRED',
        'errors': errors,
        'bundles': bundles_out,
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({
        'projectId': manifest['projectId'],
        'expected': expected,
        'completed': completed_calls,
        'rawFindings': raw_findings,
        'errors': len(errors),
    }, ensure_ascii=False, separators=(',', ':')))
    if errors or completed_calls != expected:
        fail('five_plus_zero_incomplete')
    return summary


def self_test() -> None:
    blocks = [
        ('a.py', '=== FILE: a.py ===\n' + 'a' * 10_000),
        ('b.py', '=== FILE: b.py ===\n' + 'b' * 28_000),
        ('c.py', '=== FILE: c.py ===\n' + 'c' * 5_000),
    ]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 2
    assert chunks[0]['paths'] == ['a.py', 'b.py']
    assert chunks[1]['paths'] == ['c.py']
    assert all(chunk['codeBytes'] <= MAX_CODE_BYTES for chunk in chunks)
    print('product functional-bundle 5+0 planner self-test: PASS')


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate-config')
    validate.add_argument('--config', required=True)
    plan_parser = sub.add_parser('plan')
    plan_parser.add_argument('--config', required=True)
    plan_parser.add_argument('--repo-root', required=True)
    plan_parser.add_argument('--out-dir', required=True)
    summary_parser = sub.add_parser('summarize')
    summary_parser.add_argument('--plan-dir', required=True)
    summary_parser.add_argument('--result-dir', required=True)
    summary_parser.add_argument('--output', required=True)
    sub.add_parser('self-test')
    args = parser.parse_args()

    if args.command == 'validate-config':
        config = load_config(Path(args.config))
        print(json.dumps({'projectId': config['projectId'], 'repository': config['repository'], 'targetSha': config['targetSha'], 'bundles': len(config['bundles'])}, separators=(',', ':')))
    elif args.command == 'plan':
        plan(Path(args.config), Path(args.repo_root), Path(args.out_dir))
    elif args.command == 'summarize':
        summarize(Path(args.plan_dir), Path(args.result_dir), Path(args.output))
    elif args.command == 'self-test':
        self_test()


if __name__ == '__main__':
    main()
