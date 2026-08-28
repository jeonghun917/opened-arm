#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path

CODE_SUFFIXES = {'.ts', '.tsx', '.py', '.mjs', '.js', '.css', '.sql', '.yml', '.yaml'}
MAX_CODE_BYTES = 40_000
MAX_PART_BYTES = 32_000
MAX_TOTAL_CHUNKS = 120

BUNDLES = {
    1: 'app-shell-session-security-navigation',
    2: 'dashboard-state-activity-action-inbox',
    3: 'module-runtime-registry-settings',
    4: 'github-integration-console-webhook-oidc',
    5: 'google-drive-workspace-docs',
    6: 'kaggle',
    7: 'lightning',
    8: 'incident-lifecycle-diagnostics-presentation',
    9: 'local-incident-ai-auto-summary',
    10: 'korean-rnd-drive-sync-status-board',
    11: 'meta-control-project-ai-3plus1',
    12: 'continuity-resume-router-workstreams',
    13: 'execution-plane-tool-authority-orchestration',
    14: 'coding-worker-ai-provider-closure-runner',
    15: 'failure-analysis-review-verification-evidence',
    16: 'scheduler-cron-source-sync',
    17: 'speech-ml',
    18: 'drive-store',
    19: 'report-retrieval-resource-economy-context-efficiency',
    20: 'ci-e2e-test-harness-operational-sql',
}


def is_code_file(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() not in CODE_SUFFIXES:
        return False
    if any(part in {'node_modules', 'dist', 'build'} for part in p.parts):
        return False
    return True


def bundle_for(path: str) -> int | None:
    # Bundle 20 is intentionally the verification/CI surface from the agreed taxonomy.
    if path.startswith('scripts/') or path.startswith('.github/workflows/') or (path.startswith('ops/') and path.endswith('.sql')):
        return 20
    if path in {'vite.config.ts', 'src/vite-env.d.ts'}:
        return 20

    if path == 'dashboard_security.py' or path in {'src/App.tsx', 'src/main.tsx', 'src/index.css'}:
        return 1
    if path.startswith('src/components/layout/') or path.startswith('src/components/ui/') or path.startswith('src/components/modals/'):
        return 1
    if path.startswith('src/lib/') or path.startswith('src/types/'):
        return 1
    if path == 'src/app/DashboardSessionGate.tsx':
        return 1
    if path in {'api/_lib/controlSession.ts', 'api/_lib/dashboardSecurity.ts', 'api/_lib/http.ts', 'api/health.ts'}:
        return 1

    if path in {'api/activity.ts', 'api/_lib/dashboardState.ts', 'api/_lib/inboxStore.ts', 'api/_lib/store.ts'}:
        return 2
    if path in {'src/components/activity/ActivityStream.tsx', 'src/components/home/ActionInbox.tsx', 'src/components/home/ModuleSummaryCard.tsx'}:
        return 2
    if path in {'src/core/activityClient.ts', 'src/core/contracts.ts', 'src/core/dashboardStateClient.ts', 'src/core/inboxClient.ts', 'src/core/localStore.ts'}:
        return 2

    if path in {'src/app/pluginCatalog.tsx', 'src/app/runtimeActivation.ts', 'src/components/integrations/IntegrationsView.tsx', 'src/core/registry.ts'}:
        return 3
    if path.startswith('src/components/modules/') or path.startswith('src/components/settings/'):
        return 3

    if path.startswith('api/github/') or path.startswith('src/integrations/github/') or path.startswith('src/modules/github-console/'):
        return 4
    if path == 'src/components/integrations/GitHubConnectionPanel.tsx':
        return 4
    if path.startswith('api/_lib/github') and not path.startswith('api/_lib/githubIncident') and path != 'api/_lib/githubVerification.ts':
        return 4

    if path.startswith('api/google/') or path.startswith('src/integrations/google-drive/') or path.startswith('src/modules/google-workspace/'):
        return 5
    if path == 'src/components/integrations/DriveConnectionPanel.tsx':
        return 5
    if path in {'api/_lib/googleDocsAppend.ts', 'api/_lib/googleDrive.ts', 'api/_lib/googleWorkspace.ts'}:
        return 5

    if path == 'kaggle_bridge.py' or path.startswith('src/integrations/kaggle/') or path == 'src/components/integrations/KaggleConnectionPanel.tsx':
        return 6

    if path in {'lightning_bridge.py', 'api/lightning.py', 'src/components/integrations/LightningConnectionPanel.tsx'} or path.startswith('src/integrations/lightning/'):
        return 7

    if path.startswith('api/_lib/incident') or path.startswith('api/_lib/githubIncident'):
        return 8
    if path in {'src/components/activity/IncidentFacetFilter.tsx', 'src/components/activity/IncidentView.tsx', 'src/core/incidentClient.ts', 'src/core/incidentFilters.ts', 'src/core/incidentSummaryClient.ts', 'shared/githubIncidentSources.ts', 'shared/incidentPlainLanguage.ts'}:
        return 8

    if path in {'src/components/activity/IncidentAutoSummary.tsx', 'src/components/activity/LocalIncidentAIWorkbench.tsx', 'src/core/localIncidentAI.ts', 'src/core/localIncidentAutoSummary.ts'}:
        return 9

    if path.startswith('src/components/korean/') or path.startswith('src/modules/korean/') or path.startswith('api/modules/korean/'):
        return 10
    if path in {'api/_lib/koreanDriveSync.ts', 'api/_lib/koreanStatusBoard.ts', 'api/_lib/xlsxText.ts'}:
        return 10

    if path in {'api/_lib/metaControl.ts', 'api/_lib/projectAiControl.ts', 'api/_lib/projectAiDispatch.ts'} or path.startswith('src/modules/meta-control/'):
        return 11

    if path in {'api/_lib/continuityStore.ts', 'api/_lib/projectMetaResumeStore.ts', 'api/_lib/projectResumeStore.ts', 'api/_lib/projectResumeWorkstreamStore.ts',
                'shared/continuity.ts', 'shared/continuityResume.ts', 'shared/projectMetaResume.ts', 'shared/projectResumeRouter.ts', 'shared/projectResumeWorkstream.ts'}:
        return 12

    if path in {'api/_lib/executionPlaneStore.ts', 'api/_lib/metaRequestStore.ts', 'api/_lib/projectAuthorityStore.ts', 'api/_lib/projectDispatchStore.ts',
                'shared/authorityPromotion.ts', 'shared/executionPlane.ts', 'shared/executionToolCatalog.ts', 'shared/metaRequest.ts', 'shared/orchestration.ts', 'shared/projectDispatch.ts'}:
        return 13

    if path in {'api/_lib/aiProvider.ts', 'api/_lib/closureRunner.ts', 'api/_lib/codingWorkerGitHub.ts', 'shared/codingWorker.ts'}:
        return 14

    if path in {'api/_lib/githubVerification.ts', 'shared/evidenceSlots.ts', 'shared/failureAnalysis.ts', 'shared/review.ts', 'shared/verificationCollector.ts'}:
        return 15

    if path in {'api/_lib/schedulerStore.ts', 'api/_lib/sourceChanges.ts'} or path.startswith('api/cron/'):
        return 16

    if path.startswith('src/modules/speech-ml/'):
        return 17

    if path == 'shared/driveStore.ts':
        return 18

    if path in {'shared/contextEfficiency.ts', 'shared/report.ts', 'shared/resourceEconomy.ts', 'shared/retrieval.ts'}:
        return 19

    return None


def split_text(path: str, text: str) -> list[str]:
    header = f'=== FILE: {path} ===\n'
    raw = (header + text).encode()
    if len(raw) <= MAX_PART_BYTES:
        return [header + text]
    lines = text.splitlines(True)
    parts, current = [], []
    size = len(header.encode()) + 80
    for line in lines:
        b = len(line.encode())
        if b > MAX_PART_BYTES - 500:
            raise SystemExit(f'single_line_too_large:{path}')
        if current and size + b > MAX_PART_BYTES:
            parts.append(''.join(current))
            current = []
            size = len(header.encode()) + 80
        current.append(line)
        size += b
    if current:
        parts.append(''.join(current))
    return [f'=== FILE: {path} [part {i}/{len(parts)}] ===\n{part}' for i, part in enumerate(parts, 1)]


def chunk_blocks(blocks: list[tuple[str, str]]) -> list[dict]:
    chunks, texts, paths, size = [], [], [], 0
    for path, text in blocks:
        b = len(text.encode()) + (2 if texts else 0)
        if b > MAX_CODE_BYTES:
            raise SystemExit(f'block_too_large:{path}:{b}')
        if texts and size + b > MAX_CODE_BYTES:
            chunks.append({'code': '\n\n'.join(texts), 'paths': list(dict.fromkeys(paths)), 'codeBytes': size})
            texts, paths, size = [], [], 0
        texts.append(text)
        paths.append(path)
        size += b
    if texts:
        chunks.append({'code': '\n\n'.join(texts), 'paths': list(dict.fromkeys(paths)), 'codeBytes': size})
    return chunks


def make_requirements(bundle_id: int, target_sha: str, bundle_paths: list[str], chunk_paths: list[str], chunk_index: int, chunk_count: int) -> str:
    return '\n'.join([
        'Audit mode: full Dashboard current-source 20-bundle semantic review.',
        'All findings are HYPOTHESIS_ONLY until deterministic source verification.',
        f'Exact repository: jeonghun917/dashboard-control-center@{target_sha}',
        f'Bundle {bundle_id:02d}: {BUNDLES[bundle_id]}',
        f'Bundle file count: {len(bundle_paths)}',
        f'Bundle chunk: {chunk_index}/{chunk_count}',
        f'This chunk paths: {", ".join(chunk_paths)}',
        'Review concrete defects only; do not report style preferences or speculative missing context.',
        'Check correctness, state/data flow, authorization/trust boundaries, async/races, failure handling, persistence/idempotency, API contracts, resource limits and operational failure modes.',
        'Do not infer PASS/FAIL authority. Repetition or majority is not production truth.',
        'Do not propose auto-apply, merge or deploy. Coding reasoner count is zero.',
        'Treat other chunks as separately reviewed evidence; report only defects supported by this supplied chunk.',
    ])


def plan(dashboard_root: Path, target_sha: str, out_dir: Path) -> dict:
    code_paths = []
    for p in sorted(dashboard_root.rglob('*')):
        if p.is_file():
            rel = p.relative_to(dashboard_root).as_posix()
            if is_code_file(rel):
                code_paths.append(rel)

    assignments: dict[int, list[str]] = {i: [] for i in BUNDLES}
    unassigned = []
    for path in code_paths:
        b = bundle_for(path)
        if b is None:
            unassigned.append(path)
        else:
            assignments[b].append(path)
    if unassigned:
        raise SystemExit('unassigned_code_files:' + ','.join(unassigned))
    empty = [i for i, paths in assignments.items() if not paths]
    if empty:
        raise SystemExit('empty_bundles:' + ','.join(map(str, empty)))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': 'dashboard-full-20bundle-audit-manifest-v0',
        'repository': 'jeonghun917/dashboard-control-center',
        'targetSha': target_sha,
        'codeFileCount': len(code_paths),
        'bundleCount': 20,
        'semanticReviewersPerChunk': 5,
        'reviewerIds': ['A', 'B', 'C', 'D', 'E'],
        'codingReasoners': 0,
        'automaticRetry': False,
        'extensions': dict(sorted(Counter(Path(x).suffix.lower() for x in code_paths).items())),
        'bundles': [],
    }
    total_chunks = 0
    total_bytes = 0
    for bundle_id in range(1, 21):
        paths = assignments[bundle_id]
        blocks = []
        bundle_bytes = 0
        for path in paths:
            text = (dashboard_root / path).read_text(encoding='utf-8', errors='replace')
            bundle_bytes += len(text.encode())
            for part in split_text(path, text):
                blocks.append((path, part))
        chunks = chunk_blocks(blocks)
        total_chunks += len(chunks)
        total_bytes += bundle_bytes
        bundle_dir = out_dir / f'bundle-{bundle_id:02d}'
        bundle_dir.mkdir(exist_ok=True)
        chunk_meta = []
        for idx, chunk in enumerate(chunks, 1):
            payload = {
                'task_id': f'dashboard-full-20bundle:{target_sha[:12]}:bundle-{bundle_id:02d}:chunk-{idx:02d}',
                'language': 'ko',
                'requirements': make_requirements(bundle_id, target_sha, paths, chunk['paths'], idx, len(chunks)),
                'code': chunk['code'],
            }
            raw = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()
            if len(payload['code'].encode()) > MAX_CODE_BYTES or len(payload['requirements'].encode()) > 12_000 or len(raw) > 65_536:
                raise SystemExit(f'chunk_payload_too_large:{bundle_id}:{idx}:{len(raw)}')
            input_path = bundle_dir / f'chunk-{idx:02d}.json'
            input_path.write_bytes(raw)
            chunk_meta.append({
                'index': idx,
                'input': input_path.as_posix(),
                'paths': chunk['paths'],
                'codeBytes': chunk['codeBytes'],
                'payloadBytes': len(raw),
            })
        manifest['bundles'].append({
            'id': bundle_id,
            'name': BUNDLES[bundle_id],
            'fileCount': len(paths),
            'sourceBytes': bundle_bytes,
            'paths': paths,
            'chunkCount': len(chunks),
            'chunks': chunk_meta,
        })
    if total_chunks > MAX_TOTAL_CHUNKS:
        raise SystemExit(f'total_chunk_count_out_of_bounds:{total_chunks}')
    manifest['sourceBytes'] = total_bytes
    manifest['totalChunks'] = total_chunks
    manifest['expectedSemanticReviewerCalls'] = total_chunks * 5
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps({
        'codeFileCount': manifest['codeFileCount'],
        'bundleCount': 20,
        'totalChunks': total_chunks,
        'expectedSemanticReviewerCalls': total_chunks * 5,
        'sourceBytes': total_bytes,
        'bundleFileCounts': {f'{b["id"]:02d}': b['fileCount'] for b in manifest['bundles']},
        'bundleChunkCounts': {f'{b["id"]:02d}': b['chunkCount'] for b in manifest['bundles']},
    }, ensure_ascii=False, separators=(',', ':')))
    return manifest


def summarize(plan_dir: Path, result_dir: Path, output_path: Path) -> dict:
    manifest = json.loads((plan_dir / 'manifest.json').read_text())
    bundles_out = []
    completed_calls = 0
    raw_findings = 0
    errors = []
    for bundle in manifest['bundles']:
        bundle_reviews = []
        bundle_raw = 0
        bundle_completed = 0
        for chunk in bundle['chunks']:
            stem = f'bundle-{bundle["id"]:02d}-chunk-{chunk["index"]:02d}'
            abc_path = result_dir / f'{stem}-abc.json'
            de_path = result_dir / f'{stem}-de.json'
            for expected_ids, p in [(['A', 'B', 'C'], abc_path), (['D', 'E'], de_path)]:
                if not p.exists():
                    errors.append(f'missing:{p}')
                    continue
                data = json.loads(p.read_text())
                reviews = data.get('reviews', [])
                ids = [r.get('reviewer_id') for r in reviews if r.get('status') == 'completed']
                if ids != expected_ids:
                    errors.append(f'incomplete:{p}:{ids}')
                bundle_completed += len(ids)
                bundle_raw += sum(len(r.get('findings', [])) for r in reviews if r.get('status') == 'completed')
                bundle_reviews.extend({'chunk': chunk['index'], **r} for r in reviews)
        completed_calls += bundle_completed
        raw_findings += bundle_raw
        bundles_out.append({
            'id': bundle['id'],
            'name': bundle['name'],
            'fileCount': bundle['fileCount'],
            'chunkCount': bundle['chunkCount'],
            'expectedReviewerCalls': bundle['chunkCount'] * 5,
            'completedReviewerCalls': bundle_completed,
            'rawFindingCount': bundle_raw,
            'paths': bundle['paths'],
            'reviews': bundle_reviews,
        })
    expected = manifest['expectedSemanticReviewerCalls']
    summary = {
        'schema': 'dashboard-full-20bundle-5plus0-summary-v0',
        'authority': 'HYPOTHESIS_ONLY',
        'productionPassFailAuthority': False,
        'repository': manifest['repository'],
        'targetSha': manifest['targetSha'],
        'codeFileCount': manifest['codeFileCount'],
        'bundleCount': 20,
        'totalChunks': manifest['totalChunks'],
        'semanticReviewerCallsExpected': expected,
        'semanticReviewerCallsCompleted': completed_calls,
        'codingReasonerCalls': 0,
        'automaticRetry': False,
        'rawFindingCount': raw_findings,
        'errors': errors,
        'bundles': bundles_out,
        'nextGate': 'deterministic_source_verification_required',
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps({
        'expectedReviewerCalls': expected,
        'completedReviewerCalls': completed_calls,
        'rawFindingCount': raw_findings,
        'errors': len(errors),
    }, ensure_ascii=False, separators=(',', ':')))
    if errors or completed_calls != expected:
        raise SystemExit('dashboard_full_20bundle_5plus0_incomplete')
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--dashboard-root', required=True)
    p.add_argument('--target-sha', required=True)
    p.add_argument('--out-dir', required=True)
    s = sub.add_parser('summarize')
    s.add_argument('--plan-dir', required=True)
    s.add_argument('--result-dir', required=True)
    s.add_argument('--output', required=True)
    args = ap.parse_args()
    if args.cmd == 'plan':
        if not re.fullmatch(r'[0-9a-f]{40}', args.target_sha):
            raise SystemExit('target_sha_invalid')
        plan(Path(args.dashboard_root), args.target_sha, Path(args.out_dir))
    else:
        summarize(Path(args.plan_dir), Path(args.result_dir), Path(args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
