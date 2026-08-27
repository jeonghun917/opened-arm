#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = 'project-ai-execution-pools-v0'
TASK_TYPES = {'CODING_WORKER', 'AI_REVIEW'}
TASK_STATES = {'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'BLOCKED'}
PROJECT_MODES = {'ENFORCED', 'OBSERVE_ONLY'}
ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$')
MAX_TASKS = 500
MAX_RECEIPTS = 2000


class PoolError(ValueError):
    pass


def _id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value.strip()):
        raise PoolError(f'{field} is invalid')
    return value.strip()


def _n(value: Any, field: str, none: bool = False) -> int | None:
    if value is None and none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 9007199254740991:
        raise PoolError(f'{field} must be a non-negative safe integer')
    return value


def _p(value: Any, field: str, none: bool = False) -> int | None:
    parsed = _n(value, field, none)
    if parsed is not None and parsed < 1:
        raise PoolError(f'{field} must be at least 1')
    return parsed


def _b(value: Any, field: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise PoolError(f'{field} must be boolean')
    return value


def _iso(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PoolError(f'{field} must be a timestamp')
    try:
        parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError as exc:
        raise PoolError(f'{field} must be ISO-8601-like') from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def empty_state() -> dict[str, Any]:
    return {'schema': SCHEMA, 'projects': {}, 'tasks': [], 'receipts': []}


def project_policy(project_id: str, raw: Any) -> dict[str, Any]:
    project_id = _id(project_id, 'projectId')
    if not isinstance(raw, dict):
        raise PoolError('project policy must be an object')
    mode = raw.get('mode', 'OBSERVE_ONLY')
    if mode not in PROJECT_MODES:
        raise PoolError('project mode is invalid')
    return {
        'projectId': project_id,
        'mode': mode,
        'slotCount': _p(raw.get('slotCount'), 'slotCount', True),
        'budgetUsdMicros': _n(raw.get('budgetUsdMicros'), 'budgetUsdMicros', True),
    }


def task(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PoolError('task must be an object')
    task_type = raw.get('taskType')
    task_state = raw.get('state', 'QUEUED')
    if task_type not in TASK_TYPES:
        raise PoolError('taskType is invalid')
    if task_state not in TASK_STATES:
        raise PoolError('task state is invalid')
    automatic_retry = _b(raw.get('automaticRetry'), 'automaticRetry')
    if automatic_retry:
        raise PoolError('automaticRetry must remain false')
    paid = _b(raw.get('paid'), 'paid')
    explicit_approval = _b(raw.get('explicitApproval'), 'explicitApproval')
    estimate = _n(raw.get('estimatedCostUsdMicros'), 'estimatedCostUsdMicros', True)
    if not paid and estimate not in (None, 0):
        raise PoolError('free task cannot reserve paid-model cost')
    candidate_ref = None if raw.get('candidateRef') is None else _id(raw.get('candidateRef'), 'candidateRef')
    authority_ref = None if raw.get('authorityRef') is None else _id(raw.get('authorityRef'), 'authorityRef')
    if authority_ref is None:
        raise PoolError('authorityRef is required for every execution task')
    if task_type == 'AI_REVIEW' and candidate_ref is None:
        raise PoolError('AI_REVIEW requires an exact candidateRef')
    return {
        'taskId': _id(raw.get('taskId'), 'taskId'),
        'projectId': _id(raw.get('projectId'), 'projectId'),
        'workstreamId': _id(raw.get('workstreamId'), 'workstreamId'),
        'runId': _id(raw.get('runId'), 'runId'),
        'taskType': task_type,
        'state': task_state,
        'createdAt': _iso(raw.get('createdAt'), 'createdAt'),
        'paid': paid,
        'explicitApproval': explicit_approval,
        'automaticRetry': False,
        'estimatedCostUsdMicros': estimate,
        'candidateRef': candidate_ref,
        'authorityRef': authority_ref,
        'blockedReason': None if raw.get('blockedReason') is None else str(raw.get('blockedReason'))[:500],
        'startedAt': None if raw.get('startedAt') is None else _iso(raw.get('startedAt'), 'startedAt'),
        'completedAt': None if raw.get('completedAt') is None else _iso(raw.get('completedAt'), 'completedAt'),
    }


def receipt(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PoolError('receipt must be an object')
    task_type = raw.get('taskType')
    result = raw.get('result')
    usage_authority = raw.get('usageAuthority', 'PROVIDER_REPORTED')
    if task_type not in TASK_TYPES:
        raise PoolError('receipt taskType is invalid')
    if result not in {'SUCCESS', 'FAILURE'}:
        raise PoolError('receipt result is invalid')
    if usage_authority not in {'PROVIDER_REPORTED', 'DETERMINISTIC', 'ESTIMATE_ONLY'}:
        raise PoolError('usageAuthority is invalid')
    model_id = raw.get('modelId')
    model_id = None if model_id is None else _id(model_id, 'modelId')
    model_calls = _n(raw.get('modelCalls', 0), 'modelCalls')
    if model_calls and model_id is None:
        raise PoolError('modelId is required when modelCalls > 0')
    candidate_ref = None if raw.get('candidateRef') is None else _id(raw.get('candidateRef'), 'candidateRef')
    authority_ref = None if raw.get('authorityRef') is None else _id(raw.get('authorityRef'), 'authorityRef')
    if authority_ref is None:
        raise PoolError('receipt authorityRef is required')
    if task_type == 'AI_REVIEW' and candidate_ref is None:
        raise PoolError('AI_REVIEW receipt requires exact candidateRef')
    authoritative_cost = _n(raw.get('authoritativeCostUsdMicros'), 'authoritativeCostUsdMicros', True)
    if usage_authority == 'ESTIMATE_ONLY' and authoritative_cost is not None:
        raise PoolError('ESTIMATE_ONLY receipt cannot claim authoritative cost')
    return {
        'receiptId': _id(raw.get('receiptId'), 'receiptId'),
        'taskId': _id(raw.get('taskId'), 'taskId'),
        'projectId': _id(raw.get('projectId'), 'projectId'),
        'workstreamId': _id(raw.get('workstreamId'), 'workstreamId'),
        'runId': _id(raw.get('runId'), 'runId'),
        'taskType': task_type,
        'candidateRef': candidate_ref,
        'authorityRef': authority_ref,
        'provider': _id(raw.get('provider'), 'provider'),
        'modelId': model_id,
        'result': result,
        'inputTokens': _n(raw.get('inputTokens', 0), 'inputTokens'),
        'outputTokens': _n(raw.get('outputTokens', 0), 'outputTokens'),
        'modelCalls': model_calls,
        'estimatedCostUsdMicros': _n(raw.get('estimatedCostUsdMicros'), 'estimatedCostUsdMicros', True),
        'authoritativeCostUsdMicros': authoritative_cost,
        'retryCount': _n(raw.get('retryCount', 0), 'retryCount'),
        'usageAuthority': usage_authority,
        'sourceRef': _id(raw.get('sourceRef'), 'sourceRef'),
        'startedAt': _iso(raw.get('startedAt'), 'startedAt'),
        'completedAt': _iso(raw.get('completedAt'), 'completedAt'),
        'resultAuthority': 'HYPOTHESIS_ONLY' if task_type == 'AI_REVIEW' else 'CANDIDATE_ONLY',
        'mayMerge': False,
        'mayWidenAuthority': False,
    }


def state(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get('schema') != SCHEMA:
        raise PoolError(f'state schema must be {SCHEMA}')
    projects = raw.get('projects', {})
    tasks = raw.get('tasks', [])
    receipts = raw.get('receipts', [])
    if not isinstance(projects, dict):
        raise PoolError('projects must be an object')
    if not isinstance(tasks, list) or len(tasks) > MAX_TASKS:
        raise PoolError('tasks must be a bounded array')
    if not isinstance(receipts, list) or len(receipts) > MAX_RECEIPTS:
        raise PoolError('receipts must be a bounded array')
    out = {
        'schema': SCHEMA,
        'projects': {key: project_policy(key, value) for key, value in projects.items()},
        'tasks': [task(value) for value in tasks],
        'receipts': [receipt(value) for value in receipts],
    }
    task_ids = [item['taskId'] for item in out['tasks']]
    receipt_ids = [item['receiptId'] for item in out['receipts']]
    if len(task_ids) != len(set(task_ids)):
        raise PoolError('duplicate taskId')
    if len(receipt_ids) != len(set(receipt_ids)):
        raise PoolError('duplicate receiptId')
    return out


def _completed_budget_charge(item: dict[str, Any]) -> int:
    estimate = item['estimatedCostUsdMicros'] or 0
    authoritative = item['authoritativeCostUsdMicros'] or 0
    return max(estimate, authoritative)


def budget_usage(raw_state: dict[str, Any], project_id: str) -> dict[str, int]:
    completed = sum(
        _completed_budget_charge(item)
        for item in raw_state['receipts']
        if item['projectId'] == project_id
    )
    running = sum(
        (item['estimatedCostUsdMicros'] or 0)
        for item in raw_state['tasks']
        if item['projectId'] == project_id and item['state'] == 'RUNNING' and item['paid']
    )
    return {
        'completedChargeUsdMicros': completed,
        'runningReservationUsdMicros': running,
        'committedUsdMicros': completed + running,
    }


def start_decision(raw_state: dict[str, Any], item: dict[str, Any]) -> tuple[str, str]:
    policy = raw_state['projects'].get(item['projectId'])
    if policy is None:
        return 'CONFIG_REQUIRED', 'Project execution policy is not configured.'
    if policy['mode'] == 'OBSERVE_ONLY':
        return 'CONFIG_REQUIRED', 'Project is observation-only.'
    if policy['slotCount'] is None:
        return 'CONFIG_REQUIRED', 'Project slotCount is not configured.'
    running = sum(
        1
        for existing in raw_state['tasks']
        if existing['projectId'] == item['projectId'] and existing['state'] == 'RUNNING'
    )
    if running >= policy['slotCount']:
        return 'WAIT', 'All project-local execution slots are occupied.'
    if item['paid']:
        if not item['explicitApproval']:
            return 'DENY', 'Paid model execution requires explicit approval.'
        if policy['budgetUsdMicros'] is None:
            return 'CONFIG_REQUIRED', 'Paid execution budget is not configured.'
        if item['estimatedCostUsdMicros'] is None:
            return 'CONFIG_REQUIRED', 'Paid execution requires a pre-run cost estimate.'
        projected = budget_usage(raw_state, item['projectId'])['committedUsdMicros'] + item['estimatedCostUsdMicros']
        if projected > policy['budgetUsdMicros']:
            return 'DENY', 'Projected project spend exceeds configured budget.'
    return 'ALLOW', 'Project-local slot and resource policy allow execution.'


def allocate_once(raw: Any, now: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    out = state(copy.deepcopy(raw))
    allocated_at = _iso(now or _now(), 'now')
    started: list[str] = []
    decisions: list[dict[str, str]] = []
    project_ids = sorted({item['projectId'] for item in out['tasks'] if item['state'] == 'QUEUED'})
    for project_id in project_ids:
        queue = sorted(
            (item for item in out['tasks'] if item['projectId'] == project_id and item['state'] == 'QUEUED'),
            key=lambda item: (item['createdAt'], item['taskId']),
        )
        for item in queue:
            decision, reason = start_decision(out, item)
            if decision == 'ALLOW':
                item['state'] = 'RUNNING'
                item['startedAt'] = allocated_at
                item['blockedReason'] = None
                started.append(item['taskId'])
                continue
            decisions.append({'taskId': item['taskId'], 'decision': decision, 'reason': reason})
            if decision in {'DENY', 'CONFIG_REQUIRED'}:
                item['state'] = 'BLOCKED'
                item['blockedReason'] = reason
                item['completedAt'] = allocated_at
                continue
            break
    return out, {'startedTaskIds': started, 'decisions': decisions}


def record_receipt(raw: Any, raw_receipt: Any) -> dict[str, Any]:
    out = state(copy.deepcopy(raw))
    item = receipt(raw_receipt)
    if any(existing['receiptId'] == item['receiptId'] for existing in out['receipts']):
        raise PoolError('duplicate receiptId')
    execution = next((existing for existing in out['tasks'] if existing['taskId'] == item['taskId']), None)
    if execution is None:
        raise PoolError('receipt task does not exist')
    if execution['state'] != 'RUNNING':
        raise PoolError('receipt may only close a RUNNING task')
    for field in ('projectId', 'workstreamId', 'runId', 'taskType', 'candidateRef', 'authorityRef'):
        if execution[field] != item[field]:
            raise PoolError(f'receipt {field} does not match task authority identity')
    if execution['startedAt'] and item['startedAt'] != execution['startedAt']:
        raise PoolError('receipt startedAt does not match task')
    if item['completedAt'] < item['startedAt']:
        raise PoolError('receipt completedAt precedes startedAt')
    if execution['paid'] and item['modelCalls'] < 1:
        raise PoolError('paid model task must report at least one model call')
    if execution['paid'] and item['estimatedCostUsdMicros'] != execution['estimatedCostUsdMicros']:
        raise PoolError('receipt estimate must equal the pre-run budget reservation')
    if not execution['paid'] and (
        (item['authoritativeCostUsdMicros'] or 0) > 0 or (item['estimatedCostUsdMicros'] or 0) > 0
    ):
        raise PoolError('free task cannot report paid model cost')
    out['receipts'].append(item)
    execution['state'] = 'COMPLETED' if item['result'] == 'SUCCESS' else 'FAILED'
    execution['completedAt'] = item['completedAt']
    return out


def ledger_summary(raw: Any) -> dict[str, Any]:
    parsed = state(raw)
    totals = {
        'inputTokens': 0,
        'outputTokens': 0,
        'modelCalls': 0,
        'estimatedCostUsdMicros': 0,
        'authoritativeCostUsdMicros': 0,
        'successCount': 0,
        'failureCount': 0,
        'retryCount': 0,
    }
    by_task_type = {key: {'executions': 0, 'modelCalls': 0} for key in sorted(TASK_TYPES)}
    for item in parsed['receipts']:
        for field in ('inputTokens', 'outputTokens', 'modelCalls', 'retryCount'):
            totals[field] += item[field]
        totals['estimatedCostUsdMicros'] += item['estimatedCostUsdMicros'] or 0
        totals['authoritativeCostUsdMicros'] += item['authoritativeCostUsdMicros'] or 0
        totals['successCount' if item['result'] == 'SUCCESS' else 'failureCount'] += 1
        by_task_type[item['taskType']]['executions'] += 1
        by_task_type[item['taskType']]['modelCalls'] += item['modelCalls']
    return {'schema': SCHEMA, 'commonLedger': True, 'totals': totals, 'byTaskType': by_task_type}


def _task(
    task_id: str,
    project_id: str,
    workstream_id: str,
    run_id: str,
    task_type: str,
    created_at: str,
    paid: bool = False,
    approval: bool = False,
    estimate: int | None = None,
    candidate: str | None = None,
) -> dict[str, Any]:
    return {
        'taskId': task_id,
        'projectId': project_id,
        'workstreamId': workstream_id,
        'runId': run_id,
        'taskType': task_type,
        'state': 'QUEUED',
        'createdAt': created_at,
        'paid': paid,
        'explicitApproval': approval,
        'automaticRetry': False,
        'estimatedCostUsdMicros': estimate,
        'candidateRef': candidate,
        'authorityRef': 'authority:test',
    }


def self_test() -> dict[str, Any]:
    raw = empty_state()
    raw['projects'] = {
        'a': {'mode': 'ENFORCED', 'slotCount': 1, 'budgetUsdMicros': 1000},
        'b': {'mode': 'ENFORCED', 'slotCount': 1, 'budgetUsdMicros': 1000},
        'c': {'mode': 'ENFORCED', 'slotCount': 2, 'budgetUsdMicros': 1000},
        'u': {'mode': 'ENFORCED', 'slotCount': 1, 'budgetUsdMicros': None},
    }
    zero = '2026-08-27T00:00:00Z'
    raw['tasks'] = [
        _task('a-1', 'a', 'ws-a', 'run-a1', 'CODING_WORKER', zero),
        _task('a-2', 'a', 'ws-a', 'run-a2', 'AI_REVIEW', '2026-08-27T00:00:01Z', candidate='sha:a2'),
        _task('b-1', 'b', 'ws-b', 'run-b1', 'CODING_WORKER', zero),
        _task('c-r1', 'c', 'ws-c', 'run-c1', 'AI_REVIEW', zero, True, True, 10, 'sha:abc'),
        _task('c-r2', 'c', 'ws-c', 'run-c2', 'AI_REVIEW', '2026-08-27T00:00:01Z', True, True, 10, 'sha:abc'),
        _task('u-1', 'u', 'ws-u', 'run-u1', 'AI_REVIEW', zero, True, True, 10, 'sha:u1'),
    ]
    allocated, report = allocate_once(raw, '2026-08-27T00:01:00Z')
    started = set(report['startedTaskIds'])
    assert {'a-1', 'b-1', 'c-r1', 'c-r2'} <= started and 'a-2' not in started
    assert next(item for item in allocated['tasks'] if item['taskId'] == 'u-1')['state'] == 'BLOCKED'
    assert next(item for item in allocated['tasks'] if item['taskId'] == 'a-2')['state'] == 'QUEUED'

    for field in ('paid', 'explicitApproval', 'automaticRetry'):
        invalid_boolean = _task('strict-bool', 'a', 'ws-a', 'run-strict', 'CODING_WORKER', zero)
        invalid_boolean[field] = 'false'
        try:
            task(invalid_boolean)
        except PoolError:
            pass
        else:
            raise AssertionError(f'{field} string boolean must fail closed')

    allocated = record_receipt(allocated, {
        'receiptId': 'ra', 'taskId': 'a-1', 'projectId': 'a', 'workstreamId': 'ws-a', 'runId': 'run-a1',
        'taskType': 'CODING_WORKER', 'candidateRef': None, 'authorityRef': 'authority:test', 'provider': 'LOCAL_TEST',
        'modelId': None, 'result': 'SUCCESS', 'modelCalls': 0, 'inputTokens': 0, 'outputTokens': 0,
        'estimatedCostUsdMicros': 0, 'authoritativeCostUsdMicros': 0, 'retryCount': 0,
        'usageAuthority': 'DETERMINISTIC', 'sourceRef': 'test:a1', 'startedAt': '2026-08-27T00:01:00Z',
        'completedAt': '2026-08-27T00:02:00Z',
    })
    allocated = record_receipt(allocated, {
        'receiptId': 'rc', 'taskId': 'c-r1', 'projectId': 'c', 'workstreamId': 'ws-c', 'runId': 'run-c1',
        'taskType': 'AI_REVIEW', 'candidateRef': 'sha:abc', 'authorityRef': 'authority:test',
        'provider': 'AWS_BEDROCK_QWEN', 'modelId': 'qwen.qwen3-coder-30b-a3b-v1:0', 'result': 'SUCCESS',
        'modelCalls': 1, 'inputTokens': 100, 'outputTokens': 20, 'estimatedCostUsdMicros': 10,
        'authoritativeCostUsdMicros': None, 'retryCount': 0, 'usageAuthority': 'PROVIDER_REPORTED',
        'sourceRef': 'test:c1', 'startedAt': '2026-08-27T00:01:00Z', 'completedAt': '2026-08-27T00:02:00Z',
    })
    review_receipt = next(item for item in allocated['receipts'] if item['receiptId'] == 'rc')
    assert review_receipt['resultAuthority'] == 'HYPOTHESIS_ONLY' and review_receipt['mayMerge'] is False
    summary = ledger_summary(allocated)
    assert summary['commonLedger']
    assert summary['byTaskType']['CODING_WORKER']['executions'] == 1
    assert summary['byTaskType']['AI_REVIEW']['executions'] == 1
    assert summary['totals']['modelCalls'] == 1

    second, second_report = allocate_once(allocated, '2026-08-27T00:03:00Z')
    assert 'a-2' in second_report['startedTaskIds']
    base_bad = {
        'receiptId': 'bad', 'taskId': 'a-2', 'projectId': 'a', 'workstreamId': 'ws-a', 'runId': 'run-a2',
        'taskType': 'AI_REVIEW', 'candidateRef': 'sha:a2', 'authorityRef': 'authority:test', 'provider': 'LOCAL_TEST',
        'modelId': None, 'result': 'SUCCESS', 'modelCalls': 0, 'inputTokens': 0, 'outputTokens': 0,
        'estimatedCostUsdMicros': 0, 'authoritativeCostUsdMicros': 0, 'retryCount': 0,
        'usageAuthority': 'DETERMINISTIC', 'sourceRef': 'test:bad', 'startedAt': '2026-08-27T00:03:00Z',
        'completedAt': '2026-08-27T00:04:00Z',
    }
    for field, wrong in (('projectId', 'b'), ('candidateRef', 'sha:other'), ('authorityRef', 'authority:other')):
        bad = dict(base_bad)
        bad[field] = wrong
        try:
            record_receipt(second, bad)
        except PoolError:
            pass
        else:
            raise AssertionError(f'{field} mismatch must fail closed')

    conservative = empty_state()
    conservative['projects'] = {'p': {'mode': 'ENFORCED', 'slotCount': 1, 'budgetUsdMicros': 100}}
    conservative['tasks'] = [_task('p-1', 'p', 'ws-p', 'run-p1', 'AI_REVIEW', zero, True, True, 100, 'sha:p1')]
    conservative, _ = allocate_once(conservative, '2026-08-27T00:01:00Z')
    conservative = record_receipt(conservative, {
        'receiptId': 'rp1', 'taskId': 'p-1', 'projectId': 'p', 'workstreamId': 'ws-p', 'runId': 'run-p1',
        'taskType': 'AI_REVIEW', 'candidateRef': 'sha:p1', 'authorityRef': 'authority:test', 'provider': 'PROVIDER_TEST',
        'modelId': 'model:test', 'result': 'SUCCESS', 'modelCalls': 1, 'inputTokens': 1, 'outputTokens': 1,
        'estimatedCostUsdMicros': 100, 'authoritativeCostUsdMicros': 1, 'retryCount': 0,
        'usageAuthority': 'PROVIDER_REPORTED', 'sourceRef': 'provider:test', 'startedAt': '2026-08-27T00:01:00Z',
        'completedAt': '2026-08-27T00:02:00Z',
    })
    assert budget_usage(conservative, 'p')['committedUsdMicros'] == 100
    conservative['tasks'].append(_task('p-2', 'p', 'ws-p', 'run-p2', 'AI_REVIEW', '2026-08-27T00:03:00Z', True, True, 1, 'sha:p2'))
    conservative, conservative_report = allocate_once(conservative, '2026-08-27T00:04:00Z')
    assert 'p-2' not in conservative_report['startedTaskIds']
    assert next(item for item in conservative['tasks'] if item['taskId'] == 'p-2')['state'] == 'BLOCKED'

    return {
        'schema': SCHEMA,
        'status': 'PASS',
        'checks': {
            'projectQueuesIndependent': True,
            'missingBudgetPaidExecutionBlocked': True,
            'parallelAiReviewSupported': True,
            'commonAllocatorAndLedger': True,
            'reviewHasNoMergeAuthority': True,
            'receiptIdentityBound': True,
            'authorityRefBound': True,
            'candidateRefBound': True,
            'reservationCannotBeLoweredByReceipt': True,
            'automaticRetryForbidden': True,
            'strictBooleanInputs': True,
        },
    }


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text())


def _dump(path: str, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('self-test')
    allocate = sub.add_parser('allocate')
    allocate.add_argument('state')
    allocate.add_argument('output')
    record = sub.add_parser('record')
    record.add_argument('state')
    record.add_argument('receipt')
    record.add_argument('output')
    summary = sub.add_parser('summary')
    summary.add_argument('state')
    args = parser.parse_args(argv)
    if args.cmd == 'self-test':
        print(json.dumps(self_test(), sort_keys=True))
        return 0
    if args.cmd == 'allocate':
        next_state, report = allocate_once(_load(args.state))
        _dump(args.output, next_state)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.cmd == 'record':
        next_state = record_receipt(_load(args.state), _load(args.receipt))
        _dump(args.output, next_state)
        print(json.dumps(ledger_summary(next_state), sort_keys=True))
        return 0
    if args.cmd == 'summary':
        print(json.dumps(ledger_summary(_load(args.state)), sort_keys=True))
        return 0
    raise AssertionError('unreachable')


if __name__ == '__main__':
    raise SystemExit(main())
