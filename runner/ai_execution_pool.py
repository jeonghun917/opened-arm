#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = 'project-ai-execution-pools-v0'
TASK_TYPES = {'CODING_WORKER', 'AI_REVIEW'}
TASK_STATES = {'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'BLOCKED'}
PROJECT_MODES = {'ENFORCED', 'OBSERVE_ONLY'}

CODING_WORKER_VERSION = 0
CONTRACT_EVIDENCE_PREFIX = 'coding-worker-contract:v0:sha256:'
CODING_OPERATIONS = {'create_file', 'update_file', 'delete_file'}
DEFAULT_MAX_MUTATIONS = 20
DEFAULT_MAX_FILE_BYTES = 128 * 1024
DEFAULT_MAX_TOTAL_BYTES = 512 * 1024
MAX_RULES = 100
MAX_MUTATIONS = 50
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 1024 * 1024

EXECUTION_PLANE_REGISTRY_VERSION = 0
EXECUTION_PLANE_KINDS = {'github_actions', 'vercel', 'neon', 'aws_bedrock', 'google_docs', 'other'}
EXECUTION_PLANE_STATUSES = {'ENABLED', 'MANUAL_ONLY', 'DISABLED', 'BLOCKED'}
EXECUTION_PLANE_SELECTIONS = {'EXACT', 'ORDERED'}
EXECUTION_APPROVALS = {'NONE', 'MANUAL_REQUIRED'}
EXECUTION_COST_CLASSES = {'FREE', 'PAID'}

ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$')
REPOSITORY_RE = re.compile(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
SHA_RE = re.compile(r'^[a-f0-9]{40}$', re.I)
BRANCH_RE = re.compile(r'^(?!/)(?!.*(?:^|/)\.\.(?:/|$))(?!.*//)(?!.*\.\.)(?!.*[~^:?*\[\]\\\s])[A-Za-z0-9._/-]{1,200}$')
PATH_RE = re.compile(r'^(?!/)(?!.*(?:^|/)\.\.(?:/|$))(?!.*//)[A-Za-z0-9._@+(), /-]{1,500}$')
PLANE_ID_RE = re.compile(r'^[a-z0-9][a-z0-9._-]{0,119}$')

MAX_TASKS = 500
MAX_RECEIPTS = 2000


class PoolError(ValueError):
    pass


def _id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value.strip()):
        raise PoolError(f'{field} is invalid')
    return value.strip()


def _text(value: Any, field: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise PoolError(f'{field} must be a string')
    out = ' '.join(value.replace('\x00', ' ').split())
    if not out or len(out) > max_length:
        raise PoolError(f'{field} is outside the allowed length')
    return out


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


def _bounded(value: Any, field: str, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    parsed = _n(value, field)
    if parsed < minimum or parsed > maximum:
        raise PoolError(f'{field} is outside the allowed range')
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


def _repository(value: Any, field: str = 'repository') -> str:
    out = _text(value, field, 200).lower()
    if not REPOSITORY_RE.fullmatch(out):
        raise PoolError(f'{field} is invalid')
    return out


def _branch(value: Any, field: str = 'branch') -> str:
    out = _text(value, field, 200)
    if (
        not BRANCH_RE.fullmatch(out)
        or out.endswith('.')
        or out.endswith('/')
        or out.endswith('.lock')
    ):
        raise PoolError(f'{field} is invalid')
    return out


def _sha(value: Any, field: str) -> str:
    out = _text(value, field, 40).lower()
    if not SHA_RE.fullmatch(out):
        raise PoolError(f'{field} is invalid')
    return out


def _path(value: Any, field: str = 'path') -> str:
    if not isinstance(value, str):
        raise PoolError(f'{field} must be a string')
    out = value.strip()
    if (
        not PATH_RE.fullmatch(out)
        or '\\' in out
        or any(part in {'', '.', '..'} for part in out.split('/'))
    ):
        raise PoolError(f'{field} is invalid')
    return out


def _path_rule(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise PoolError(f'{field} must be a string')
    raw = value.strip()
    prefix = raw.endswith('/**')
    base = raw[:-3] if prefix else raw
    normalized = _path(base, field)
    if '*' in raw and not prefix:
        raise PoolError(f'{field} is invalid')
    return f'{normalized}/**' if prefix else normalized


def _path_rules(value: Any, field: str, require_non_empty: bool) -> list[str]:
    if not isinstance(value, list):
        raise PoolError(f'{field} must be an array')
    if len(value) > MAX_RULES or (require_non_empty and not value):
        raise PoolError(f'{field} count is invalid')
    return sorted(set(_path_rule(item, f'{field}[{index}]') for index, item in enumerate(value)))


def _matches_rule(path: str, rule: str) -> bool:
    if rule.endswith('/**'):
        return path.startswith(rule[:-2])
    return path == rule


def _same_path_set(left: list[str], right: list[str]) -> bool:
    return sorted(set(left)) == sorted(set(right))


def _string_list(value: Any, field: str, max_length: int = 1000) -> list[str]:
    if not isinstance(value, list):
        raise PoolError(f'{field} must be an array')
    return list(dict.fromkeys(_text(item, f'{field}[{index}]', max_length) for index, item in enumerate(value)))


def _has_authority_override(value: dict[str, Any]) -> bool:
    return any(key in value for key in ('repository', 'branch', 'baseSha', 'providerMutation', 'target', 'pathPolicy'))


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


def normalize_coding_contract(raw: Any, expected_task_id: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PoolError('codingContract must be an object')
    if raw.get('version', CODING_WORKER_VERSION) != CODING_WORKER_VERSION:
        raise PoolError(f'codingContract.version must be {CODING_WORKER_VERSION}')
    task_id = _id(raw.get('taskId'), 'codingContract.taskId')
    if expected_task_id is not None and task_id != expected_task_id:
        raise PoolError('codingContract.taskId does not match taskId')
    target = raw.get('target')
    if not isinstance(target, dict):
        raise PoolError('codingContract.target must be an object')
    policy = raw.get('pathPolicy')
    if not isinstance(policy, dict):
        raise PoolError('codingContract.pathPolicy must be an object')
    if not isinstance(policy.get('allowDelete'), bool):
        raise PoolError('codingContract.pathPolicy.allowDelete must be boolean')
    return {
        'version': CODING_WORKER_VERSION,
        'taskId': task_id,
        'target': {
            'repository': _repository(target.get('repository'), 'codingContract.target.repository'),
            'branch': _branch(target.get('branch'), 'codingContract.target.branch'),
            'baseSha': _sha(target.get('baseSha'), 'codingContract.target.baseSha'),
        },
        'pathPolicy': {
            'allowedPaths': _path_rules(policy.get('allowedPaths'), 'codingContract.pathPolicy.allowedPaths', True),
            'forbiddenPaths': _path_rules(policy.get('forbiddenPaths', []), 'codingContract.pathPolicy.forbiddenPaths', False),
            'allowDelete': policy['allowDelete'],
            'maxMutations': _bounded(
                policy.get('maxMutations'),
                'codingContract.pathPolicy.maxMutations',
                DEFAULT_MAX_MUTATIONS,
                1,
                MAX_MUTATIONS,
            ),
            'maxFileBytes': _bounded(
                policy.get('maxFileBytes'),
                'codingContract.pathPolicy.maxFileBytes',
                DEFAULT_MAX_FILE_BYTES,
                1,
                MAX_FILE_BYTES,
            ),
            'maxTotalBytes': _bounded(
                policy.get('maxTotalBytes'),
                'codingContract.pathPolicy.maxTotalBytes',
                DEFAULT_MAX_TOTAL_BYTES,
                1,
                MAX_TOTAL_BYTES,
            ),
        },
    }


def _canonical_coding_contract(contract: dict[str, Any]) -> str:
    payload = {
        'taskId': contract['taskId'],
        'target': {
            'repository': contract['target']['repository'],
            'branch': contract['target']['branch'],
            'baseSha': contract['target']['baseSha'],
        },
        'pathPolicy': {
            'allowedPaths': contract['pathPolicy']['allowedPaths'],
            'forbiddenPaths': contract['pathPolicy']['forbiddenPaths'],
            'allowDelete': contract['pathPolicy']['allowDelete'],
            'maxMutations': contract['pathPolicy']['maxMutations'],
            'maxFileBytes': contract['pathPolicy']['maxFileBytes'],
            'maxTotalBytes': contract['pathPolicy']['maxTotalBytes'],
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


def coding_worker_contract_evidence_ref(raw: Any) -> str:
    contract = normalize_coding_contract(raw)
    digest = hashlib.sha256(_canonical_coding_contract(contract).encode('utf-8')).hexdigest()
    return f'{CONTRACT_EVIDENCE_PREFIX}{digest}'


def authorize_coding_mutations(contract_raw: Any, raw_mutations: Any) -> list[dict[str, Any]]:
    contract = normalize_coding_contract(contract_raw)
    policy = contract['pathPolicy']
    if not isinstance(raw_mutations, list):
        raise PoolError('codingMutations must be an array')
    if not raw_mutations or len(raw_mutations) > policy['maxMutations']:
        raise PoolError('coding mutation count is invalid')

    ids: set[str] = set()
    paths: set[str] = set()
    total_bytes = 0
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_mutations):
        if not isinstance(raw, dict):
            raise PoolError(f'codingMutations[{index}] must be an object')
        if _has_authority_override(raw):
            raise PoolError('coding mutation authority override is forbidden')
        mutation_id = _id(raw.get('mutationId'), f'codingMutations[{index}].mutationId').lower()
        if mutation_id in ids:
            raise PoolError('duplicate coding mutationId')
        ids.add(mutation_id)
        path = _path(raw.get('path'), f'codingMutations[{index}].path')
        if path in paths:
            raise PoolError('duplicate coding mutation path')
        paths.add(path)
        operation = raw.get('operation')
        if operation not in CODING_OPERATIONS:
            raise PoolError('coding mutation operation is invalid')
        rationale = _text(raw.get('rationale'), f'codingMutations[{index}].rationale', 1000)

        if not any(_matches_rule(path, rule) for rule in policy['allowedPaths']):
            raise PoolError('coding mutation path is not allowed')
        if any(_matches_rule(path, rule) for rule in policy['forbiddenPaths']):
            raise PoolError('coding mutation path is forbidden')
        if operation == 'delete_file' and not policy['allowDelete']:
            raise PoolError('coding delete is forbidden')

        normalized = {
            'mutationId': mutation_id,
            'path': path,
            'operation': operation,
            'rationale': rationale,
        }
        if operation in {'create_file', 'update_file'}:
            content = raw.get('content')
            if not isinstance(content, str):
                raise PoolError('coding mutation content is required')
            size = len(content.encode('utf-8'))
            if size > policy['maxFileBytes']:
                raise PoolError('coding mutation file is too large')
            total_bytes += size
            normalized['content'] = content
        elif 'content' in raw:
            raise PoolError('coding delete content is forbidden')
        out.append(normalized)

    if total_bytes > policy['maxTotalBytes']:
        raise PoolError('coding mutation total content is too large')
    return out


def _coding_verification(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PoolError('codingVerification must be an object')
    verifier = raw.get('verifier')
    if verifier != 'DETERMINISTIC':
        raise PoolError('codingVerification.verifier must be DETERMINISTIC')
    verified = _b(raw.get('verified'), 'codingVerification.verified')
    changed = raw.get('changedPaths')
    observed = raw.get('observedChangedPaths')
    if not isinstance(changed, list) or not isinstance(observed, list):
        raise PoolError('coding verification changed paths must be arrays')
    return {
        'verifier': 'DETERMINISTIC',
        'verified': verified,
        'repository': _repository(raw.get('repository'), 'codingVerification.repository'),
        'branch': _branch(raw.get('branch'), 'codingVerification.branch'),
        'baseSha': _sha(raw.get('baseSha'), 'codingVerification.baseSha'),
        'commitSha': _sha(raw.get('commitSha'), 'codingVerification.commitSha'),
        'changedPaths': list(dict.fromkeys(_path(item, 'codingVerification.changedPaths') for item in changed)),
        'contractEvidenceRef': _text(raw.get('contractEvidenceRef'), 'codingVerification.contractEvidenceRef', 200),
        'observedBaseSha': _sha(raw.get('observedBaseSha'), 'codingVerification.observedBaseSha'),
        'observedBranchHeadSha': _sha(raw.get('observedBranchHeadSha'), 'codingVerification.observedBranchHeadSha'),
        'observedChangedPaths': list(dict.fromkeys(_path(item, 'codingVerification.observedChangedPaths') for item in observed)),
        'evidenceRefs': _string_list(raw.get('evidenceRefs', []), 'codingVerification.evidenceRefs', 2000),
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
    task_id = _id(raw.get('taskId'), 'taskId')
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

    coding_contract = None
    coding_contract_evidence_ref = None
    coding_mutations: list[dict[str, Any]] = []
    if task_type == 'AI_REVIEW':
        if candidate_ref is None:
            raise PoolError('AI_REVIEW requires an exact candidateRef')
        if raw.get('codingContract') is not None or raw.get('codingMutations') not in (None, []):
            raise PoolError('AI_REVIEW must not carry coding authority')
    else:
        if candidate_ref is not None:
            raise PoolError('CODING_WORKER must not use candidateRef as repository authority')
        coding_contract = normalize_coding_contract(raw.get('codingContract'), task_id)
        coding_contract_evidence_ref = coding_worker_contract_evidence_ref(coding_contract)
        supplied_ref = _text(raw.get('codingContractEvidenceRef'), 'codingContractEvidenceRef', 200)
        if supplied_ref != coding_contract_evidence_ref:
            raise PoolError('codingContractEvidenceRef does not match exact coding contract')
        coding_mutations = authorize_coding_mutations(coding_contract, raw.get('codingMutations'))

    return {
        'taskId': task_id,
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
        'codingContract': coding_contract,
        'codingContractEvidenceRef': coding_contract_evidence_ref,
        'codingMutations': coding_mutations,
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
    if task_type == 'CODING_WORKER' and candidate_ref is not None:
        raise PoolError('CODING_WORKER receipt must not use candidateRef as repository authority')
    authoritative_cost = _n(raw.get('authoritativeCostUsdMicros'), 'authoritativeCostUsdMicros', True)
    if usage_authority == 'ESTIMATE_ONLY' and authoritative_cost is not None:
        raise PoolError('ESTIMATE_ONLY receipt cannot claim authoritative cost')
    coding_verification = None
    if task_type == 'CODING_WORKER':
        coding_verification = _coding_verification(raw.get('codingVerification'))
    elif raw.get('codingVerification') is not None:
        raise PoolError('AI_REVIEW receipt must not carry codingVerification')
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
        'codingVerification': coding_verification,
        'resultAuthority': 'HYPOTHESIS_ONLY' if task_type == 'AI_REVIEW' else 'CANDIDATE_ONLY',
        'requiresIndependentReview': task_type == 'CODING_WORKER',
        'mayCloseContinuity': False,
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


def _validate_coding_receipt(execution: dict[str, Any], item: dict[str, Any]) -> None:
    verification = item['codingVerification']
    contract = execution['codingContract']
    if verification is None or contract is None:
        raise PoolError('CODING_WORKER requires exact coding verification')
    target = contract['target']
    if (
        verification['repository'] != target['repository']
        or verification['branch'] != target['branch']
        or verification['baseSha'] != target['baseSha']
    ):
        raise PoolError('coding verification target identity mismatch')
    if verification['contractEvidenceRef'] != execution['codingContractEvidenceRef']:
        raise PoolError('coding verification contract digest mismatch')
    if verification['commitSha'] == target['baseSha']:
        raise PoolError('coding verification commit must advance baseSha')
    expected_paths = [mutation['path'] for mutation in execution['codingMutations']]
    if not _same_path_set(expected_paths, verification['changedPaths']):
        raise PoolError('coding receipt changed paths do not match authorized plan')
    if verification['observedBaseSha'] != target['baseSha']:
        raise PoolError('coding verification observed baseSha mismatch')
    if verification['observedBranchHeadSha'] != verification['commitSha']:
        raise PoolError('coding verification branch head mismatch')
    if not _same_path_set(expected_paths, verification['observedChangedPaths']):
        raise PoolError('coding verification observed changed paths mismatch')
    if verification['verified'] and not verification['evidenceRefs']:
        raise PoolError('verified coding receipt requires independent evidence')
    if item['result'] == 'SUCCESS' and not verification['verified']:
        raise PoolError('successful coding receipt requires deterministic verification')


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
    if execution['taskType'] == 'CODING_WORKER':
        _validate_coding_receipt(execution, item)
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


def _plane_id(value: Any, field: str) -> str:
    out = _text(value, field, 120).lower()
    if not PLANE_ID_RE.fullmatch(out):
        raise PoolError(f'{field} must be a stable lowercase slug')
    return out


def _capability(value: Any, field: str) -> str:
    return _id(value, field)


def normalize_execution_plane_registry(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PoolError('execution plane registry must be an object')
    if raw.get('version') != EXECUTION_PLANE_REGISTRY_VERSION:
        raise PoolError(f'execution plane registry version must be {EXECUTION_PLANE_REGISTRY_VERSION}')
    registered = raw.get('registeredCapabilities')
    planes = raw.get('planes')
    routes = raw.get('routes')
    if not isinstance(registered, list) or not isinstance(planes, list) or not isinstance(routes, list):
        raise PoolError('execution plane registry arrays are required')
    registered_capabilities = list(dict.fromkeys(
        _capability(value, f'registeredCapabilities[{index}]') for index, value in enumerate(registered)
    ))
    registered_set = set(registered_capabilities)

    normalized_planes = []
    seen_planes: set[str] = set()
    for index, item in enumerate(planes):
        if not isinstance(item, dict):
            raise PoolError(f'planes[{index}] must be an object')
        plane = _plane_id(item.get('planeId'), f'planes[{index}].planeId')
        if plane in seen_planes:
            raise PoolError(f'duplicate execution plane {plane}')
        seen_planes.add(plane)
        kind = item.get('kind')
        status = item.get('status')
        if kind not in EXECUTION_PLANE_KINDS:
            raise PoolError(f'planes[{index}].kind is invalid')
        if status not in EXECUTION_PLANE_STATUSES:
            raise PoolError(f'planes[{index}].status is invalid')
        automatic_retry = _b(item.get('automaticRetry'), f'planes[{index}].automaticRetry')
        capabilities = list(dict.fromkeys(
            _capability(value, f'planes[{index}].capabilities[{cap_index}]')
            for cap_index, value in enumerate(item.get('capabilities', []))
        ))
        if any(cap not in registered_set for cap in capabilities):
            raise PoolError(f'planes[{index}] references an unregistered capability')
        normalized_planes.append({
            'planeId': plane,
            'kind': kind,
            'resource': _text(item.get('resource'), f'planes[{index}].resource', 1000),
            'status': status,
            'capabilities': capabilities,
            'reason': None if item.get('reason') is None else _text(item.get('reason'), f'planes[{index}].reason', 1000),
            'automaticRetry': automatic_retry,
            'authorityRefs': _string_list(item.get('authorityRefs', []), f'planes[{index}].authorityRefs'),
            'evidenceRefs': _string_list(item.get('evidenceRefs', []), f'planes[{index}].evidenceRefs'),
            'updatedAt': _iso(item.get('updatedAt'), f'planes[{index}].updatedAt'),
        })

    normalized_routes = []
    seen_routes: set[str] = set()
    for index, item in enumerate(routes):
        if not isinstance(item, dict):
            raise PoolError(f'routes[{index}] must be an object')
        cap = _capability(item.get('capability'), f'routes[{index}].capability')
        if cap not in registered_set:
            raise PoolError(f'routes[{index}] references an unregistered capability')
        if cap in seen_routes:
            raise PoolError(f'duplicate capability route {cap}')
        seen_routes.add(cap)
        selection = item.get('selection')
        approval = item.get('approval')
        cost_class = item.get('costClass')
        if selection not in EXECUTION_PLANE_SELECTIONS:
            raise PoolError(f'routes[{index}].selection is invalid')
        if approval not in EXECUTION_APPROVALS:
            raise PoolError(f'routes[{index}].approval is invalid')
        if cost_class not in EXECUTION_COST_CLASSES:
            raise PoolError(f'routes[{index}].costClass is invalid')
        auto_retry_allowed = _b(item.get('autoRetryAllowed'), f'routes[{index}].autoRetryAllowed')
        required = list(dict.fromkeys(
            _plane_id(value, f'routes[{index}].requiredPlaneIds') for value in item.get('requiredPlaneIds', [])
        ))
        fallback = list(dict.fromkeys(
            _plane_id(value, f'routes[{index}].fallbackPlaneIds') for value in item.get('fallbackPlaneIds', [])
        ))
        if not required:
            raise PoolError('execution route requires at least one required plane')
        if selection == 'EXACT' and len(required) != 1:
            raise PoolError('EXACT execution route requires exactly one required plane')
        for plane in required + fallback:
            if plane not in seen_planes:
                raise PoolError(f'capability route {cap} references unknown plane {plane}')
        for plane in required:
            definition = next(value for value in normalized_planes if value['planeId'] == plane)
            if cap not in definition['capabilities']:
                raise PoolError(f'required plane {plane} does not provide {cap}')
        normalized_routes.append({
            'capability': cap,
            'requiredPlaneIds': required,
            'fallbackPlaneIds': fallback,
            'selection': selection,
            'approval': approval,
            'costClass': cost_class,
            'autoRetryAllowed': auto_retry_allowed,
        })

    return {
        'version': EXECUTION_PLANE_REGISTRY_VERSION,
        'registeredCapabilities': registered_capabilities,
        'planes': normalized_planes,
        'routes': normalized_routes,
        'updatedAt': _iso(raw.get('updatedAt'), 'executionPlane.updatedAt'),
    }


def _execution_policy(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {
            'allowedPlaneIds': [],
            'forbiddenPlaneIds': [],
            'requireManualApproval': False,
            'allowFallback': True,
            'maxAutomaticRetries': 0,
        }
    if not isinstance(raw, dict):
        raise PoolError('execution policy must be an object')
    allowed = raw.get('allowedPlaneIds', [])
    forbidden = raw.get('forbiddenPlaneIds', [])
    if not isinstance(allowed, list) or not isinstance(forbidden, list):
        raise PoolError('execution policy plane lists must be arrays')
    return {
        'allowedPlaneIds': list(dict.fromkeys(_plane_id(item, 'allowedPlaneIds') for item in allowed)),
        'forbiddenPlaneIds': list(dict.fromkeys(_plane_id(item, 'forbiddenPlaneIds') for item in forbidden)),
        'requireManualApproval': _b(raw.get('requireManualApproval'), 'requireManualApproval'),
        'allowFallback': _b(raw.get('allowFallback'), 'allowFallback', True),
        'maxAutomaticRetries': _n(raw.get('maxAutomaticRetries', 0), 'maxAutomaticRetries'),
    }


def _delegated(scope: Any, registry: dict[str, Any], capability: str) -> tuple[bool, str]:
    if not isinstance(scope, dict):
        return False, 'Execution scope is missing.'
    delegated = scope.get('delegatedCapabilities')
    if not isinstance(delegated, list):
        return False, 'Execution scope delegatedCapabilities is missing.'
    if capability not in registry['registeredCapabilities']:
        return False, f'{capability} is not registered.'
    normalized = []
    try:
        normalized = [_capability(value, 'delegatedCapabilities') for value in delegated]
    except PoolError as exc:
        return False, str(exc)
    if capability not in normalized:
        return False, f'{capability} is not delegated.'
    return True, 'Delegated capability is allowed.'


def _route_candidates(route: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    return route['requiredPlaneIds'] + (route['fallbackPlaneIds'] if policy['allowFallback'] else [])


def _plane_allowed(plane: str, policy: dict[str, Any]) -> bool:
    if plane in policy['forbiddenPlaneIds']:
        return False
    if policy['allowedPlaneIds']:
        return plane in policy['allowedPlaneIds']
    return True


def resolve_execution_plane(scope: Any, registry_raw: Any, capability_name: str, policy_raw: Any = None) -> dict[str, Any]:
    registry = normalize_execution_plane_registry(registry_raw)
    capability_name = _capability(capability_name, 'capability')
    delegated, reason = _delegated(scope, registry, capability_name)
    if not delegated:
        return {'allowed': False, 'code': 'DENY_CAPABILITY', 'reason': reason}
    route = next((item for item in registry['routes'] if item['capability'] == capability_name), None)
    if route is None:
        return {'allowed': False, 'code': 'NO_ROUTE', 'reason': f'No execution route is registered for {capability_name}.'}
    policy = _execution_policy(policy_raw)
    for candidate in _route_candidates(route, policy):
        if not _plane_allowed(candidate, policy):
            continue
        plane = next((item for item in registry['planes'] if item['planeId'] == candidate), None)
        if plane is None or capability_name not in plane['capabilities']:
            continue
        if plane['status'] in {'DISABLED', 'BLOCKED'}:
            continue
        return {
            'allowed': True,
            'code': 'ROUTE_RESOLVED',
            'capability': capability_name,
            'planeId': plane['planeId'],
            'approvalRequired': (
                plane['status'] == 'MANUAL_ONLY'
                or route['approval'] == 'MANUAL_REQUIRED'
                or route['costClass'] == 'PAID'
                or policy['requireManualApproval']
            ),
        }
    return {
        'allowed': False,
        'code': 'NO_VALID_PLANE',
        'reason': f'No enabled execution plane is available for {capability_name}.',
    }


def evaluate_execution_preflight(
    scope: Any,
    registry_raw: Any,
    attempt: Any,
    policy_raw: Any = None,
) -> dict[str, Any]:
    registry = normalize_execution_plane_registry(registry_raw)
    if not isinstance(attempt, dict):
        raise PoolError('execution attempt must be an object')
    capability_name = _capability(attempt.get('capability'), 'attempt.capability')
    delegated, reason = _delegated(scope, registry, capability_name)
    if not delegated:
        return {'allowed': False, 'code': 'DENY_CAPABILITY', 'reason': reason}
    route = next((item for item in registry['routes'] if item['capability'] == capability_name), None)
    if route is None:
        return {'allowed': False, 'code': 'NO_ROUTE', 'reason': f'No execution route is registered for {capability_name}.'}
    policy = _execution_policy(policy_raw)
    requested = _plane_id(attempt.get('planeId'), 'attempt.planeId')
    resolution = resolve_execution_plane(scope, registry, capability_name, policy)
    candidates = _route_candidates(route, policy)
    if requested not in candidates or not _plane_allowed(requested, policy):
        return {
            'allowed': False,
            'code': 'DENY_WRONG_PLANE',
            'reason': f'{requested} is not an authorized execution plane for {capability_name}.',
            **({'routeTo': resolution['planeId']} if resolution.get('code') == 'ROUTE_RESOLVED' else {}),
        }
    plane = next((item for item in registry['planes'] if item['planeId'] == requested), None)
    if plane is None or capability_name not in plane['capabilities']:
        return {
            'allowed': False,
            'code': 'DENY_WRONG_PLANE',
            'reason': f'{requested} does not provide {capability_name}.',
            **({'routeTo': resolution['planeId']} if resolution.get('code') == 'ROUTE_RESOLVED' else {}),
        }
    if plane['status'] in {'DISABLED', 'BLOCKED'}:
        return {
            'allowed': False,
            'code': 'DENY_PLANE_DISABLED',
            'reason': plane['reason'] or f'{requested} is {plane["status"].lower()}.',
            **({'routeTo': resolution['planeId']} if resolution.get('code') == 'ROUTE_RESOLVED' else {}),
        }
    if resolution.get('code') != 'ROUTE_RESOLVED':
        return resolution
    if route['selection'] == 'EXACT' and resolution['planeId'] != requested:
        return {
            'allowed': False,
            'code': 'DENY_WRONG_PLANE',
            'reason': f'{capability_name} must run on {resolution["planeId"]}.',
            'routeTo': resolution['planeId'],
        }

    retry_count = _n(attempt.get('automaticRetryCount', 0), 'attempt.automaticRetryCount')
    if retry_count > 0 and (
        not route['autoRetryAllowed']
        or not plane['automaticRetry']
        or retry_count > policy['maxAutomaticRetries']
    ):
        return {
            'allowed': False,
            'code': 'DENY_RETRY',
            'reason': f'Automatic retry {retry_count} is not authorized for {capability_name} on {requested}.',
        }

    manual_approval = _b(attempt.get('manualApprovalGranted'), 'attempt.manualApprovalGranted')
    approval_required = (
        plane['status'] == 'MANUAL_ONLY'
        or route['approval'] == 'MANUAL_REQUIRED'
        or route['costClass'] == 'PAID'
        or policy['requireManualApproval']
    )
    if approval_required and not manual_approval:
        return {
            'allowed': False,
            'code': 'MANUAL_APPROVAL_REQUIRED',
            'reason': f'{capability_name} on {requested} requires explicit manual approval.',
        }
    return {'allowed': True, 'code': 'ALLOW', 'capability': capability_name, 'planeId': requested}


def durable_store_preflight(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {'allowed': False, 'code': 'CONFIG_REQUIRED', 'reason': 'Shared Platform durable store is not configured.'}
    status = raw.get('status')
    if status == 'CONFIG_REQUIRED':
        return {'allowed': False, 'code': 'CONFIG_REQUIRED', 'reason': 'Shared Platform durable store is not configured.'}
    if status != 'ACTIVE':
        return {'allowed': False, 'code': 'DENY_STORE', 'reason': 'Durable store status is invalid.'}
    if raw.get('ownerProjectId') != 'shared-platform':
        return {'allowed': False, 'code': 'DENY_STORE', 'reason': 'Durable store must be owned by shared-platform.'}
    if raw.get('atomicity') not in {'LEASE_CAS', 'SERIALIZABLE'}:
        return {'allowed': False, 'code': 'DENY_STORE', 'reason': 'Durable store must provide atomic lease/CAS or serializable transactions.'}
    backend_ref = raw.get('backendRef')
    if not isinstance(backend_ref, str) or not backend_ref.strip():
        return {'allowed': False, 'code': 'CONFIG_REQUIRED', 'reason': 'Durable store backendRef is missing.'}
    return {'allowed': True, 'code': 'ALLOW', 'backendRef': backend_ref.strip()}


def _coding_contract(task_id: str = 'coding-1') -> dict[str, Any]:
    raw = {
        'version': 0,
        'taskId': task_id,
        'target': {
            'repository': 'jeonghun917/example',
            'branch': 'feat/example',
            'baseSha': '1' * 40,
        },
        'pathPolicy': {
            'allowedPaths': ['src/**', 'README.md'],
            'forbiddenPaths': ['src/secret.txt'],
            'allowDelete': False,
            'maxMutations': 3,
            'maxFileBytes': 1024,
            'maxTotalBytes': 2048,
        },
    }
    return normalize_coding_contract(raw)


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
    out = {
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
    if task_type == 'CODING_WORKER':
        contract = _coding_contract(task_id)
        out['candidateRef'] = None
        out['codingContract'] = contract
        out['codingContractEvidenceRef'] = coding_worker_contract_evidence_ref(contract)
        out['codingMutations'] = [{
            'mutationId': 'm1',
            'path': 'src/main.py',
            'operation': 'update_file',
            'content': 'print("ok")\n',
            'rationale': 'test mutation',
        }]
    return out


def _coding_receipt(execution: dict[str, Any], receipt_id: str = 'ra') -> dict[str, Any]:
    contract = execution['codingContract']
    target = contract['target']
    commit = '2' * 40
    expected_paths = [item['path'] for item in execution['codingMutations']]
    return {
        'receiptId': receipt_id,
        'taskId': execution['taskId'],
        'projectId': execution['projectId'],
        'workstreamId': execution['workstreamId'],
        'runId': execution['runId'],
        'taskType': 'CODING_WORKER',
        'candidateRef': None,
        'authorityRef': execution['authorityRef'],
        'provider': 'GITHUB_CONTENTS_TEST',
        'modelId': None,
        'result': 'SUCCESS',
        'modelCalls': 0,
        'inputTokens': 0,
        'outputTokens': 0,
        'estimatedCostUsdMicros': 0,
        'authoritativeCostUsdMicros': 0,
        'retryCount': 0,
        'usageAuthority': 'DETERMINISTIC',
        'sourceRef': 'GitHubCommit:test',
        'startedAt': execution['startedAt'],
        'completedAt': '2026-08-27T00:02:00Z',
        'codingVerification': {
            'verifier': 'DETERMINISTIC',
            'verified': True,
            'repository': target['repository'],
            'branch': target['branch'],
            'baseSha': target['baseSha'],
            'commitSha': commit,
            'changedPaths': expected_paths,
            'contractEvidenceRef': execution['codingContractEvidenceRef'],
            'observedBaseSha': target['baseSha'],
            'observedBranchHeadSha': commit,
            'observedChangedPaths': expected_paths,
            'evidenceRefs': ['GitHubCommit:test', 'GitHubDiff:test'],
        },
    }


def _execution_registry() -> dict[str, Any]:
    now = '2026-08-27T00:00:00Z'
    return {
        'version': 0,
        'registeredCapabilities': ['shared.free', 'shared.paid'],
        'updatedAt': now,
        'planes': [
            {
                'planeId': 'opened-arm.free',
                'kind': 'github_actions',
                'resource': 'jeonghun917/opened-arm/.github/workflows/free.yml',
                'status': 'ENABLED',
                'capabilities': ['shared.free'],
                'automaticRetry': False,
                'authorityRefs': ['authority:test'],
                'evidenceRefs': [],
                'updatedAt': now,
            },
            {
                'planeId': 'opened-arm.paid',
                'kind': 'aws_bedrock',
                'resource': 'infra/aws/semantic-review/pool.py',
                'status': 'ENABLED',
                'capabilities': ['shared.paid'],
                'automaticRetry': False,
                'authorityRefs': ['authority:test'],
                'evidenceRefs': [],
                'updatedAt': now,
            },
        ],
        'routes': [
            {
                'capability': 'shared.free',
                'requiredPlaneIds': ['opened-arm.free'],
                'fallbackPlaneIds': [],
                'selection': 'EXACT',
                'approval': 'NONE',
                'costClass': 'FREE',
                'autoRetryAllowed': False,
            },
            {
                'capability': 'shared.paid',
                'requiredPlaneIds': ['opened-arm.paid'],
                'fallbackPlaneIds': [],
                'selection': 'EXACT',
                'approval': 'MANUAL_REQUIRED',
                'costClass': 'PAID',
                'autoRetryAllowed': False,
            },
        ],
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

    exact_contract = _coding_contract('digest-test')
    exact_ref = coding_worker_contract_evidence_ref(exact_contract)
    tampered = copy.deepcopy(exact_contract)
    tampered['target']['branch'] = 'feat/other'
    assert coding_worker_contract_evidence_ref(tampered) != exact_ref
    for mutation in (
        {'mutationId': 'bad-path', 'path': 'outside.txt', 'operation': 'update_file', 'content': 'x', 'rationale': 'x'},
        {'mutationId': 'bad-forbidden', 'path': 'src/secret.txt', 'operation': 'update_file', 'content': 'x', 'rationale': 'x'},
        {'mutationId': 'bad-delete', 'path': 'src/main.py', 'operation': 'delete_file', 'rationale': 'x'},
        {'mutationId': 'bad-override', 'path': 'src/main.py', 'operation': 'update_file', 'content': 'x', 'rationale': 'x', 'branch': 'main'},
    ):
        try:
            authorize_coding_mutations(exact_contract, [mutation])
        except PoolError:
            pass
        else:
            raise AssertionError('coding authority widening must fail closed')

    coding_execution = next(item for item in allocated['tasks'] if item['taskId'] == 'a-1')
    allocated = record_receipt(allocated, _coding_receipt(coding_execution))
    coding_receipt = next(item for item in allocated['receipts'] if item['receiptId'] == 'ra')
    assert coding_receipt['resultAuthority'] == 'CANDIDATE_ONLY'
    assert coding_receipt['requiresIndependentReview'] is True
    assert coding_receipt['mayCloseContinuity'] is False and coding_receipt['mayMerge'] is False

    for field, wrong in (
        ('branch', 'feat/other'),
        ('baseSha', '3' * 40),
        ('contractEvidenceRef', f'{CONTRACT_EVIDENCE_PREFIX}' + '0' * 64),
        ('observedBranchHeadSha', '4' * 40),
    ):
        bad_state = allocate_once({
            'schema': SCHEMA,
            'projects': {'x': {'mode': 'ENFORCED', 'slotCount': 1, 'budgetUsdMicros': 100}},
            'tasks': [_task('x-1', 'x', 'ws-x', 'run-x', 'CODING_WORKER', zero)],
            'receipts': [],
        }, '2026-08-27T00:01:00Z')[0]
        execution = bad_state['tasks'][0]
        bad_receipt = _coding_receipt(execution, 'bad-coding')
        bad_receipt['codingVerification'][field] = wrong
        try:
            record_receipt(bad_state, bad_receipt)
        except PoolError:
            pass
        else:
            raise AssertionError(f'coding verification {field} mismatch must fail closed')

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

    registry = _execution_registry()
    scope = {'delegatedCapabilities': ['shared.free', 'shared.paid']}
    free_resolution = resolve_execution_plane(scope, registry, 'shared.free')
    assert free_resolution['allowed'] and free_resolution['planeId'] == 'opened-arm.free'
    paid_denied = evaluate_execution_preflight(scope, registry, {
        'capability': 'shared.paid',
        'planeId': 'opened-arm.paid',
        'manualApprovalGranted': False,
        'automaticRetryCount': 0,
    })
    assert paid_denied['code'] == 'MANUAL_APPROVAL_REQUIRED'
    paid_allowed = evaluate_execution_preflight(scope, registry, {
        'capability': 'shared.paid',
        'planeId': 'opened-arm.paid',
        'manualApprovalGranted': True,
        'automaticRetryCount': 0,
    })
    assert paid_allowed['allowed']
    retry_denied = evaluate_execution_preflight(scope, registry, {
        'capability': 'shared.paid',
        'planeId': 'opened-arm.paid',
        'manualApprovalGranted': True,
        'automaticRetryCount': 1,
    })
    assert retry_denied['code'] == 'DENY_RETRY'
    wrong_plane = evaluate_execution_preflight(scope, registry, {
        'capability': 'shared.free',
        'planeId': 'opened-arm.paid',
        'manualApprovalGranted': False,
        'automaticRetryCount': 0,
    })
    assert wrong_plane['code'] == 'DENY_WRONG_PLANE'
    not_delegated = resolve_execution_plane({'delegatedCapabilities': []}, registry, 'shared.free')
    assert not_delegated['code'] == 'DENY_CAPABILITY'

    assert durable_store_preflight({'status': 'CONFIG_REQUIRED'})['code'] == 'CONFIG_REQUIRED'
    assert durable_store_preflight({
        'status': 'ACTIVE',
        'ownerProjectId': 'dashboard-control-center',
        'atomicity': 'SERIALIZABLE',
        'backendRef': 'dashboard:neon',
    })['code'] == 'DENY_STORE'
    assert durable_store_preflight({
        'status': 'ACTIVE',
        'ownerProjectId': 'shared-platform',
        'atomicity': 'LEASE_CAS',
        'backendRef': 'shared-platform:test-store',
    })['allowed']

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
            'codingTargetLocked': True,
            'codingPathPolicyLocked': True,
            'codingContractDigestBound': True,
            'codingReceiptDeterministicallyVerified': True,
            'codingCandidateOnly': True,
            'codingIndependentReviewRequired': True,
            'executionPlaneRegistryNormalized': True,
            'executionPlaneScopeDelegationRequired': True,
            'executionPlaneWrongRouteDenied': True,
            'executionPlanePaidApprovalRequired': True,
            'executionPlaneRetryFailClosed': True,
            'durableStoreOwnershipFailClosed': True,
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
