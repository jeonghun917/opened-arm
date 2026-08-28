#!/usr/bin/env python3
import importlib.util
import json
import re
import sys
from pathlib import Path

POOL = Path(__file__).resolve().parents[1] / 'infra' / 'aws' / 'semantic-review' / 'pool.py'
spec = importlib.util.spec_from_file_location('semantic_review_pool_base', POOL)
if spec is None or spec.loader is None:
    raise SystemExit('semantic_review_pool_import_failed')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if module.REVIEW_COUNT != 2:
    raise SystemExit('D/E wrapper requires SEMANTIC_REVIEW_COUNT=2')

module.REVIEWERS = [
    ('D', 'Prioritize API contracts, persistence semantics, idempotency, serialization, and cross-module invariants.'),
    ('E', 'Prioritize resource limits, performance traps, lifecycle cleanup, observability, and operational failure modes.'),
]

_BASE_EXTRACT_JSON = module.extract_json
_TOP_LEVEL_KEYS = {'reviewer_id', 'findings'}
_FINDING_KEYS = {'category', 'severity', 'line', 'title', 'rationale', 'confidence'}


def _strict_repaired_shape(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_KEYS:
        return False
    reviewer_id = value.get('reviewer_id')
    findings = value.get('findings')
    if reviewer_id not in {'D', 'E'} or not isinstance(findings, list) or len(findings) > 5:
        return False
    return all(isinstance(item, dict) and set(item) == _FINDING_KEYS for item in findings)


def _cleaned_candidates(text: str) -> list[str]:
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
    return module.json_candidates(cleaned)


def _extract_json_with_one_missing_comma(text: str) -> dict:
    original_error: json.JSONDecodeError | None = None
    try:
        parsed = _BASE_EXTRACT_JSON(text)
        if not _strict_repaired_shape(parsed):
            raise ValueError('D/E response must match the exact reviewer/findings schema')
        return parsed
    except json.JSONDecodeError as error:
        original_error = error
        if error.msg != "Expecting ',' delimiter":
            raise

    for candidate in _cleaned_candidates(text):
        repaired_escapes = module.repair_invalid_json_string_escapes(candidate)
        try:
            json.loads(repaired_escapes)
        except json.JSONDecodeError as error:
            if error.msg != "Expecting ',' delimiter":
                continue
            patched = repaired_escapes[:error.pos] + ',' + repaired_escapes[error.pos:]
            try:
                parsed = json.loads(patched)
            except json.JSONDecodeError:
                continue
            if _strict_repaired_shape(parsed):
                return parsed

    if original_error is not None:
        raise original_error
    raise ValueError('D/E response could not be repaired safely')


module.extract_json = _extract_json_with_one_missing_comma


def parser_self_test() -> None:
    valid = '{"reviewer_id":"D","findings":[]}'
    assert module.extract_json(valid) == {'reviewer_id': 'D', 'findings': []}

    one_missing_comma = (
        '{"reviewer_id":"D","findings":[{'
        '"category":"api_contract","severity":"medium","line":35,'
        '"title":"Example","rationale":"Bounded example" '
        '"confidence":0.8}]}'
    )
    repaired = module.extract_json(one_missing_comma)
    assert repaired['findings'][0]['confidence'] == 0.8

    invalid_samples = [
        '{"reviewer_id":"D","findings":[{"category":"api_contract" "severity":"medium" "line":35,"title":"x","rationale":"y","confidence":0.8}]}',
        '{"reviewer_id":"D","findings":[{"category":"api_contract","severity":"medium","line":35,"title":"x","rationale":"y"}]}',
        '{"reviewer_id":"D","findings":[{"category":"api_contract","severity":"medium","line":35,"title":"x","rationale":"truncated"}',
    ]
    for sample in invalid_samples:
        try:
            module.extract_json(sample)
        except (json.JSONDecodeError, ValueError):
            pass
        else:
            raise AssertionError(f'unsafe structural repair admitted: {sample!r}')

    print('D/E single-comma JSON parser self-test: PASS')


if '--parser-self-test' in sys.argv:
    parser_self_test()
    raise SystemExit(0)

raise SystemExit(module.main())
