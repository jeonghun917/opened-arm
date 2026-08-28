#!/usr/bin/env python3
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

POOL = Path(__file__).resolve().parents[1] / 'infra' / 'aws' / 'semantic-review' / 'pool.py'
spec = importlib.util.spec_from_file_location('semantic_review_pool_base', POOL)
if spec is None or spec.loader is None:
    raise SystemExit('semantic_review_pool_import_failed')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ORIGINAL_PROMPT = module.prompt_for
ORIGINAL_EXTRACT = module.extract_json


def strict_prompt(request: dict, reviewer_id: str, focus: str) -> str:
    return ORIGINAL_PROMPT(request, reviewer_id, focus) + r'''
STRICT JSON TRANSPORT RULES:
- title and rationale must each be a single line.
- Do not place a double-quote character inside title or rationale text.
- Do not place a backslash character inside title or rationale text.
- Refer to code identifiers with backticks or plain text instead of quotation marks.
- Keep the exact JSON keys and structure from the required schema.
'''


def repair_unescaped_json_string_quotes(value: str) -> str:
    """Escape only quote characters that cannot legally terminate the current JSON string.

    This does not insert commas, infer keys, add braces, or complete truncated output.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(value):
        ch = value[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
                escaped = False
            i += 1
            continue

        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue

        if ch == '\\':
            out.append(ch)
            escaped = True
            i += 1
            continue

        if ch != '"':
            out.append(ch)
            i += 1
            continue

        j = i + 1
        while j < len(value) and value[j].isspace():
            j += 1
        next_non_ws = value[j] if j < len(value) else ''
        if not next_non_ws or next_non_ws in ',]}:':
            out.append(ch)
            in_string = False
        else:
            out.append('\\"')
        i += 1
    return ''.join(out)


def strict_extract_json(text: str) -> dict:
    try:
        return ORIGINAL_EXTRACT(text)
    except json.JSONDecodeError as original_error:
        cleaned = text.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

        for candidate in module.json_candidates(cleaned):
            repaired = module.repair_invalid_json_string_escapes(candidate)
            repaired = repair_unescaped_json_string_quotes(repaired)
            if repaired == candidate:
                continue
            try:
                obj = json.loads(repaired)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get('findings'), list):
                raise ValueError('response must contain findings array')
            return obj
        raise original_error


def self_test() -> None:
    malformed_quote = (
        '{"reviewer_id":"D","findings":[{"category":"correctness","severity":"medium",'
        '"line":1,"title":"Bad "quoted" value","rationale":"Uses "inner" quotes",'
        '"confidence":0.7}]}'
    )
    obj = strict_extract_json(malformed_quote)
    assert obj['findings'][0]['title'] == 'Bad "quoted" value'
    assert obj['findings'][0]['rationale'] == 'Uses "inner" quotes'

    invalid_escape = r'{"reviewer_id":"D","findings":[{"rationale":"regex \s and \d"}]}'
    obj = strict_extract_json(invalid_escape)
    assert obj['findings'][0]['rationale'] == r'regex \s and \d'

    for sample in (
        '{"reviewer_id":"D","findings":[{"title":"x" "rationale":"missing comma"}]}',
        '{"reviewer_id":"D","findings":[{"title":"truncated"}',
        '{"reviewer_id":"D"}',
    ):
        try:
            strict_extract_json(sample)
        except (json.JSONDecodeError, ValueError):
            pass
        else:
            raise AssertionError(f'structural corruption was admitted: {sample!r}')
    print('Dashboard audit semantic transport self-test: PASS')


module.prompt_for = strict_prompt
module.extract_json = strict_extract_json

if '--self-test' in sys.argv:
    self_test()
    raise SystemExit(0)

group = os.environ.get('AUDIT_REVIEWER_GROUP', 'ABC').upper()
if group == 'ABC':
    if module.REVIEW_COUNT != 3:
        raise SystemExit('ABC audit wrapper requires SEMANTIC_REVIEW_COUNT=3')
elif group == 'DE':
    if module.REVIEW_COUNT != 2:
        raise SystemExit('DE audit wrapper requires SEMANTIC_REVIEW_COUNT=2')
    module.REVIEWERS = [
        ('D', 'Prioritize API contracts, persistence semantics, idempotency, serialization, and cross-module invariants.'),
        ('E', 'Prioritize resource limits, performance traps, lifecycle cleanup, observability, and operational failure modes.'),
    ]
else:
    raise SystemExit('AUDIT_REVIEWER_GROUP must be ABC or DE')

raise SystemExit(module.main())
