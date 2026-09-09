#!/usr/bin/env python3
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path

AWS_INFRA_DIR = Path(__file__).resolve().parents[1]
if str(AWS_INFRA_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_INFRA_DIR))

from producer_receipt import (  # noqa: E402
    ProducerInvocationError,
    RECEIPT_SCHEMA,
    canonical_json_bytes,
    finalize_receipt,
    new_receipt,
    producer_error_evidence,
    sha256_json,
    validate_receipt_context,
)

ACTIVE_MODEL_ID = "qwen.qwen3-coder-30b-a3b-v1:0"
MODEL_ID = os.environ.get("SEMANTIC_REVIEW_MODEL_ID", ACTIVE_MODEL_ID)
MAX_REVIEW_COUNT = 3
REVIEW_COUNT = int(os.environ.get("SEMANTIC_REVIEW_COUNT", "3"))
if REVIEW_COUNT not in {2, 3}:
    raise SystemExit("SEMANTIC_REVIEW_COUNT must be exactly 2 or 3 for legacy compatibility")
INPUT_PATH = Path(os.environ.get("SEMANTIC_REVIEW_INPUT", sys.argv[1] if len(sys.argv) > 1 else "semantic-review-input.json"))
OUTPUT_PATH = Path(os.environ.get("SEMANTIC_REVIEW_OUTPUT", "semantic-review-pool-results.json"))
INTAKE_OUTPUT_PATH = Path(os.environ.get("SEMANTIC_REVIEW_INTAKE_OUTPUT", "semantic-review-intake-result.json"))
SEMANTIC_POOL_ID = "opened-arm.semantic-review-pool"
SEMANTIC_WORKFLOW_PATHS = frozenset({
    ".github/workflows/aws-semantic-review-intake.yml",
})
MAX_FINDINGS_PER_REVIEWER = 5

# Amazon Bedrock's active Qwen3-Coder-30B-A3B contract is 256K input/output
# context with at most 16K output tokens. The production profile fixes output at
# 800 tokens. Keep a deterministic 10% safety margin plus explicit provider
# framing reserve; callers cannot change any of these values.
MODEL_CONTEXT_WINDOW_TOKENS = 256 * 1024
MODEL_MAX_OUTPUT_TOKENS = 16 * 1024
SERVER_OUTPUT_TOKENS = 800
PROVIDER_FRAMING_RESERVE_TOKENS = 4 * 1024
SAFETY_HEADROOM_TOKENS = MODEL_CONTEXT_WINDOW_TOKENS // 10
SEMANTIC_INPUT_BUDGET_TOKENS = (
    MODEL_CONTEXT_WINDOW_TOKENS
    - SERVER_OUTPUT_TOKENS
    - PROVIDER_FRAMING_RESERVE_TOKENS
    - SAFETY_HEADROOM_TOKENS
)

# GitHub workflow_dispatch permits 65,535 characters across inputs. Leave room
# for the fixed review_count/dry_run inputs and JSON framing.
DISPATCH_PAYLOAD_B64_MAX_CHARS = 60_000
INFLATED_PAYLOAD_MAX_BYTES = MODEL_CONTEXT_WINDOW_TOKENS
MAX_REQUIREMENTS_BYTES = 12_000
REQUEST_V1_SCHEMA = "semantic-review-request-v1"
CONTEXT_V1_SCHEMA = "semantic-review-context-v1"
INTAKE_V1_SCHEMA = "semantic-review-intake-v1"

SEMANTIC_OUTPUT_PROFILE = {
    "schema": "qwen-semantic-review-output-profile-v1",
    "version": 1,
    "profile_id": "qwen-semantic-review-bounded-v1",
    "model_id": ACTIVE_MODEL_ID,
    "temperature": 0.25,
    "top_p": 0.9,
    "max_output_tokens": SERVER_OUTPUT_TOKENS,
    "normalized_finding_cap": MAX_FINDINGS_PER_REVIEWER,
    "overflow_policy": "INCOMPLETE_FAIL_CLOSED_RAW_PRESERVED",
}

REVIEWER_PROFILES = [
    ("A", "Prioritize correctness, state transitions, data flow, and hidden edge cases."),
    ("B", "Prioritize authorization, trust boundaries, security, and unsafe assumptions."),
    ("C", "Prioritize async behavior, nullability, boundary conditions, races, and failure handling."),
]
REVIEWERS = REVIEWER_PROFILES[:REVIEW_COUNT]

CANONICAL_CATEGORIES = {
    "correctness", "authorization", "security", "async", "boundary",
    "nullability", "data_integrity", "error_handling", "concurrency",
    "resource", "performance", "api_contract", "other",
}
TITLE_STOPWORDS = {
    "a", "an", "and", "the", "to", "of", "in", "on", "for", "with",
    "check", "issue", "potential", "possible", "missing",
}
TITLE_TOKEN_MAP = {
    "bypasses": "bypass", "bypassed": "bypass", "bypassing": "bypass",
    "unauthorized": "authorization", "unauthorised": "authorization",
    "authorisation": "authorization",
}
VALID_JSON_SIMPLE_ESCAPES = set('"\\/bfnrt')
HEX_DIGITS = set("0123456789abcdefABCDEF")
FINDING_KEYS = {"category", "severity", "line", "title", "rationale", "confidence"}
TASK_RE = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")


def validate_request(raw: object) -> dict:
    if not isinstance(raw, dict) or set(raw) - {
        "task_id", "language", "requirements", "code", "receipt_context", "_intake",
    }:
        raise ValueError("review_request_shape_invalid")
    task_id = raw.get("task_id")
    language = raw.get("language")
    requirements = raw.get("requirements")
    code = raw.get("code")
    if not isinstance(task_id, str) or not TASK_RE.fullmatch(task_id):
        raise ValueError("review_task_id_invalid")
    if not isinstance(language, str) or not language.strip() or len(language) > 40:
        raise ValueError("review_language_invalid")
    if not isinstance(requirements, str) or not requirements.strip() or len(requirements.encode()) > 12000:
        raise ValueError("review_requirements_invalid")
    intake = raw.get("_intake")
    if not isinstance(code, str) or not code.strip() or (intake is None and len(code.encode()) > 48000):
        raise ValueError("review_code_invalid")
    receipt_context = validate_receipt_context(raw.get("receipt_context"))
    if receipt_context is not None and REVIEW_COUNT != 3:
        raise ValueError("receipt_v1_requires_exactly_three_reviewers")
    normalized = {
        "task_id": task_id,
        "language": language.strip(),
        "requirements": requirements,
        "code": code,
        "receipt_context": receipt_context,
    }
    if intake is not None:
        if not isinstance(intake, dict) or set(intake) != {"transport", "context", "budget"}:
            raise ValueError("review_intake_evidence_invalid")
        context = intake.get("context")
        budget = intake.get("budget")
        transport = intake.get("transport")
        if not isinstance(context, dict) or not isinstance(budget, dict) or not isinstance(transport, dict):
            raise ValueError("review_intake_evidence_invalid")
        versioned = context.get("schema") == CONTEXT_V1_SCHEMA
        try:
            expected_context = exact_context_metadata(context if versioned else None, code, versioned)
            expected_budget = semantic_budget_evidence(normalized, expected_context)
        except IntakeError as exc:
            raise ValueError("review_intake_evidence_incomplete") from exc
        if context != expected_context or budget != expected_budget:
            raise ValueError("review_intake_evidence_incomplete")
        normalized["_intake"] = intake
    return normalized


def numbered(code: str) -> str:
    return "\n".join(f"{i:04d}: {line}" for i, line in enumerate(code.splitlines(), 1))


def prompt_for(request: dict, reviewer_id: str, focus: str) -> str:
    return f"""You are one independent semantic code reviewer in a review pool.
Your output is HYPOTHESIS ONLY. You do not have authority to mark code PASS or FAIL.
Review the supplied requirements and code independently. Do not assume another reviewer will catch anything.
{focus}

Return exactly one JSON object and no markdown or prose:
{{
  "reviewer_id": "{reviewer_id}",
  "findings": [
    {{
      "category": "correctness|authorization|security|async|boundary|nullability|data_integrity|error_handling|concurrency|resource|performance|api_contract|other",
      "severity": "high|medium|low",
      "line": 0,
      "title": "brief title",
      "rationale": "specific explanation tied to the supplied code and requirements",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Only report concrete defects supported by the supplied requirements/code.
- Do not report style preferences or speculative missing context.
- Use an empty findings array when no concrete defect is found.
- line is the first materially relevant line, or 0 only when no single line applies.
- Maximum 5 findings.
- Pick the closest category from the fixed category list; do not invent category names.
- Output valid JSON. Backslashes inside JSON strings must be escaped as JSON backslashes.

Task ID: {request.get('task_id', 'unknown')}
Language: {request.get('language', 'unknown')}
Requirements:
{request.get('requirements', '')}

Code:
{numbered(request.get('code', ''))}
"""


class IntakeError(ValueError):
    def __init__(self, code: str, message: str, evidence: dict | None = None):
        super().__init__(message)
        self.code = code
        self.evidence = evidence or {}


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decode_request_payload(raw_b64: str) -> tuple[bytes, dict]:
    if not raw_b64 or len(raw_b64) > DISPATCH_PAYLOAD_B64_MAX_CHARS:
        raise IntakeError(
            "semantic_review_payload_transport_invalid",
            "payload_b64 is missing or exceeds the server-defined dispatch limit",
            {"payload_b64_chars": len(raw_b64), "payload_b64_max_chars": DISPATCH_PAYLOAD_B64_MAX_CHARS},
        )
    try:
        packed = base64.b64decode(raw_b64, validate=True)
    except Exception as exc:
        raise IntakeError("semantic_review_payload_base64_invalid", "payload_b64 is not valid base64") from exc
    if base64.b64encode(packed).decode("ascii") != raw_b64:
        raise IntakeError("semantic_review_payload_base64_noncanonical", "payload_b64 must use canonical base64")

    compressed = packed.startswith(b"\x1f\x8b")
    if compressed:
        try:
            inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
            raw = inflater.decompress(packed, INFLATED_PAYLOAD_MAX_BYTES + 1)
            if len(raw) > INFLATED_PAYLOAD_MAX_BYTES or inflater.unconsumed_tail:
                raise IntakeError(
                    "semantic_review_payload_inflated_too_large",
                    "compressed payload exceeds the server-defined inflated limit",
                )
            raw += inflater.flush()
            if not inflater.eof or inflater.unused_data:
                raise IntakeError(
                    "semantic_review_payload_gzip_invalid",
                    "payload must contain exactly one complete gzip member",
                )
        except IntakeError:
            raise
        except Exception as exc:
            raise IntakeError("semantic_review_payload_gzip_invalid", "payload gzip stream is invalid") from exc
    else:
        raw = packed

    if len(raw) > INFLATED_PAYLOAD_MAX_BYTES:
        raise IntakeError(
            "semantic_review_payload_inflated_too_large",
            "payload exceeds the server-defined inflated limit",
        )
    return raw, {
        "encoding": "gzip+base64" if compressed else "json+base64",
        "payload_b64_chars": len(raw_b64),
        "packed_bytes": len(packed),
        "inflated_bytes": len(raw),
        "inflated_sha256": sha256_hex(raw),
    }


def exact_context_metadata(raw_context: object, code: str, versioned: bool) -> dict:
    code_bytes = code.encode("utf-8")
    digest = sha256_hex(code_bytes)
    markers = re.findall(r"^--- (.+) \([^)]+\) ---$", code, re.MULTILINE)
    if not versioned:
        return {
            "schema": "semantic-review-context-legacy-v0",
            "complete": True,
            "original_bytes": len(code_bytes),
            "admitted_bytes": len(code_bytes),
            "code_sha256": digest,
            "changed_paths": markers,
            "omitted_files": [],
            "omitted_sections": [],
        }

    if not isinstance(raw_context, dict):
        raise IntakeError(
            "semantic_review_context_evidence_missing",
            "versioned requests require exact context completeness evidence",
        )
    allowed = {
        "schema", "complete", "original_bytes", "admitted_bytes", "code_sha256",
        "changed_paths", "omitted_files", "omitted_sections",
    }
    unknown = sorted(set(raw_context) - allowed)
    if unknown:
        raise IntakeError("semantic_review_context_evidence_invalid", f"unknown context evidence keys: {unknown}")
    changed_paths = raw_context.get("changed_paths")
    omitted_files = raw_context.get("omitted_files")
    omitted_sections = raw_context.get("omitted_sections")
    if (
        raw_context.get("schema") != CONTEXT_V1_SCHEMA
        or raw_context.get("complete") is not True
        or raw_context.get("original_bytes") != len(code_bytes)
        or raw_context.get("admitted_bytes") != len(code_bytes)
        or raw_context.get("code_sha256") != digest
        or not isinstance(changed_paths, list)
        or not all(isinstance(item, str) and item for item in changed_paths)
        or len(changed_paths) != len(set(changed_paths))
        or changed_paths != markers
        or omitted_files != []
        or omitted_sections != []
    ):
        raise IntakeError(
            "semantic_review_context_incomplete",
            "context evidence must prove one complete, untruncated review scope",
            {
                "original_context_bytes": raw_context.get("original_bytes"),
                "admitted_context_bytes": raw_context.get("admitted_bytes"),
                "actual_context_bytes": len(code_bytes),
                "omitted_files": omitted_files if isinstance(omitted_files, list) else None,
                "omitted_sections": omitted_sections if isinstance(omitted_sections, list) else None,
            },
        )
    return dict(raw_context)


def semantic_budget_evidence(request: dict, context: dict) -> dict:
    if MODEL_ID != ACTIVE_MODEL_ID:
        raise IntakeError(
            "semantic_review_model_profile_unregistered",
            "semantic review model must match the server-defined active profile",
        )
    prompt_bytes = {
        reviewer_id: len(prompt_for(request, reviewer_id, focus).encode("utf-8"))
        for reviewer_id, focus in REVIEWER_PROFILES
    }
    estimate = max(prompt_bytes.values())
    complete = estimate <= SEMANTIC_INPUT_BUDGET_TOKENS
    evidence = {
        "model_id": ACTIVE_MODEL_ID,
        "model_context_window_tokens": MODEL_CONTEXT_WINDOW_TOKENS,
        "model_max_output_tokens": MODEL_MAX_OUTPUT_TOKENS,
        "configured_output_tokens": SERVER_OUTPUT_TOKENS,
        "provider_framing_reserve_tokens": PROVIDER_FRAMING_RESERVE_TOKENS,
        "safety_headroom_tokens": SAFETY_HEADROOM_TOKENS,
        "input_budget_tokens": SEMANTIC_INPUT_BUDGET_TOKENS,
        "token_estimator": "utf8_bytes_upper_bound_v1",
        "prompt_token_estimate_by_reviewer": prompt_bytes,
        "prompt_token_estimate": estimate,
        "original_context_bytes": context["original_bytes"],
        "admitted_context_bytes": context["admitted_bytes"] if complete else 0,
        "reviewer_scope_sha256": {reviewer_id: context["code_sha256"] for reviewer_id, _ in REVIEWER_PROFILES},
        "omitted_files": [] if complete else context.get("changed_paths", []),
        "omitted_sections": [] if complete else ["semantic_context"],
        "completeness": complete,
    }
    if not complete:
        raise IntakeError(
            "semantic_review_context_budget_exceeded",
            "complete semantic packet exceeds the server-defined model input budget",
            evidence,
        )
    return evidence


def validate_and_normalize_request(raw_b64: str, review_count: int) -> tuple[dict, dict, dict]:
    if review_count != MAX_REVIEW_COUNT:
        raise IntakeError(
            "semantic_review_count_invalid",
            f"review_count must be exactly {MAX_REVIEW_COUNT}",
        )
    raw, transport = decode_request_payload(raw_b64)
    try:
        request = json.loads(raw)
    except Exception as exc:
        raise IntakeError("semantic_review_payload_json_invalid", "payload JSON is invalid") from exc
    if not isinstance(request, dict):
        raise IntakeError("semantic_review_payload_invalid", "payload must decode to one JSON object")

    versioned = "schema" in request
    allowed = {"task_id", "language", "requirements", "code", "receipt_context"}
    if versioned:
        allowed |= {"schema", "context"}
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise IntakeError("semantic_review_payload_invalid", f"unknown payload keys: {unknown}")
    if versioned and request.get("schema") != REQUEST_V1_SCHEMA:
        raise IntakeError("semantic_review_payload_schema_invalid", "unsupported semantic request schema")

    task_id = request.get("task_id")
    language = request.get("language")
    requirements = request.get("requirements")
    code = request.get("code")
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,160}", task_id):
        raise IntakeError("semantic_review_task_id_invalid", "task_id is missing or invalid")
    if not isinstance(language, str) or not language.strip() or len(language) > 40:
        raise IntakeError("semantic_review_language_invalid", "language is missing or invalid")
    if (
        not isinstance(requirements, str)
        or not requirements.strip()
        or len(requirements.encode("utf-8")) > MAX_REQUIREMENTS_BYTES
    ):
        raise IntakeError("semantic_review_requirements_invalid", "requirements are missing or exceed 12 KiB")
    if not isinstance(code, str) or not code.strip():
        raise IntakeError("semantic_review_code_invalid", "code is missing or empty")

    try:
        receipt_context = validate_receipt_context(request.get("receipt_context"))
    except ValueError as exc:
        raise IntakeError("semantic_review_receipt_context_invalid", str(exc)) from exc
    if receipt_context is not None and review_count != MAX_REVIEW_COUNT:
        raise IntakeError(
            "semantic_review_receipt_reviewer_count_invalid",
            "versioned producer receipts require exactly three reviewers",
        )

    context = exact_context_metadata(request.get("context"), code, versioned)
    normalized = {
        "task_id": task_id,
        "language": language.strip(),
        "requirements": requirements,
        "code": code,
    }
    if receipt_context is not None:
        normalized["receipt_context"] = receipt_context
    budget = semantic_budget_evidence(normalized, context)
    normalized["_intake"] = {"transport": transport, "context": context, "budget": budget}
    return normalized, transport, budget


def validate_intake_from_env() -> int:
    raw_b64 = os.environ.get("PAYLOAD_B64", "")
    dry_run = os.environ.get("DRY_RUN", "").lower() == "true"
    try:
        review_count = int(os.environ.get("SEMANTIC_REVIEW_COUNT", "0"))
    except ValueError:
        review_count = 0
    receipt = {
        "schema": INTAKE_V1_SCHEMA,
        "authority": "HYPOTHESIS_ONLY",
        "production_pass_fail_authority": False,
        "automatic_retry": False,
        "review_budget": MAX_REVIEW_COUNT,
        "coding_calls": 0,
        "dry_run": dry_run,
        "provider_invoked": False,
    }
    try:
        normalized, transport, budget = validate_and_normalize_request(raw_b64, review_count)
    except IntakeError as exc:
        receipt.update({
            "completeness": False,
            "provider_invocation_allowed": False,
            "error_code": exc.code,
            "error": str(exc),
            "evidence": exc.evidence,
        })
        INTAKE_OUTPUT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True))
        print(json.dumps(receipt, separators=(",", ":")))
        return 2

    INPUT_PATH.write_text(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")))
    receipt.update({
        "task_id": normalized["task_id"],
        "completeness": True,
        "provider_invocation_allowed": not dry_run,
        "transport": transport,
        "context": normalized["_intake"]["context"],
        "budget": budget,
        "next_gate": "paid_semantic_review_requires_explicit_invocation" if dry_run else "semantic_review_in_progress",
    })
    if normalized.get("receipt_context") is not None:
        receipt["producer_receipt_schema"] = RECEIPT_SCHEMA
        receipt["receipt_context_present"] = True
    INTAKE_OUTPUT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({
        "schema": receipt["schema"],
        "task_id": receipt["task_id"],
        "completeness": receipt["completeness"],
        "transport": transport["encoding"],
        "prompt_token_estimate": budget["prompt_token_estimate"],
        "input_budget_tokens": budget["input_budget_tokens"],
    }, separators=(",", ":")))
    return 0


def repair_invalid_json_string_escapes(value: str) -> str:
    r"""Escape only invalid backslash sequences that occur inside JSON strings.

    This deliberately does not repair missing quotes/braces, infer fields, or complete
    truncated JSON. It only turns a model-emitted literal such as \s into \\s so the
    intended backslash remains literal text in the decoded JSON string.
    """
    output: list[str] = []
    in_string = False
    i = 0
    while i < len(value):
        ch = value[i]
        if not in_string:
            output.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if ch == '"':
            output.append(ch)
            in_string = False
            i += 1
            continue

        if ch != "\\":
            output.append(ch)
            i += 1
            continue

        if i + 1 >= len(value):
            output.append(ch)
            i += 1
            continue

        nxt = value[i + 1]
        if nxt in VALID_JSON_SIMPLE_ESCAPES:
            output.extend((ch, nxt))
            i += 2
            continue

        if nxt == "u" and i + 5 < len(value) and all(c in HEX_DIGITS for c in value[i + 2:i + 6]):
            output.append(value[i:i + 6])
            i += 6
            continue

        output.append("\\\\")
        i += 1

    return "".join(output)


def json_candidates(cleaned: str) -> list[str]:
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, re.S)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))
    return candidates


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    last_error: Exception | None = None
    for candidate in json_candidates(cleaned):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as error:
            last_error = error
            repaired = repair_invalid_json_string_escapes(candidate)
            if repaired == candidate:
                continue
            try:
                obj = json.loads(repaired)
            except json.JSONDecodeError as repaired_error:
                last_error = repaired_error
                continue

        if not isinstance(obj, dict) or not isinstance(obj.get("findings"), list):
            raise ValueError("response must contain findings array")
        return obj

    if last_error is not None:
        raise last_error
    raise ValueError("response must contain one JSON object")


def parser_self_test() -> None:
    valid = '{"reviewer_id":"A","findings":[{"rationale":"line\\nnext and \\\\d and \\u1234"}]}'
    valid_obj = extract_json(valid)
    assert valid_obj["reviewer_id"] == "A"
    assert valid_obj["findings"][0]["rationale"] == "line\nnext and \\d and ሴ"

    malformed = r'{"reviewer_id":"A","findings":[{"rationale":"regex \s and \d"}]}'
    recovered = extract_json(malformed)
    assert recovered["findings"][0]["rationale"] == r"regex \s and \d"

    fenced = '```json\n' + malformed + '\n```'
    assert extract_json(fenced) == recovered

    structurally_invalid = [
        r'{"reviewer_id":"A","findings":[{"rationale":"truncated \s"}',
        '{"reviewer_id":"A"}',
        'not json at all',
    ]
    for sample in structurally_invalid:
        try:
            extract_json(sample)
        except (json.JSONDecodeError, ValueError):
            pass
        else:
            raise AssertionError(f"structurally invalid model output was admitted: {sample!r}")

    print("Semantic review JSON parser self-test: PASS")


def canonicalize_category(raw_category: str, title: str, rationale: str) -> str:
    raw = re.sub(r"[^a-z0-9_]+", "_", raw_category.strip().lower()).strip("_")
    if raw in CANONICAL_CATEGORIES:
        return raw
    text = f"{raw} {title} {rationale}".lower()
    if any(token in text for token in ("authorization", "authorisation", "permission", "access control", "access_control", "auth bypass", "auth_bypass", "unauthorized", "unauthorised")):
        return "authorization"
    if any(token in text for token in ("injection", "xss", "csrf", "secret", "credential", "crypto", "security")):
        return "security"
    if any(token in text for token in ("await", "promise", "async")):
        return "async"
    if any(token in text for token in ("null", "nullable", "undefined")):
        return "nullability"
    if any(token in text for token in ("boundary", "off by one", "off_by_one", "range")):
        return "boundary"
    if any(token in text for token in ("transaction", "integrity", "data loss", "data_loss", "corrupt")):
        return "data_integrity"
    if any(token in text for token in ("exception", "error handling", "error_handling", "swallow")):
        return "error_handling"
    if any(token in text for token in ("race", "deadlock", "concurrent", "concurrency")):
        return "concurrency"
    if any(token in text for token in ("resource", "leak", "close", "cleanup")):
        return "resource"
    if any(token in text for token in ("performance", "slow", "latency", "complexity")):
        return "performance"
    if any(token in text for token in ("api", "contract", "schema")):
        return "api_contract"
    return "correctness" if raw else "other"


def normalize_finding(raw: dict) -> dict:
    severity = str(raw.get("severity", "medium")).lower()
    if severity not in {"high", "medium", "low"}:
        severity = "medium"
    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0)))
    title = str(raw.get("title", "")).strip()[:200]
    rationale = str(raw.get("rationale", "")).strip()[:1200]
    category = canonicalize_category(str(raw.get("category", "")), title, rationale)
    return {
        "category": category,
        "severity": severity,
        "line": max(0, int(raw.get("line", 0) or 0)),
        "title": title,
        "rationale": rationale,
        "confidence": confidence,
    }


def validate_strict_finding(raw: object, index: int) -> dict:
    if not isinstance(raw, dict) or set(raw) != FINDING_KEYS:
        raise ValueError(f"finding_{index}_shape_invalid")
    category = raw.get("category")
    severity = raw.get("severity")
    line = raw.get("line")
    title = raw.get("title")
    rationale = raw.get("rationale")
    confidence = raw.get("confidence")
    if category not in CANONICAL_CATEGORIES:
        raise ValueError(f"finding_{index}_category_invalid")
    if severity not in {"high", "medium", "low"}:
        raise ValueError(f"finding_{index}_severity_invalid")
    if not isinstance(line, int) or isinstance(line, bool) or line < 0:
        raise ValueError(f"finding_{index}_line_invalid")
    if not isinstance(title, str) or not title.strip() or len(title.strip()) > 200:
        raise ValueError(f"finding_{index}_title_invalid")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale.strip()) > 1200:
        raise ValueError(f"finding_{index}_rationale_invalid")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or confidence < 0
        or confidence > 1
    ):
        raise ValueError(f"finding_{index}_confidence_invalid")
    return {
        "category": category,
        "severity": severity,
        "line": line,
        "title": title.strip(),
        "rationale": rationale.strip(),
        "confidence": float(confidence),
    }


def usage_evidence(raw: object) -> dict:
    raw_usage = raw if isinstance(raw, dict) else {}

    def token(field: str) -> int | None:
        value = raw_usage.get(field)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    input_tokens = token("inputTokens")
    output_tokens = token("outputTokens")
    total_tokens = token("totalTokens")
    return {
        "available": input_tokens is not None and output_tokens is not None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw": raw_usage,
    }


def normalize_reviewer_response(
    reviewer_id: str,
    raw_content: object,
    stop_reason: object,
    raw_usage: object,
    *,
    strict: bool,
) -> dict:
    content = raw_content if isinstance(raw_content, list) else []
    text_parts = [
        part["text"]
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    model_text = "\n".join(text_parts).strip()
    content_complete = bool(content) and len(text_parts) == len(content) and all(
        isinstance(part, dict) and set(part) == {"text"} for part in content
    )
    finish_reason = stop_reason if isinstance(stop_reason, str) else None
    usage = usage_evidence(raw_usage)
    issues: list[str] = []
    truncated = finish_reason == "max_tokens"
    overflow = False
    raw_finding_count = None
    findings: list[dict] = []

    if not content_complete or not model_text:
        issues.append("raw_model_output_missing_or_unsupported")
    if strict and finish_reason != "end_turn":
        issues.append("model_finish_reason_not_complete")
    elif not strict and truncated:
        issues.append("model_output_truncated_by_token_limit")
    if not usage["available"]:
        issues.append("usage_evidence_missing")

    if model_text and not truncated:
        try:
            obj = json.loads(model_text) if strict else extract_json(model_text)
            if not isinstance(obj, dict):
                raise ValueError("reviewer_response_shape_invalid")
            if strict and set(obj) != {"reviewer_id", "findings"}:
                raise ValueError("reviewer_response_shape_invalid")
            if strict and obj.get("reviewer_id") != reviewer_id:
                raise ValueError("reviewer_identity_mismatch")
            raw_findings = obj.get("findings")
            if not isinstance(raw_findings, list):
                raise ValueError("response_must_contain_findings_array")
            raw_finding_count = len(raw_findings)
            overflow = raw_finding_count > MAX_FINDINGS_PER_REVIEWER
            if overflow:
                issues.append("normalized_finding_overflow")
            if strict:
                validated = [validate_strict_finding(item, i) for i, item in enumerate(raw_findings)]
            else:
                validated = [normalize_finding(item) for item in raw_findings if isinstance(item, dict)]
            findings = validated[:MAX_FINDINGS_PER_REVIEWER]
        except (ValueError, json.JSONDecodeError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            issues.append(f"model_output_invalid:{detail[:300]}")

    raw_digest = sha256_json(content)
    normalized = {"reviewer_id": reviewer_id, "findings": findings}
    complete = not issues and not truncated and not overflow
    return {
        "reviewer_id": reviewer_id,
        "status": "completed" if complete else "incomplete",
        "provider_state": "RETURNED",
        "usage": usage,
        "findings": findings,
        "raw_model_output": {
            "content": content,
            "text": model_text,
            "digest_algorithm": "sha256",
            "digest": raw_digest,
            "byte_count": len(canonical_json_bytes(content)),
        },
        "normalized_output": {
            "value": normalized,
            "derived_from_raw_digest": raw_digest,
            "digest_algorithm": "sha256",
            "digest": sha256_json(normalized),
        },
        "finish_reason": finish_reason,
        "normalization": {
            "complete": complete,
            "raw_finding_count": raw_finding_count,
            "normalized_finding_count": len(findings),
            "finding_cap": MAX_FINDINGS_PER_REVIEWER,
            "omitted_finding_count": max(0, (raw_finding_count or 0) - len(findings)),
            "truncated": truncated,
            "overflow": overflow,
            "issues": list(dict.fromkeys(issues)),
        },
    }


def invoke_one(request: dict, reviewer_id: str, focus: str) -> dict:
    messages = [{"role": "user", "content": [{"text": prompt_for(request, reviewer_id, focus)}]}]
    inference = {
        "maxTokens": SEMANTIC_OUTPUT_PROFILE["max_output_tokens"],
        "temperature": SEMANTIC_OUTPUT_PROFILE["temperature"],
        "topP": SEMANTIC_OUTPUT_PROFILE["top_p"],
    }
    # Linux limits each argv entry to MAX_ARG_STRLEN (normally 128 KiB), which
    # is smaller than the admitted model-context budget. Let AWS CLI load the
    # structured messages from a private temporary file so a complete large
    # prompt never becomes one process argument. Each reviewer gets a distinct
    # file and the context manager removes it after the provider process exits.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"semantic-review-{reviewer_id.lower()}-",
        suffix="-messages.json",
    ) as messages_file:
        json.dump(messages, messages_file, ensure_ascii=False, separators=(",", ":"))
        messages_file.flush()
        cmd = [
            "aws", "--cli-connect-timeout", "5", "--cli-read-timeout", "60",
            "bedrock-runtime", "converse", "--model-id", MODEL_ID,
            "--messages", f"file://{messages_file.name}",
            "--inference-config", json.dumps(inference, separators=(",", ":")),
            "--output", "json", "--no-cli-pager",
        ]
        started = time.monotonic()
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=70)
        except subprocess.TimeoutExpired as exc:
            raise ProducerInvocationError(
                "AMBIGUOUS", "provider_response_timeout", "No terminal provider response was observed within 70 seconds."
            ) from exc
    latency_ms = round((time.monotonic() - started) * 1000)
    if proc.returncode != 0:
        raise ProducerInvocationError(
            "FAILED",
            "provider_cli_failed",
            proc.stderr.strip() or proc.stdout.strip() or f"aws_cli_exit_{proc.returncode}",
        )
    try:
        response = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProducerInvocationError(
            "INCOMPLETE", "provider_response_invalid_json", proc.stdout[:1000]
        ) from exc
    review = normalize_reviewer_response(
        reviewer_id,
        response.get("output", {}).get("message", {}).get("content", []),
        response.get("stopReason"),
        response.get("usage"),
        strict=request.get("receipt_context") is not None,
    )
    review["latency_ms"] = latency_ms
    review["output_profile"] = dict(SEMANTIC_OUTPUT_PROFILE)
    return review


def failed_review(reviewer_id: str, error: object) -> dict:
    evidence = producer_error_evidence(error)
    normalized = {"reviewer_id": reviewer_id, "findings": []}
    raw_digest = sha256_json([])
    return {
        "reviewer_id": reviewer_id,
        "status": "error" if evidence["state"] == "FAILED" else "incomplete",
        "error": evidence,
        "provider_state": evidence["state"],
        "latency_ms": 0,
        "usage": {
            "available": False,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "raw": {},
        },
        "findings": [],
        "raw_model_output": {
            "content": [],
            "text": "",
            "digest_algorithm": "sha256",
            "digest": raw_digest,
            "byte_count": len(canonical_json_bytes([])),
        },
        "normalized_output": {
            "value": normalized,
            "derived_from_raw_digest": raw_digest,
            "digest_algorithm": "sha256",
            "digest": sha256_json(normalized),
        },
        "finish_reason": None,
        "normalization": {
            "complete": False,
            "raw_finding_count": None,
            "normalized_finding_count": 0,
            "finding_cap": MAX_FINDINGS_PER_REVIEWER,
            "omitted_finding_count": 0,
            "truncated": False,
            "overflow": False,
            "issues": [evidence["code"]],
        },
        "output_profile": dict(SEMANTIC_OUTPUT_PROFILE),
    }


def title_tokens(title: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", title.lower())
    normalized = []
    for token in tokens:
        token = TITLE_TOKEN_MAP.get(token, token)
        if token not in TITLE_STOPWORDS:
            normalized.append(token)
    return set(normalized)


def title_similarity(a: str, b: str) -> float:
    left, right = title_tokens(a), title_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def same_cluster(a: dict, b: dict) -> bool:
    if a["line"] != b["line"]:
        return False
    if a["category"] == b["category"]:
        return True
    return title_similarity(a["title"], b["title"]) >= 0.45


def aggregate(reviews: list[dict]) -> list[dict]:
    clusters = []
    for review in reviews:
        if review.get("status") != "completed":
            continue
        reviewer_id = review["reviewer_id"]
        for finding in review.get("findings", []):
            target = None
            for cluster in clusters:
                if any(same_cluster(finding, obs) for obs in cluster["observations"]):
                    target = cluster
                    break
            if target is None:
                target = {
                    "category": finding["category"],
                    "category_variants": [],
                    "line": finding["line"],
                    "support_count": 0,
                    "reviewer_ids": [],
                    "observations": [],
                }
                clusters.append(target)
            if reviewer_id not in target["reviewer_ids"]:
                target["reviewer_ids"].append(reviewer_id)
            target["observations"].append({"reviewer_id": reviewer_id, **finding})

    for cluster in clusters:
        variants = [obs["category"] for obs in cluster["observations"]]
        counts = Counter(variants)
        cluster["category"] = sorted(counts, key=lambda c: (-counts[c], c))[0]
        cluster["category_variants"] = sorted(counts)
        cluster["support_count"] = len(cluster["reviewer_ids"])
    clusters.sort(key=lambda x: (-x["support_count"], x["line"], x["category"]))
    return clusters


def build_pool_result(request: dict, input_bytes: bytes, reviews: list[dict], wall_latency_ms: int, *, environ=None) -> dict:
    reviews = sorted(reviews, key=lambda item: item["reviewer_id"])
    aggregated = aggregate(reviews)
    completed = sum(review.get("status") == "completed" for review in reviews)

    def token_total(field: str) -> int:
        return sum(
            value
            for value in (review.get("usage", {}).get(field) for review in reviews)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        )

    total_usage = {
        "input_tokens": token_total("input_tokens"),
        "output_tokens": token_total("output_tokens"),
    }
    output = {
        "schema": "semantic-review-pool-v0",
        "task_id": request.get("task_id", "unknown"),
        "model_id": MODEL_ID,
        "authority": "HYPOTHESIS_ONLY",
        "production_pass_fail_authority": False,
        "automatic_retry": False,
        "review_budget": REVIEW_COUNT,
        "completed_reviews": completed,
        "coding_calls": 0,
        "paid_adjudicator_calls": 0,
        "wall_latency_ms": wall_latency_ms,
        "usage": total_usage,
        "reviews": reviews,
        "aggregated_findings": aggregated,
        "finding_count": len(aggregated),
        "result_state": "FINDINGS_OBSERVED" if aggregated else "NO_FINDINGS_OBSERVED",
        "next_gate": "deterministic_verification_required",
    }
    intake = request.get("_intake")
    if isinstance(intake, dict):
        output["context_budget"] = intake.get("budget")

    receipt, identity_issues = new_receipt(
        context=request.get("receipt_context"),
        input_bytes=input_bytes,
        producer_kind="SEMANTIC_REVIEW_POOL",
        producer_id=SEMANTIC_POOL_ID,
        authority="HYPOTHESIS_ONLY",
        model_id=MODEL_ID,
        expected_workflow_paths=SEMANTIC_WORKFLOW_PATHS,
        environ=environ,
    )
    if receipt is not None:
        provider_failed = any(review.get("provider_state") == "FAILED" for review in reviews)
        truncated = any(review.get("normalization", {}).get("truncated") is True for review in reviews)
        overflow = any(review.get("normalization", {}).get("overflow") is True for review in reviews)
        raw_complete = all(
            bool(review.get("raw_model_output", {}).get("content"))
            and bool(review.get("raw_model_output", {}).get("text"))
            and "raw_model_output_missing_or_unsupported" not in review.get("normalization", {}).get("issues", [])
            for review in reviews
        ) and len(reviews) == REVIEW_COUNT
        normalized_complete = all(
            review.get("normalization", {}).get("complete") is True for review in reviews
        ) and len(reviews) == REVIEW_COUNT
        issues = []
        for review in reviews:
            reviewer_id = review.get("reviewer_id", "unknown")
            for issue in review.get("normalization", {}).get("issues", []):
                issues.append(f"reviewer_{reviewer_id}:{issue}")
            if review.get("provider_state") in {"FAILED", "AMBIGUOUS", "INCOMPLETE"}:
                issues.append(f"reviewer_{reviewer_id}:provider_state_{review['provider_state'].lower()}")
        if [review.get("reviewer_id") for review in reviews] != ["A", "B", "C"]:
            issues.append("reviewer_set_incomplete")

        output_core = {
            "reviewers": reviews,
            "normalized_pool_output": {
                "aggregated_findings": aggregated,
                "finding_count": len(aggregated),
                "digest_algorithm": "sha256",
                "digest": sha256_json(aggregated),
            },
            "usage": total_usage,
            "output_profile": dict(SEMANTIC_OUTPUT_PROFILE),
            "raw_authority": "HYPOTHESIS_ONLY",
            "adjudication": "SEPARATE_SOURCE_GROUNDED_REQUIRED",
            "majority_vote_is_truth": False,
        }
        receipt_output = {
            **output_core,
            "digest_algorithm": "sha256",
            "digest": sha256_json(output_core),
        }
        receipt = finalize_receipt(
            receipt,
            identity_issues=identity_issues,
            output=receipt_output,
            raw_output_complete=raw_complete,
            normalized_output_complete=normalized_complete,
            truncated=truncated,
            overflow=overflow,
            provider_failed=provider_failed,
            issues=issues,
        )
        output["producer_receipt"] = receipt
        if not receipt["completeness"]["complete"]:
            output["result_state"] = "INCOMPLETE"
            output["next_gate"] = "producer_receipt_incomplete_stop"
    return output


def main() -> int:
    input_bytes = INPUT_PATH.read_bytes()
    request = validate_request(json.loads(input_bytes))

    started = time.monotonic()
    reviews = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=REVIEW_COUNT) as executor:
        future_map = {
            executor.submit(invoke_one, request, reviewer_id, focus): reviewer_id
            for reviewer_id, focus in REVIEWERS
        }
        for future in concurrent.futures.as_completed(future_map):
            reviewer_id = future_map[future]
            try:
                reviews.append(future.result())
            except Exception as exc:
                reviews.append(failed_review(reviewer_id, exc))

    output = build_pool_result(
        request,
        input_bytes,
        reviews,
        round((time.monotonic() - started) * 1000),
    )
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({
        "schema": output["schema"],
        "authority": output["authority"],
        "completed_reviews": output["completed_reviews"],
        "finding_count": output["finding_count"],
        "wall_latency_ms": output["wall_latency_ms"],
        "usage": output["usage"],
        "result_state": output["result_state"],
    }, separators=(",", ":")))
    receipt = output.get("producer_receipt")
    if receipt is not None:
        return 0 if receipt["completeness"]["complete"] else 2
    return 0 if output["completed_reviews"] > 0 else 2


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--parser-self-test":
        parser_self_test()
        raise SystemExit(0)
    if len(sys.argv) == 2 and sys.argv[1] == "--validate-intake":
        raise SystemExit(validate_intake_from_env())
    raise SystemExit(main())
