#!/usr/bin/env python3
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
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

MODEL_ID = "qwen.qwen3-coder-30b-a3b-v1:0"
REVIEW_COUNT = int(os.environ.get("SEMANTIC_REVIEW_COUNT", "3"))
if REVIEW_COUNT not in {2, 3}:
    raise SystemExit("SEMANTIC_REVIEW_COUNT must be exactly 2 or 3 for legacy compatibility")
INPUT_PATH = Path(os.environ.get("SEMANTIC_REVIEW_INPUT", sys.argv[1] if len(sys.argv) > 1 else "semantic-review-input.json"))
OUTPUT_PATH = Path(os.environ.get("SEMANTIC_REVIEW_OUTPUT", "semantic-review-pool-results.json"))
SEMANTIC_POOL_ID = "opened-arm.semantic-review-pool"
SEMANTIC_WORKFLOW_PATHS = frozenset({
    ".github/workflows/aws-semantic-review-intake.yml",
})
MAX_FINDINGS_PER_REVIEWER = 5
SEMANTIC_OUTPUT_PROFILE = {
    "schema": "qwen-semantic-review-output-profile-v1",
    "version": 1,
    "profile_id": "qwen-semantic-review-bounded-v1",
    "model_id": MODEL_ID,
    "temperature": 0.25,
    "top_p": 0.9,
    "max_output_tokens": 800,
    "normalized_finding_cap": MAX_FINDINGS_PER_REVIEWER,
    "overflow_policy": "INCOMPLETE_FAIL_CLOSED_RAW_PRESERVED",
}

REVIEWERS = [
    ("A", "Prioritize correctness, state transitions, data flow, and hidden edge cases."),
    ("B", "Prioritize authorization, trust boundaries, security, and unsafe assumptions."),
    ("C", "Prioritize async behavior, nullability, boundary conditions, races, and failure handling."),
][:REVIEW_COUNT]

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
        "task_id", "language", "requirements", "code", "receipt_context",
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
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > 48000:
        raise ValueError("review_code_invalid")
    receipt_context = validate_receipt_context(raw.get("receipt_context"))
    if receipt_context is not None and REVIEW_COUNT != 3:
        raise ValueError("receipt_v1_requires_exactly_three_reviewers")
    return {
        "task_id": task_id,
        "language": language.strip(),
        "requirements": requirements,
        "code": code,
        "receipt_context": receipt_context,
    }


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
    cmd = [
        "aws", "--cli-connect-timeout", "5", "--cli-read-timeout", "60",
        "bedrock-runtime", "converse", "--model-id", MODEL_ID,
        "--messages", json.dumps(messages, separators=(",", ":")),
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
    raise SystemExit(main())
