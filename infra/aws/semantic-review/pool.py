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

MODEL_ID = os.environ.get("SEMANTIC_REVIEW_MODEL_ID", "qwen.qwen3-coder-30b-a3b-v1:0")
REVIEW_COUNT = int(os.environ.get("SEMANTIC_REVIEW_COUNT", "3"))
MAX_REVIEW_COUNT = 3
INPUT_PATH = Path(os.environ.get("SEMANTIC_REVIEW_INPUT", sys.argv[1] if len(sys.argv) > 1 else "semantic-review-input.json"))
OUTPUT_PATH = Path(os.environ.get("SEMANTIC_REVIEW_OUTPUT", "semantic-review-pool-results.json"))

if REVIEW_COUNT < 1 or REVIEW_COUNT > MAX_REVIEW_COUNT:
    raise SystemExit(f"SEMANTIC_REVIEW_COUNT must be between 1 and {MAX_REVIEW_COUNT}")

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

Task ID: {request.get('task_id', 'unknown')}
Language: {request.get('language', 'unknown')}
Requirements:
{request.get('requirements', '')}

Code:
{numbered(request.get('code', ''))}
"""


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict) or not isinstance(obj.get("findings"), list):
        raise ValueError("response must contain findings array")
    return obj


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


def invoke_one(request: dict, reviewer_id: str, focus: str) -> dict:
    messages = [{"role": "user", "content": [{"text": prompt_for(request, reviewer_id, focus)}]}]
    inference = {"maxTokens": 800, "temperature": 0.25, "topP": 0.9}
    cmd = [
        "aws", "--cli-connect-timeout", "5", "--cli-read-timeout", "60",
        "bedrock-runtime", "converse", "--model-id", MODEL_ID,
        "--messages", json.dumps(messages, separators=(",", ":")),
        "--inference-config", json.dumps(inference, separators=(",", ":")),
        "--output", "json", "--no-cli-pager",
    ]
    started = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=70)
    latency_ms = round((time.monotonic() - started) * 1000)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"aws cli exit {proc.returncode}")
    response = json.loads(proc.stdout)
    content = response.get("output", {}).get("message", {}).get("content", [])
    text = "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and "text" in part).strip()
    obj = extract_json(text)
    findings = [normalize_finding(item) for item in obj.get("findings", [])[:5] if isinstance(item, dict)]
    usage = response.get("usage", {})
    return {
        "reviewer_id": reviewer_id,
        "status": "completed",
        "latency_ms": latency_ms,
        "usage": {
            "input_tokens": int(usage.get("inputTokens", 0) or 0),
            "output_tokens": int(usage.get("outputTokens", 0) or 0),
        },
        "findings": findings,
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
                if any(same_cluster(finding, observation) for observation in cluster["observations"]):
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
        variants = [observation["category"] for observation in cluster["observations"]]
        counts = Counter(variants)
        cluster["category"] = sorted(counts, key=lambda category: (-counts[category], category))[0]
        cluster["category_variants"] = sorted(counts)
        cluster["support_count"] = len(cluster["reviewer_ids"])
    clusters.sort(key=lambda item: (-item["support_count"], item["line"], item["category"]))
    return clusters


def main() -> int:
    request = json.loads(INPUT_PATH.read_text())
    if not isinstance(request, dict) or not request.get("code"):
        raise SystemExit("input must be an object with non-empty code")

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
                reviews.append({
                    "reviewer_id": reviewer_id,
                    "status": "error",
                    "error": str(exc)[:2000],
                    "findings": [],
                })

    reviews.sort(key=lambda item: item["reviewer_id"])
    aggregated = aggregate(reviews)
    completed = sum(review.get("status") == "completed" for review in reviews)
    total_input = sum(review.get("usage", {}).get("input_tokens", 0) for review in reviews)
    total_output = sum(review.get("usage", {}).get("output_tokens", 0) for review in reviews)

    output = {
        "schema": "semantic-review-pool-v0",
        "task_id": request.get("task_id", "unknown"),
        "candidate_ref": request.get("candidate_ref"),
        "authority_ref": request.get("authority_ref"),
        "model_id": MODEL_ID,
        "authority": "HYPOTHESIS_ONLY",
        "production_pass_fail_authority": False,
        "automatic_retry": False,
        "review_budget": REVIEW_COUNT,
        "completed_reviews": completed,
        "wall_latency_ms": round((time.monotonic() - started) * 1000),
        "usage": {"input_tokens": total_input, "output_tokens": total_output},
        "reviews": reviews,
        "aggregated_findings": aggregated,
        "finding_count": len(aggregated),
        "result_state": "FINDINGS_OBSERVED" if aggregated else "NO_FINDINGS_OBSERVED",
        "next_gate": "deterministic_verification_required",
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({
        "schema": output["schema"],
        "authority": output["authority"],
        "completed_reviews": completed,
        "finding_count": len(aggregated),
        "wall_latency_ms": output["wall_latency_ms"],
        "usage": output["usage"],
    }, separators=(",", ":")))
    return 0 if completed > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
