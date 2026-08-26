#!/usr/bin/env python3
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
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
      "category": "short_machine_readable_category",
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


def normalize_finding(raw: dict) -> dict:
    severity = str(raw.get("severity", "medium")).lower()
    if severity not in {"high", "medium", "low"}:
        severity = "medium"
    confidence = float(raw.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    category = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("category", "unknown")).strip().lower()).strip("_") or "unknown"
    return {
        "category": category,
        "severity": severity,
        "line": max(0, int(raw.get("line", 0) or 0)),
        "title": str(raw.get("title", "")).strip()[:200],
        "rationale": str(raw.get("rationale", "")).strip()[:1200],
        "confidence": confidence,
    }


def invoke_one(request: dict, reviewer_id: str, focus: str) -> dict:
    messages = [{"role": "user", "content": [{"text": prompt_for(request, reviewer_id, focus)}]}]
    inference = {"maxTokens": 800, "temperature": 0.25, "topP": 0.9}
    cmd = [
        "aws",
        "--cli-connect-timeout", "5",
        "--cli-read-timeout", "60",
        "bedrock-runtime",
        "converse",
        "--model-id", MODEL_ID,
        "--messages", json.dumps(messages, separators=(",", ":")),
        "--inference-config", json.dumps(inference, separators=(",", ":")),
        "--output", "json",
        "--no-cli-pager",
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
    findings = [normalize_finding(x) for x in obj.get("findings", [])[:5] if isinstance(x, dict)]
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


def aggregate(reviews: list[dict]) -> list[dict]:
    grouped = {}
    for review in reviews:
        if review.get("status") != "completed":
            continue
        reviewer_id = review["reviewer_id"]
        for finding in review.get("findings", []):
            key = (finding["category"], finding["line"])
            bucket = grouped.setdefault(key, {
                "category": finding["category"],
                "line": finding["line"],
                "support_count": 0,
                "reviewer_ids": [],
                "observations": [],
            })
            if reviewer_id not in bucket["reviewer_ids"]:
                bucket["support_count"] += 1
                bucket["reviewer_ids"].append(reviewer_id)
            bucket["observations"].append({"reviewer_id": reviewer_id, **finding})

    result = list(grouped.values())
    result.sort(key=lambda x: (-x["support_count"], x["line"], x["category"]))
    return result


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

    reviews.sort(key=lambda x: x["reviewer_id"])
    aggregated = aggregate(reviews)
    completed = sum(r.get("status") == "completed" for r in reviews)
    total_input = sum(r.get("usage", {}).get("input_tokens", 0) for r in reviews)
    total_output = sum(r.get("usage", {}).get("output_tokens", 0) for r in reviews)

    output = {
        "schema": "semantic-review-pool-v0",
        "task_id": request.get("task_id", "unknown"),
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
