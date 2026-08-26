#!/usr/bin/env python3
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

FIXTURES_PATH = Path(__file__).with_name("benchmark-fixtures.json")
OUTPUT_PATH = Path(os.environ.get("BENCHMARK_OUTPUT", "semantic-review-benchmark-results.json"))
REPEATS = int(os.environ.get("BENCHMARK_REPEATS", "3"))
MAX_INVOCATIONS = int(os.environ.get("BENCHMARK_MAX_INVOCATIONS", "48"))

MODELS = [
    {
        "name": "qwen3-coder-30b-a3b",
        "model_id": "qwen.qwen3-coder-30b-a3b-v1:0",
        "input_usd_per_m": 0.15,
        "output_usd_per_m": 0.60,
    },
    {
        "name": "qwen3-coder-next",
        "model_id": "qwen.qwen3-coder-next",
        "input_usd_per_m": 0.50,
        "output_usd_per_m": 1.20,
    },
]

CATEGORIES = {
    "SQL_INJECTION",
    "AUTH_BYPASS",
    "MISSING_AWAIT",
    "BOUNDARY_ERROR",
    "NULL_DEREFERENCE",
    "NONE",
}


def numbered(code: str) -> str:
    return "\n".join(f"{i:02d}: {line}" for i, line in enumerate(code.splitlines(), 1))


def prompt_for(fixture: dict) -> str:
    return f"""You are a code-review classifier. Review only the supplied snippet.
Return exactly one JSON object and no markdown or prose.
Schema:
{{"verdict":"BUG|CLEAN","category":"SQL_INJECTION|AUTH_BYPASS|MISSING_AWAIT|BOUNDARY_ERROR|NULL_DEREFERENCE|NONE","line":0,"confidence":0.0}}

Rules:
- Report the single most important real defect represented by the category list.
- If none of those defects is present, return verdict CLEAN, category NONE, line 0.
- Do not invent missing surrounding context.
- For BUG, line is the first line that materially causes the defect.

Language: {fixture['language']}
Code:
{numbered(fixture['code'])}
"""


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        obj = json.loads(match.group(0))
    verdict = str(obj.get("verdict", "")).upper()
    category = str(obj.get("category", "")).upper()
    if verdict not in {"BUG", "CLEAN"}:
        raise ValueError(f"invalid verdict: {verdict!r}")
    if category not in CATEGORIES:
        raise ValueError(f"invalid category: {category!r}")
    if verdict == "CLEAN":
        category = "NONE"
    return {
        "verdict": verdict,
        "category": category,
        "line": int(obj.get("line", 0) or 0),
        "confidence": float(obj.get("confidence", 0.0) or 0.0),
    }


def invoke(model: dict, fixture: dict) -> dict:
    messages = [{"role": "user", "content": [{"text": prompt_for(fixture)}]}]
    inference = {"maxTokens": 256, "temperature": 0.2, "topP": 0.9}
    cmd = [
        "aws",
        "bedrock-runtime",
        "converse",
        "--model-id",
        model["model_id"],
        "--messages",
        json.dumps(messages, separators=(",", ":")),
        "--inference-config",
        json.dumps(inference, separators=(",", ":")),
        "--output",
        "json",
        "--no-cli-pager",
    ]
    started = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
    latency_ms = round((time.monotonic() - started) * 1000)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"aws cli exit {proc.returncode}")
    response = json.loads(proc.stdout)
    content = response.get("output", {}).get("message", {}).get("content", [])
    text = "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and "text" in part).strip()
    parsed = parse_json_object(text)
    usage = response.get("usage", {})
    input_tokens = int(usage.get("inputTokens", 0) or 0)
    output_tokens = int(usage.get("outputTokens", 0) or 0)
    cost = (
        input_tokens * model["input_usd_per_m"] / 1_000_000
        + output_tokens * model["output_usd_per_m"] / 1_000_000
    )
    return {
        **parsed,
        "raw_text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
        "latency_ms": latency_ms,
    }


def summarize(model: dict, rows: list[dict], fixtures: list[dict]) -> dict:
    completed = [r for r in rows if not r.get("error")]
    fixture_by_id = {f["id"]: f for f in fixtures}
    buggy = [r for r in completed if fixture_by_id[r["fixture_id"]]["expected_verdict"] == "BUG"]
    clean = [r for r in completed if fixture_by_id[r["fixture_id"]]["expected_verdict"] == "CLEAN"]

    exact = sum(
        r["verdict"] == fixture_by_id[r["fixture_id"]]["expected_verdict"]
        and r["category"] == fixture_by_id[r["fixture_id"]]["expected_category"]
        for r in completed
    )
    correct_bug = sum(
        r["verdict"] == "BUG" and r["category"] == fixture_by_id[r["fixture_id"]]["expected_category"]
        for r in buggy
    )
    false_pos = sum(r["verdict"] == "BUG" for r in clean)

    grouped = defaultdict(list)
    for r in completed:
        grouped[r["fixture_id"]].append((r["verdict"], r["category"]))
    consistency_values = []
    for values in grouped.values():
        counts = Counter(values)
        consistency_values.append(max(counts.values()) / len(values))

    accuracy = exact / len(completed) if completed else 0.0
    recall = correct_bug / len(buggy) if buggy else 0.0
    fpr = false_pos / len(clean) if clean else 0.0
    consistency = sum(consistency_values) / len(consistency_values) if consistency_values else 0.0
    total_cost = sum(r.get("estimated_cost_usd", 0.0) for r in completed)
    cost_per_review = total_cost / len(completed) if completed else math.inf
    avg_latency = sum(r.get("latency_ms", 0) for r in completed) / len(completed) if completed else math.inf
    quality = 0.70 * recall + 0.20 * (1.0 - fpr) + 0.10 * consistency

    return {
        "name": model["name"],
        "model_id": model["model_id"],
        "completed_invocations": len(completed),
        "failed_invocations": len(rows) - len(completed),
        "accuracy": round(accuracy, 4),
        "bug_recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "consistency": round(consistency, 4),
        "quality_score": round(quality, 4),
        "avg_latency_ms": round(avg_latency, 1) if math.isfinite(avg_latency) else None,
        "estimated_total_cost_usd": round(total_cost, 8),
        "estimated_cost_per_review_usd": round(cost_per_review, 8) if math.isfinite(cost_per_review) else None,
        "eligible": len(completed) == len(fixtures) * REPEATS and recall >= 0.80 and fpr <= 0.15 and consistency >= 0.80,
    }


def choose_winner(summaries: list[dict]) -> dict:
    eligible = [s for s in summaries if s["eligible"]]
    if not eligible:
        return {"status": "NO_PROMOTION", "reason": "No candidate passed the quality gate"}
    best_quality = max(s["quality_score"] for s in eligible)
    near_best = [s for s in eligible if best_quality - s["quality_score"] <= 0.05]
    winner = min(near_best, key=lambda s: s["estimated_cost_per_review_usd"])
    return {
        "status": "CANDIDATE",
        "model": winner["name"],
        "model_id": winner["model_id"],
        "rule": "Pass quality gate; among candidates within 0.05 of best quality, choose lowest cost per review",
    }


def main() -> int:
    fixtures = json.loads(FIXTURES_PATH.read_text())
    planned = len(MODELS) * len(fixtures) * REPEATS
    if planned > MAX_INVOCATIONS:
        raise SystemExit(f"planned invocations {planned} exceed cap {MAX_INVOCATIONS}")

    all_rows = []
    model_blocked = set()
    print(f"planned_invocations={planned}; max_invocations={MAX_INVOCATIONS}; retries=disabled")

    for model in MODELS:
        for fixture in fixtures:
            for repeat in range(1, REPEATS + 1):
                if model["name"] in model_blocked:
                    continue
                base = {"model": model["name"], "model_id": model["model_id"], "fixture_id": fixture["id"], "repeat": repeat}
                try:
                    result = invoke(model, fixture)
                    row = {**base, **result}
                    print(json.dumps({k: row[k] for k in ("model", "fixture_id", "repeat", "verdict", "category", "input_tokens", "output_tokens", "latency_ms")}))
                except Exception as exc:
                    message = str(exc)
                    row = {**base, "error": message[:2000]}
                    print(json.dumps({"model": model["name"], "fixture_id": fixture["id"], "repeat": repeat, "error": message[:500]}), file=sys.stderr)
                    if "AccessDenied" in message or "not authorized" in message.lower() or "ValidationException" in message:
                        model_blocked.add(model["name"])
                        print(f"model_blocked={model['name']}; no retries or further invocations for this model", file=sys.stderr)
                all_rows.append(row)

    summaries = [summarize(model, [r for r in all_rows if r["model"] == model["name"]], fixtures) for model in MODELS]
    selection = choose_winner(summaries)
    output = {
        "benchmark": "semantic-review-benchmark-v0",
        "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        "repeats": REPEATS,
        "planned_invocations": planned,
        "actual_attempted_invocations": len(all_rows),
        "automatic_retry": False,
        "summaries": summaries,
        "selection": selection,
        "rows": all_rows,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True))
    print("SUMMARY=" + json.dumps({"summaries": summaries, "selection": selection}, separators=(",", ":")))
    return 0 if any(s["completed_invocations"] for s in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
