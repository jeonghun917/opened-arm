#!/usr/bin/env python3
"""Pure regression tests for P04 paid-producer receipts and completeness."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path


AWS_DIR = Path(__file__).resolve().parent
ROOT = AWS_DIR.parents[1]
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

from producer_receipt import (  # noqa: E402
    ProducerInvocationError,
    RECEIPT_SCHEMA,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reasoner = load_module("p04_reasoner", AWS_DIR / "coding-worker" / "reasoner.py")
pool = load_module("p04_pool", AWS_DIR / "semantic-review" / "pool.py")

BASE_SHA = "a" * 40
TARGET_SHA = "b" * 40
PRODUCER_SHA = "c" * 40
TASK_REF = "continuity-task:v0:meta:deterministic-execution-loop-v0"
TASK_REVISION = "deterministic-execution-loop-v0-p04-cleanup-review-retirement-next-r51"


def receipt_context(target_sha: str, operation_id: str = "operation-p04") -> dict:
    return {
        "task_ref": TASK_REF,
        "task_revision": TASK_REVISION,
        "operation_id": operation_id,
        "cycle_id": "cycle-p04",
        "repository": "example/repo",
        "base_sha": BASE_SHA,
        "target_sha": target_sha,
    }


def github_env(workflow_path: str, job: str) -> dict[str, str]:
    return {
        "GITHUB_REPOSITORY": "jeonghun917/opened-arm",
        "GITHUB_WORKFLOW": "P04 fixture",
        "GITHUB_WORKFLOW_REF": f"jeonghun917/opened-arm/{workflow_path}@refs/heads/feat/p04",
        "GITHUB_REF": "refs/heads/feat/p04",
        "GITHUB_SHA": PRODUCER_SHA,
        "GITHUB_RUN_ID": "34199999999",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_JOB": job,
    }


def raw_coding_request(*, context: dict | None, max_mutations: int = 2, max_total_bytes: int = 4096) -> dict:
    value = {
        "task_id": "p04-coding",
        "goal": "Update one file",
        "acceptance_criteria": ["Only src/a.ts changes"],
        "stop_and_escalate": ["Escalate outside scope"],
        "requested_change": "Change 1 to 2",
        "target": {"repository": "example/repo", "branch": "feat/p04", "base_sha": BASE_SHA},
        "path_policy": {
            "allowed_paths": ["src/**"],
            "forbidden_paths": [],
            "allow_delete": False,
            "max_mutations": max_mutations,
            "max_file_bytes": 131072,
            "max_total_bytes": max_total_bytes,
        },
        "files": [{"path": "src/a.ts", "content": "export const a = 1;\n"}],
    }
    if context is not None:
        value["receipt_context"] = context
    return value


def coding_invocation(request: dict, *, stop_reason: str = "end_turn") -> dict:
    raw_value = {
        "decision": "PROPOSE",
        "summary": "bounded",
        "mutations": [{
            "mutation_id": "update-a",
            "path": "src/a.ts",
            "operation": "update_file",
            "content": "export const a = 2;\n",
            "rationale": "requested",
        }],
        "assumptions": [],
        "unresolved": [],
    }
    raw_text = json.dumps(raw_value, separators=(",", ":"))
    normalized = reasoner.normalize_model_output(raw_text, request, stop_reason, strict=True)
    content = [{"text": raw_text}]
    return {
        "proposal": normalized["proposal"],
        "normalization_complete": normalized["complete"],
        "truncated": normalized["truncated"],
        "overflow": normalized["overflow"],
        "issues": normalized["issues"],
        "raw_content": content,
        "raw_text": raw_text,
        "raw_content_complete": True,
        "finish_reason": stop_reason,
        "usage": reasoner.usage_evidence({"inputTokens": 40, "outputTokens": 20, "totalTokens": 60}),
        "latency_ms": 123,
        "output_profile": reasoner.output_profile_for(request),
    }


def finding(index: int = 0) -> dict:
    return {
        "category": "correctness",
        "severity": "medium",
        "line": index + 1,
        "title": f"Finding {index}",
        "rationale": f"Grounded rationale {index}",
        "confidence": 0.75,
    }


def semantic_request(*, context: dict | None) -> dict:
    value = {
        "task_id": "p04-semantic",
        "language": "python",
        "requirements": "Review exact candidate without mutation.",
        "code": "0001: value = 1\n",
    }
    if context is not None:
        value["receipt_context"] = context
    return pool.validate_request(value)


def semantic_review(reviewer_id: str, findings: list[dict] | None = None, *, stop_reason: str = "end_turn") -> dict:
    raw = {"reviewer_id": reviewer_id, "findings": [] if findings is None else findings}
    result = pool.normalize_reviewer_response(
        reviewer_id,
        [{"text": json.dumps(raw, separators=(",", ":"))}],
        stop_reason,
        {"inputTokens": 30, "outputTokens": 10, "totalTokens": 40},
        strict=True,
    )
    result["latency_ms"] = 100
    result["output_profile"] = dict(pool.SEMANTIC_OUTPUT_PROFILE)
    return result


class ReceiptIdentityTests(unittest.TestCase):
    def test_new_coding_receipt_binds_exact_identity_and_digests(self):
        request = reasoner.validate_request(raw_coding_request(context=receipt_context(BASE_SHA)))
        input_bytes = canonical_json_bytes(request)
        result = reasoner.build_result_document(
            request,
            input_bytes,
            coding_invocation(request),
            environ=github_env(".github/workflows/aws-qwen-coding-reasoner.yml", "reason"),
        )
        receipt = result["producer_receipt"]
        self.assertEqual(result["schema"], "qwen-coding-reasoner-v0")
        self.assertEqual(receipt["schema"], RECEIPT_SCHEMA)
        self.assertEqual(receipt["version"], 1)
        self.assertEqual(receipt["task"], {"ref": TASK_REF, "revision": TASK_REVISION})
        self.assertEqual(receipt["operation"], {"id": "operation-p04", "cycle_id": "cycle-p04"})
        self.assertEqual(receipt["target"]["base_sha"], BASE_SHA)
        self.assertEqual(receipt["target"]["target_sha"], BASE_SHA)
        self.assertEqual(receipt["producer"]["source_sha"], PRODUCER_SHA)
        self.assertEqual(receipt["workflow"]["run_id"], 34199999999)
        self.assertEqual(receipt["workflow"]["run_attempt"], 1)
        self.assertEqual(receipt["workflow"]["source_ref"], "refs/heads/feat/p04")
        self.assertEqual(receipt["input"]["digest"], sha256_bytes(input_bytes))
        output = receipt["output"]
        output_core = {key: value for key, value in output.items() if key not in {"digest_algorithm", "digest"}}
        self.assertEqual(output["digest"], sha256_json(output_core))
        self.assertEqual(receipt["final"], {"state": "SUCCEEDED", "success": True})
        self.assertFalse(receipt["automatic_retry"])

    def test_workflow_identity_drift_cannot_issue_complete_receipt(self):
        request = reasoner.validate_request(raw_coding_request(context=receipt_context(BASE_SHA)))
        environ = github_env(".github/workflows/aws-qwen-coding-reasoner.yml", "reason")
        environ["GITHUB_REF"] = "refs/heads/other"
        result = reasoner.build_result_document(
            request,
            canonical_json_bytes(request),
            coding_invocation(request),
            environ=environ,
        )
        receipt = result["producer_receipt"]
        self.assertFalse(receipt["completeness"]["complete"])
        self.assertIn("workflow_execution_ref_mismatch", receipt["completeness"]["issues"])
        self.assertEqual(result["next_gate"], "producer_receipt_incomplete_stop")

    def test_legacy_coding_result_is_readable_but_not_upgraded(self):
        request = reasoner.validate_request(raw_coding_request(context=None))
        result = reasoner.build_result_document(request, canonical_json_bytes(request), coding_invocation(request))
        self.assertEqual(result["schema"], "qwen-coding-reasoner-v0")
        self.assertEqual(result["proposal"]["decision"], "PROPOSE")
        self.assertNotIn("producer_receipt", result)
        self.assertNotIn("result_state", result)

    def test_receipt_target_mismatch_fails_before_paid_execution(self):
        bad = receipt_context(TARGET_SHA)
        with self.assertRaisesRegex(ValueError, "receipt_target_sha_mismatch"):
            reasoner.validate_request(raw_coding_request(context=bad))

    def test_only_standalone_producer_workflows_can_issue_new_receipts(self):
        self.assertEqual(
            reasoner.CODING_WORKFLOW_PATHS,
            {".github/workflows/aws-qwen-coding-reasoner.yml"},
        )
        self.assertEqual(
            pool.SEMANTIC_WORKFLOW_PATHS,
            {".github/workflows/aws-semantic-review-intake.yml"},
        )

        coding_request = reasoner.validate_request(raw_coding_request(context=receipt_context(BASE_SHA)))
        coding_result = reasoner.build_result_document(
            coding_request,
            canonical_json_bytes(coding_request),
            coding_invocation(coding_request),
            environ=github_env(".github/workflows/aws-project-ai-bundle.yml", "coding"),
        )
        review_request = semantic_request(context=receipt_context(TARGET_SHA, "operation-retired-review"))
        review_result = pool.build_pool_result(
            review_request,
            canonical_json_bytes(review_request),
            [semantic_review(reviewer_id) for reviewer_id in "ABC"],
            10,
            environ=github_env(".github/workflows/aws-project-ai-bundle.yml", "semantic_review"),
        )
        for result in (coding_result, review_result):
            receipt = result["producer_receipt"]
            self.assertFalse(receipt["completeness"]["complete"])
            self.assertIn("workflow_path_not_admitted", receipt["completeness"]["issues"])

    def test_project_ai_bundle_dispatch_is_absent(self):
        workflow_dir = ROOT / ".github" / "workflows"
        self.assertFalse((workflow_dir / "aws-project-ai-bundle.yml").exists())
        coding_workflow = (workflow_dir / "aws-qwen-coding-reasoner.yml").read_text()
        semantic_workflow = (workflow_dir / "aws-semantic-review-intake.yml").read_text()
        self.assertNotIn("infra/aws/semantic-review/pool.py", coding_workflow)
        self.assertNotIn("infra/aws/coding-worker/reasoner.py", semantic_workflow)
        self.assertNotIn("aws-project-ai-bundle", coding_workflow + semantic_workflow)
        policy = json.loads((ROOT / "ops" / "project-ai-capacity-policy.json").read_text())
        self.assertNotIn("bundleWorkflow", policy["execution"])
        self.assertEqual(policy["execution"]["semanticReviewCount"], 3)
        self.assertEqual(policy["execution"]["semanticReviewCodingCalls"], 0)
        self.assertEqual(policy["execution"]["codingReasonerSemanticCalls"], 0)
        self.assertEqual(policy["execution"]["adjudication"], "SEPARATE_SOURCE_GROUNDED")


class CodingCompletenessTests(unittest.TestCase):
    def test_server_profiles_are_finite_and_bound_to_operation_budget(self):
        compact = reasoner.validate_request(raw_coding_request(context=receipt_context(BASE_SHA)))
        standard = reasoner.validate_request(raw_coding_request(
            context=receipt_context(BASE_SHA), max_mutations=12, max_total_bytes=131072,
        ))
        extended = reasoner.validate_request(raw_coding_request(
            context=receipt_context(BASE_SHA), max_mutations=50, max_total_bytes=1048576,
        ))
        profiles = [reasoner.output_profile_for(value) for value in (compact, standard, extended)]
        self.assertEqual([p["max_output_tokens"] for p in profiles], [1800, 4096, 8192])
        for request, profile in zip((compact, standard, extended), profiles):
            self.assertEqual(profile["model_id"], "qwen.qwen3-coder-30b-a3b-v1:0")
            self.assertEqual(profile["operation_budget"]["max_mutations"], request["path_policy"]["max_mutations"])
            self.assertFalse(profile["arbitrary_multi_file_fit_guaranteed"])

    def test_caller_model_temperature_and_token_overrides_are_rejected(self):
        for key, value in (("model_id", "other"), ("temperature", 1), ("max_tokens", 99999)):
            raw = raw_coding_request(context=receipt_context(BASE_SHA))
            raw[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "request_shape_invalid"):
                reasoner.validate_request(raw)
        self.assertEqual(reasoner.MODEL_ID, "qwen.qwen3-coder-30b-a3b-v1:0")

    def test_truncated_coding_output_is_incomplete(self):
        request = reasoner.validate_request(raw_coding_request(context=receipt_context(BASE_SHA)))
        invocation = coding_invocation(request, stop_reason="max_tokens")
        result = reasoner.build_result_document(
            request,
            canonical_json_bytes(request),
            invocation,
            environ=github_env(".github/workflows/aws-qwen-coding-reasoner.yml", "reason"),
        )
        receipt = result["producer_receipt"]
        self.assertEqual(result["proposal"]["decision"], "ESCALATE")
        self.assertTrue(receipt["completeness"]["truncated"])
        self.assertFalse(receipt["completeness"]["complete"])
        self.assertEqual(receipt["final"]["state"], "INCOMPLETE")
        self.assertEqual(result["next_gate"], "producer_receipt_incomplete_stop")

    def test_usage_finish_state_and_complete_raw_output_are_recorded(self):
        request = reasoner.validate_request(raw_coding_request(context=receipt_context(BASE_SHA)))
        result = reasoner.build_result_document(
            request,
            canonical_json_bytes(request),
            coding_invocation(request),
            environ=github_env(".github/workflows/aws-qwen-coding-reasoner.yml", "reason"),
        )
        output = result["producer_receipt"]["output"]
        self.assertEqual(output["finish_reason"], "end_turn")
        self.assertEqual(output["usage"]["total_tokens"], 60)
        self.assertTrue(output["usage"]["available"])
        self.assertTrue(output["raw_model_output"]["text"])
        self.assertEqual(
            output["normalized_output"]["derived_from_raw_digest"],
            output["raw_model_output"]["digest"],
        )

    def test_provider_failure_writes_terminal_failure_receipt(self):
        request = reasoner.validate_request(raw_coding_request(context=receipt_context(BASE_SHA)))
        result = reasoner.build_result_document(
            request,
            canonical_json_bytes(request),
            provider_error=ProducerInvocationError("FAILED", "provider_cli_failed", "provider unavailable"),
            environ=github_env(".github/workflows/aws-qwen-coding-reasoner.yml", "reason"),
        )
        receipt = result["producer_receipt"]
        self.assertEqual(receipt["final"], {"state": "FAILED", "success": False})
        self.assertIn("provider_cli_failed", receipt["completeness"]["issues"])
        self.assertFalse(result["automatic_retry"])
        self.assertEqual(result["next_gate"], "producer_receipt_incomplete_stop")

    def test_coding_output_overflow_is_explicit_and_incomplete(self):
        request = reasoner.validate_request(raw_coding_request(
            context=receipt_context(BASE_SHA), max_mutations=2,
        ))
        raw_value = {
            "decision": "PROPOSE",
            "summary": "too many mutations",
            "mutations": [
                {
                    "mutation_id": f"update-{index}",
                    "path": f"src/{index}.ts",
                    "operation": "update_file",
                    "content": f"export const value = {index};\n",
                    "rationale": "requested",
                }
                for index in range(3)
            ],
            "assumptions": [],
            "unresolved": [],
        }
        raw_text = json.dumps(raw_value, separators=(",", ":"))
        normalized = reasoner.normalize_model_output(raw_text, request, "end_turn", strict=True)
        invocation = {
            "proposal": normalized["proposal"],
            "normalization_complete": normalized["complete"],
            "truncated": normalized["truncated"],
            "overflow": normalized["overflow"],
            "issues": normalized["issues"],
            "raw_content": [{"text": raw_text}],
            "raw_text": raw_text,
            "raw_content_complete": True,
            "finish_reason": "end_turn",
            "usage": reasoner.usage_evidence({"inputTokens": 40, "outputTokens": 20}),
            "latency_ms": 123,
            "output_profile": reasoner.output_profile_for(request),
        }
        result = reasoner.build_result_document(
            request,
            canonical_json_bytes(request),
            invocation,
            environ=github_env(".github/workflows/aws-qwen-coding-reasoner.yml", "reason"),
        )
        receipt = result["producer_receipt"]
        self.assertTrue(receipt["completeness"]["overflow"])
        self.assertFalse(receipt["completeness"]["complete"])
        self.assertEqual(receipt["final"], {"state": "INCOMPLETE", "success": False})
        self.assertEqual(result["next_gate"], "producer_receipt_incomplete_stop")

    def test_ambiguous_provider_state_is_incomplete_and_not_success(self):
        request = reasoner.validate_request(raw_coding_request(context=receipt_context(BASE_SHA)))
        result = reasoner.build_result_document(
            request,
            canonical_json_bytes(request),
            provider_error=ProducerInvocationError(
                "AMBIGUOUS", "provider_response_timeout", "terminal response not observed",
            ),
            environ=github_env(".github/workflows/aws-qwen-coding-reasoner.yml", "reason"),
        )
        receipt = result["producer_receipt"]
        self.assertEqual(receipt["output"]["provider_state"], "AMBIGUOUS")
        self.assertEqual(receipt["final"], {"state": "INCOMPLETE", "success": False})
        self.assertEqual(result["next_gate"], "producer_receipt_incomplete_stop")


class SemanticCompletenessTests(unittest.TestCase):
    def test_exact_three_reviewers_and_complete_v1_receipt(self):
        self.assertEqual(pool.REVIEW_COUNT, 3)
        self.assertEqual([item[0] for item in pool.REVIEWERS], ["A", "B", "C"])
        request = semantic_request(context=receipt_context(TARGET_SHA, "operation-semantic"))
        input_bytes = canonical_json_bytes(request)
        reviews = [semantic_review(reviewer_id, [finding(i)]) for i, reviewer_id in enumerate("ABC")]
        result = pool.build_pool_result(
            request,
            input_bytes,
            reviews,
            300,
            environ=github_env(".github/workflows/aws-semantic-review-intake.yml", "review"),
        )
        receipt = result["producer_receipt"]
        self.assertEqual(result["completed_reviews"], 3)
        self.assertEqual(result["coding_calls"], 0)
        self.assertEqual(result["paid_adjudicator_calls"], 0)
        self.assertEqual(receipt["authority"], "HYPOTHESIS_ONLY")
        self.assertTrue(receipt["completeness"]["complete"])
        self.assertEqual(receipt["target"]["target_sha"], TARGET_SHA)
        self.assertEqual([r["reviewer_id"] for r in receipt["output"]["reviewers"]], ["A", "B", "C"])
        self.assertTrue(all(r["raw_model_output"]["text"] for r in reviews))
        self.assertFalse(receipt["output"]["majority_vote_is_truth"])
        self.assertEqual(receipt["output"]["adjudication"], "SEPARATE_SOURCE_GROUNDED_REQUIRED")
        semantic_output = receipt["output"]
        semantic_output_core = {
            key: value for key, value in semantic_output.items()
            if key not in {"digest_algorithm", "digest"}
        }
        self.assertEqual(semantic_output["digest"], sha256_json(semantic_output_core))
        for review in reviews:
            self.assertEqual(review["provider_state"], "RETURNED")
            self.assertEqual(review["finish_reason"], "end_turn")
            self.assertTrue(review["usage"]["available"])
            self.assertEqual(
                review["normalized_output"]["derived_from_raw_digest"],
                review["raw_model_output"]["digest"],
            )

    def test_missing_reviewer_raw_output_stops_completeness(self):
        request = semantic_request(context=receipt_context(TARGET_SHA))
        missing = pool.normalize_reviewer_response(
            "B", [], "end_turn", {"inputTokens": 1, "outputTokens": 0}, strict=True,
        )
        missing["latency_ms"] = 1
        missing["output_profile"] = dict(pool.SEMANTIC_OUTPUT_PROFILE)
        reviews = [semantic_review("A"), missing, semantic_review("C")]
        result = pool.build_pool_result(
            request, canonical_json_bytes(request), reviews, 10,
            environ=github_env(".github/workflows/aws-semantic-review-intake.yml", "review"),
        )
        self.assertEqual(result["completed_reviews"], 2)
        self.assertEqual(result["result_state"], "INCOMPLETE")
        self.assertFalse(result["producer_receipt"]["completeness"]["raw_output_complete"])

    def test_invalid_and_truncated_raw_outputs_are_incomplete(self):
        invalid = pool.normalize_reviewer_response(
            "A", [{"text": "not json"}], "end_turn",
            {"inputTokens": 1, "outputTokens": 1}, strict=True,
        )
        truncated = pool.normalize_reviewer_response(
            "B", [{"text": '{"reviewer_id":"B","findings":[]}'}], "max_tokens",
            {"inputTokens": 1, "outputTokens": 1}, strict=True,
        )
        self.assertEqual(invalid["status"], "incomplete")
        self.assertIn("not json", invalid["raw_model_output"]["text"])
        self.assertEqual(truncated["status"], "incomplete")
        self.assertTrue(truncated["normalization"]["truncated"])

    def test_finding_overflow_preserves_raw_and_reports_omission(self):
        raw_findings = [finding(i) for i in range(6)]
        review = semantic_review("A", raw_findings)
        self.assertEqual(review["status"], "incomplete")
        self.assertTrue(review["normalization"]["overflow"])
        self.assertEqual(review["normalization"]["raw_finding_count"], 6)
        self.assertEqual(review["normalization"]["normalized_finding_count"], 5)
        self.assertEqual(review["normalization"]["omitted_finding_count"], 1)
        preserved = json.loads(review["raw_model_output"]["text"])
        self.assertEqual(len(preserved["findings"]), 6)

        request = semantic_request(context=receipt_context(TARGET_SHA))
        reviews = [review, semantic_review("B"), semantic_review("C")]
        result = pool.build_pool_result(
            request, canonical_json_bytes(request), reviews, 10,
            environ=github_env(".github/workflows/aws-semantic-review-intake.yml", "review"),
        )
        self.assertEqual(result["result_state"], "INCOMPLETE")
        self.assertTrue(result["producer_receipt"]["completeness"]["overflow"])
        self.assertEqual(result["finding_count"], 0)

    def test_semantic_provider_failure_is_terminal_and_not_retried(self):
        request = semantic_request(context=receipt_context(TARGET_SHA))
        reviews = [
            semantic_review("A"),
            pool.failed_review(
                "B", ProducerInvocationError("FAILED", "provider_cli_failed", "provider failed")
            ),
            semantic_review("C"),
        ]
        result = pool.build_pool_result(
            request, canonical_json_bytes(request), reviews, 10,
            environ=github_env(".github/workflows/aws-semantic-review-intake.yml", "review"),
        )
        receipt = result["producer_receipt"]
        self.assertEqual(receipt["final"], {"state": "FAILED", "success": False})
        self.assertFalse(result["automatic_retry"])
        self.assertFalse(receipt["automatic_retry"])

    def test_semantic_ambiguous_provider_state_is_incomplete(self):
        request = semantic_request(context=receipt_context(TARGET_SHA))
        reviews = [
            semantic_review("A"),
            pool.failed_review(
                "B", ProducerInvocationError("AMBIGUOUS", "provider_response_timeout", "timed out")
            ),
            semantic_review("C"),
        ]
        result = pool.build_pool_result(
            request, canonical_json_bytes(request), reviews, 10,
            environ=github_env(".github/workflows/aws-semantic-review-intake.yml", "review"),
        )
        receipt = result["producer_receipt"]
        self.assertEqual(receipt["final"], {"state": "INCOMPLETE", "success": False})
        self.assertEqual(receipt["output"]["reviewers"][1]["provider_state"], "AMBIGUOUS")
        self.assertEqual(result["next_gate"], "producer_receipt_incomplete_stop")

    def test_legacy_semantic_result_is_not_promoted_to_v1(self):
        request = semantic_request(context=None)
        reviews = [semantic_review(reviewer_id) for reviewer_id in "ABC"]
        result = pool.build_pool_result(request, canonical_json_bytes(request), reviews, 10)
        self.assertEqual(result["schema"], "semantic-review-pool-v0")
        self.assertEqual(result["completed_reviews"], 3)
        self.assertNotIn("producer_receipt", result)

    def test_semantic_caller_cannot_override_model_temperature_or_token_limit(self):
        for key, value in (("model_id", "other"), ("temperature", 1), ("max_tokens", 99999)):
            raw = {
                "task_id": "p04-semantic",
                "language": "python",
                "requirements": "Review exact candidate without mutation.",
                "code": "value = 1\n",
                "receipt_context": receipt_context(TARGET_SHA),
                key: value,
            }
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "review_request_shape_invalid"):
                pool.validate_request(raw)

    def test_legacy_two_reviewer_mode_remains_available_but_cannot_issue_v1(self):
        previous = os.environ.get("SEMANTIC_REVIEW_COUNT")
        os.environ["SEMANTIC_REVIEW_COUNT"] = "2"
        try:
            legacy_pool = load_module("p04_pool_legacy_two", AWS_DIR / "semantic-review" / "pool.py")
            self.assertEqual(legacy_pool.REVIEW_COUNT, 2)
            legacy_pool.validate_request({
                "task_id": "legacy-two",
                "language": "python",
                "requirements": "Legacy five-review split compatibility.",
                "code": "value = 1\n",
            })
            with self.assertRaisesRegex(ValueError, "receipt_v1_requires_exactly_three_reviewers"):
                legacy_pool.validate_request({
                    "task_id": "v1-two-forbidden",
                    "language": "python",
                    "requirements": "Must fail closed.",
                    "code": "value = 1\n",
                    "receipt_context": receipt_context(TARGET_SHA),
                })
        finally:
            if previous is None:
                os.environ.pop("SEMANTIC_REVIEW_COUNT", None)
            else:
                os.environ["SEMANTIC_REVIEW_COUNT"] = previous


if __name__ == "__main__":
    unittest.main(verbosity=2)
