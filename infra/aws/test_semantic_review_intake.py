#!/usr/bin/env python3
import base64
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parent / "semantic-review" / "pool.py"
SPEC = importlib.util.spec_from_file_location("semantic_review_pool", MODULE_PATH)
assert SPEC and SPEC.loader
pool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pool)


def encoded(payload: dict, *, compressed: bool) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    packed = gzip.compress(raw, mtime=0) if compressed else raw
    return base64.b64encode(packed).decode("ascii")


def exact_size_code(size: int, path: str = "src/example.py") -> str:
    prefix = f"--- {path} (modified) ---\n"
    if size < len(prefix):
        raise ValueError("size is smaller than the required exact-context marker")
    body = ("+value = 'semantic review complete context'\n" * ((size // 44) + 2))
    return (prefix + body)[:size]


def versioned_payload(size: int) -> dict:
    path = "src/example.py"
    code = exact_size_code(size, path)
    code_bytes = code.encode("utf-8")
    return {
        "schema": pool.REQUEST_V1_SCHEMA,
        "task_id": "sem3:test",
        "language": "python",
        "requirements": "Review the exact candidate without truncation.",
        "code": code,
        "context": {
            "schema": pool.CONTEXT_V1_SCHEMA,
            "complete": True,
            "original_bytes": len(code_bytes),
            "admitted_bytes": len(code_bytes),
            "code_sha256": hashlib.sha256(code_bytes).hexdigest(),
            "changed_paths": [path],
            "omitted_files": [],
            "omitted_sections": [],
        },
    }


class SemanticReviewIntakeTests(unittest.TestCase):
    def test_legacy_small_request_remains_readable(self):
        payload = {
            "task_id": "sem3:legacy",
            "language": "python",
            "requirements": "Review this exact code.",
            "code": "print('ok')\n",
        }
        normalized, transport, budget = pool.validate_and_normalize_request(
            encoded(payload, compressed=False), 3,
        )
        self.assertEqual(transport["encoding"], "json+base64")
        self.assertEqual(normalized["code"], payload["code"])
        self.assertTrue(budget["completeness"])
        self.assertEqual(budget["admitted_context_bytes"], len(payload["code"].encode()))

    def test_current_p04_sized_context_is_admitted_complete(self):
        payload = versioned_payload(98_971)
        payload_b64 = encoded(payload, compressed=True)
        self.assertLessEqual(len(payload_b64), pool.DISPATCH_PAYLOAD_B64_MAX_CHARS)
        normalized, transport, budget = pool.validate_and_normalize_request(payload_b64, 3)
        self.assertEqual(transport["encoding"], "gzip+base64")
        self.assertEqual(normalized["code"], payload["code"])
        self.assertEqual(budget["original_context_bytes"], 98_971)
        self.assertEqual(budget["admitted_context_bytes"], 98_971)
        self.assertTrue(budget["completeness"])
        self.assertEqual(budget["omitted_files"], [])
        self.assertEqual(budget["omitted_sections"], [])

    def test_context_over_real_safe_budget_fails_before_provider(self):
        payload = versioned_payload(240_000)
        with self.assertRaises(pool.IntakeError) as raised:
            pool.validate_and_normalize_request(encoded(payload, compressed=True), 3)
        self.assertEqual(raised.exception.code, "semantic_review_context_budget_exceeded")
        evidence = raised.exception.evidence
        self.assertFalse(evidence["completeness"])
        self.assertEqual(evidence["original_context_bytes"], 240_000)
        self.assertEqual(evidence["admitted_context_bytes"], 0)
        self.assertEqual(evidence["omitted_files"], ["src/example.py"])
        self.assertEqual(evidence["omitted_sections"], ["semantic_context"])
        self.assertGreater(evidence["prompt_token_estimate"], evidence["input_budget_tokens"])

    def test_no_silent_truncation_or_omission_is_accepted(self):
        payload = versioned_payload(98_971)
        payload["context"]["admitted_bytes"] -= 1
        payload["context"]["omitted_files"] = ["src/missing.py"]
        with self.assertRaises(pool.IntakeError) as raised:
            pool.validate_and_normalize_request(encoded(payload, compressed=True), 3)
        self.assertEqual(raised.exception.code, "semantic_review_context_incomplete")

    def test_all_three_reviewers_receive_identical_complete_scope(self):
        payload = versioned_payload(98_971)
        normalized, _, budget = pool.validate_and_normalize_request(encoded(payload, compressed=True), 3)
        digest = payload["context"]["code_sha256"]
        self.assertEqual(budget["reviewer_scope_sha256"], {"A": digest, "B": digest, "C": digest})
        for reviewer_id, focus in pool.REVIEWERS:
            prompt = pool.prompt_for(normalized, reviewer_id, focus)
            self.assertIn(pool.numbered(normalized["code"]), prompt)

    def test_large_provider_prompt_uses_private_file_not_process_argv(self):
        payload = versioned_payload(116_158)
        normalized, _, _ = pool.validate_and_normalize_request(encoded(payload, compressed=True), 3)
        observed = {}

        def timeout_after_inspection(cmd, **kwargs):
            messages_index = cmd.index("--messages") + 1
            messages_ref = cmd[messages_index]
            self.assertTrue(messages_ref.startswith("file:///"))
            messages_path = Path(messages_ref.removeprefix("file://"))
            self.assertTrue(messages_path.is_file())
            self.assertEqual(messages_path.stat().st_mode & 0o777, 0o600)
            messages = json.loads(messages_path.read_text())
            self.assertIn(pool.numbered(normalized["code"]), messages[0]["content"][0]["text"])
            self.assertLess(max(len(argument.encode("utf-8")) for argument in cmd), 4_096)
            observed["messages_path"] = messages_path
            raise pool.subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        with mock.patch.object(pool.subprocess, "run", side_effect=timeout_after_inspection):
            with self.assertRaises(pool.ProducerInvocationError):
                pool.invoke_one(normalized, "A", pool.REVIEWERS[0][1])
        self.assertFalse(observed["messages_path"].exists())

    def test_review_count_is_exactly_three(self):
        payload = versioned_payload(1_000)
        for count in (0, 1, 2, 4):
            with self.subTest(count=count), self.assertRaises(pool.IntakeError) as raised:
                pool.validate_and_normalize_request(encoded(payload, compressed=True), count)
            self.assertEqual(raised.exception.code, "semantic_review_count_invalid")

    def test_caller_cannot_override_model_ref_workflow_or_retry(self):
        for key, value in (
            ("model", "other"),
            ("ref", "main"),
            ("workflow", "other.yml"),
            ("retry", True),
            ("fallback", "other"),
        ):
            payload = versioned_payload(1_000)
            payload[key] = value
            with self.subTest(key=key), self.assertRaises(pool.IntakeError) as raised:
                pool.validate_and_normalize_request(encoded(payload, compressed=True), 3)
            self.assertEqual(raised.exception.code, "semantic_review_payload_invalid")

    def test_unregistered_model_profile_fails_closed(self):
        payload = versioned_payload(1_000)
        with mock.patch.object(pool, "MODEL_ID", "qwen.caller-selected"):
            with self.assertRaises(pool.IntakeError) as raised:
                pool.validate_and_normalize_request(encoded(payload, compressed=True), 3)
        self.assertEqual(raised.exception.code, "semantic_review_model_profile_unregistered")

    def test_failure_receipt_is_written_and_input_is_not_admitted(self):
        payload = versioned_payload(240_000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            receipt_path = root / "receipt.json"
            with (
                mock.patch.object(pool, "INPUT_PATH", input_path),
                mock.patch.object(pool, "INTAKE_OUTPUT_PATH", receipt_path),
                mock.patch.dict(os.environ, {
                    "PAYLOAD_B64": encoded(payload, compressed=True),
                    "SEMANTIC_REVIEW_COUNT": "3",
                    "DRY_RUN": "false",
                }, clear=False),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(pool.validate_intake_from_env(), 2)
            receipt = json.loads(receipt_path.read_text())
            self.assertFalse(receipt["completeness"])
            self.assertFalse(receipt["provider_invocation_allowed"])
            self.assertFalse(receipt["provider_invoked"])
            self.assertEqual(receipt["error_code"], "semantic_review_context_budget_exceeded")
            self.assertFalse(input_path.exists())

    def test_workflow_keeps_fixed_lane_and_no_retry_or_fallback(self):
        workflow = (Path(__file__).parents[2] / ".github" / "workflows" / "aws-semantic-review-intake.yml").read_text()
        self.assertIn("SEMANTIC_REVIEW_MODEL_ID: qwen.qwen3-coder-30b-a3b-v1:0", workflow)
        self.assertIn("AWS_MAX_ATTEMPTS: '1'", workflow)
        self.assertIn("disable-retry: true", workflow)
        self.assertIn("python3 infra/aws/semantic-review/pool.py --validate-intake", workflow)
        self.assertNotIn("project-ai", workflow.lower())
        self.assertNotIn("coding-worker", workflow.lower())
        self.assertNotIn("fallback", workflow.lower())
        self.assertEqual(workflow.count("          - '3'"), 1)

    def test_role_trust_survives_immutable_runner_rotation_without_broad_repo_access(self):
        role = (Path(__file__).parent / "semantic-review" / "role.yml").read_text()
        repository_identity = "repo:jeonghun917@109071398/opened-arm@1339350352"
        self.assertIn("GitHubOidcSubjects:", role)
        self.assertIn("Type: CommaDelimitedList", role)
        self.assertIn("StringLike:", role)
        self.assertIn(f"{repository_identity}:ref:refs/heads/main", role)
        self.assertIn(
            f"{repository_identity}:ref:refs/tags/semantic-review-runner-*",
            role,
        )
        self.assertNotIn(f"{repository_identity}:ref:refs/heads/*", role)
        self.assertNotIn(f"{repository_identity}:ref:refs/tags/*", role)
        self.assertNotIn("repo:jeonghun917@109071398/*", role)


if __name__ == "__main__":
    unittest.main()
