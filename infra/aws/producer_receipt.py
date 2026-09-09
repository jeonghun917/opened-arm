#!/usr/bin/env python3
"""Shared, pure helpers for bounded paid-producer receipt v1 envelopes."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping


RECEIPT_SCHEMA = "opened-arm-paid-producer-receipt-v1"
RECEIPT_VERSION = 1
RECEIPT_CONTEXT_KEYS = {
    "task_ref",
    "task_revision",
    "operation_id",
    "cycle_id",
    "repository",
    "base_sha",
    "target_sha",
}
IDENTITY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
REFERENCE_RE = re.compile(r"^[A-Za-z0-9._:/@+-]{1,500}$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
PROVIDER_STATES = {"FAILED", "AMBIGUOUS", "INCOMPLETE"}


class ProducerInvocationError(RuntimeError):
    """Bounded terminal provider/producer error with explicit outcome certainty."""

    def __init__(self, provider_state: str, code: str, detail: object = "") -> None:
        if provider_state not in PROVIDER_STATES:
            raise ValueError("provider_state_invalid")
        if not IDENTITY_RE.fullmatch(code):
            raise ValueError("producer_error_code_invalid")
        self.provider_state = provider_state
        self.code = code
        self.detail = (str(detail).strip() or code)[:1000]
        super().__init__(code)


def producer_error_evidence(error: object) -> dict:
    if isinstance(error, ProducerInvocationError):
        return {
            "state": error.provider_state,
            "code": error.code,
            "detail": error.detail,
        }
    return {
        "state": "INCOMPLETE",
        "code": "producer_exception",
        "detail": (str(error).strip() or error.__class__.__name__)[:1000],
    }


def _fail(code: str) -> None:
    raise ValueError(code)


def _text(value: object, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        _fail(f"{field}_must_be_string")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        _fail(f"{field}_invalid")
    return normalized


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def validate_receipt_context(
    raw: object,
    *,
    expected_repository: str | None = None,
    expected_base_sha: str | None = None,
    expected_target_sha: str | None = None,
) -> dict | None:
    """Return normalized v1 ownership context, or None for an explicit legacy input."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != RECEIPT_CONTEXT_KEYS:
        _fail("receipt_context_shape_invalid")

    task_ref = _text(raw.get("task_ref"), "receipt_task_ref", 500)
    task_revision = _text(raw.get("task_revision"), "receipt_task_revision", 500)
    if not REFERENCE_RE.fullmatch(task_ref) or not REFERENCE_RE.fullmatch(task_revision):
        _fail("receipt_task_identity_invalid")
    operation_id = _text(raw.get("operation_id"), "receipt_operation_id", 200)
    cycle_id = _text(raw.get("cycle_id"), "receipt_cycle_id", 200)
    if not IDENTITY_RE.fullmatch(operation_id) or not IDENTITY_RE.fullmatch(cycle_id):
        _fail("receipt_operation_identity_invalid")

    repository = _text(raw.get("repository"), "receipt_repository", 200)
    base_sha = _text(raw.get("base_sha"), "receipt_base_sha", 40).lower()
    target_sha = _text(raw.get("target_sha"), "receipt_target_sha", 40).lower()
    if not REPOSITORY_RE.fullmatch(repository):
        _fail("receipt_repository_invalid")
    if not SHA_RE.fullmatch(base_sha) or not SHA_RE.fullmatch(target_sha):
        _fail("receipt_sha_invalid")
    if expected_repository is not None and repository != expected_repository:
        _fail("receipt_repository_mismatch")
    if expected_base_sha is not None and base_sha != expected_base_sha.lower():
        _fail("receipt_base_sha_mismatch")
    if expected_target_sha is not None and target_sha != expected_target_sha.lower():
        _fail("receipt_target_sha_mismatch")

    return {
        "task_ref": task_ref,
        "task_revision": task_revision,
        "operation_id": operation_id,
        "cycle_id": cycle_id,
        "repository": repository,
        "base_sha": base_sha,
        "target_sha": target_sha,
    }


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def collect_workflow_identity(
    expected_workflow_paths: set[str] | frozenset[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict, list[str]]:
    """Read non-caller-controlled GitHub execution identity and report any gaps."""
    env = os.environ if environ is None else environ
    repository = env.get("GITHUB_REPOSITORY", "").strip()
    workflow_name = env.get("GITHUB_WORKFLOW", "").strip()
    workflow_ref = env.get("GITHUB_WORKFLOW_REF", "").strip()
    execution_ref = env.get("GITHUB_REF", "").strip()
    source_sha = env.get("GITHUB_SHA", "").strip().lower()
    run_id = _positive_int(env.get("GITHUB_RUN_ID"))
    run_attempt = _positive_int(env.get("GITHUB_RUN_ATTEMPT"))
    job = env.get("GITHUB_JOB", "").strip()
    issues: list[str] = []

    workflow_path = ""
    workflow_source_ref = ""
    prefix = f"{repository}/" if repository else ""
    if prefix and workflow_ref.startswith(prefix) and "@" in workflow_ref[len(prefix):]:
        workflow_path, workflow_source_ref = workflow_ref[len(prefix):].rsplit("@", 1)

    if repository != "jeonghun917/opened-arm":
        issues.append("workflow_repository_unverified")
    if not workflow_name:
        issues.append("workflow_name_missing")
    if not workflow_ref or not workflow_path:
        issues.append("workflow_ref_invalid")
    elif workflow_path not in expected_workflow_paths:
        issues.append("workflow_path_not_admitted")
    if not execution_ref:
        issues.append("execution_ref_missing")
    elif workflow_source_ref != execution_ref:
        issues.append("workflow_execution_ref_mismatch")
    if not SHA_RE.fullmatch(source_sha):
        issues.append("producer_source_sha_invalid")
    if run_id is None:
        issues.append("github_run_id_invalid")
    if run_attempt is None:
        issues.append("github_run_attempt_invalid")
    if not job:
        issues.append("github_job_missing")

    return {
        "repository": repository or None,
        "name": workflow_name or None,
        "path": workflow_path or None,
        "source_ref": workflow_source_ref or None,
        "workflow_ref": workflow_ref or None,
        "execution_ref": execution_ref or None,
        "job": job or None,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "producer_source_sha": source_sha if SHA_RE.fullmatch(source_sha) else None,
    }, issues


def new_receipt(
    *,
    context: dict | None,
    input_bytes: bytes,
    producer_kind: str,
    producer_id: str,
    authority: str,
    model_id: str,
    expected_workflow_paths: set[str] | frozenset[str],
    environ: Mapping[str, str] | None = None,
) -> tuple[dict | None, list[str]]:
    """Create the immutable identity portion of a receipt; legacy input returns None."""
    if context is None:
        return None, []
    workflow, issues = collect_workflow_identity(expected_workflow_paths, environ=environ)
    return {
        "schema": RECEIPT_SCHEMA,
        "version": RECEIPT_VERSION,
        "producer_kind": producer_kind,
        "authority": authority,
        "task": {
            "ref": context["task_ref"],
            "revision": context["task_revision"],
        },
        "operation": {
            "id": context["operation_id"],
            "cycle_id": context["cycle_id"],
        },
        "input": {
            "digest_algorithm": "sha256",
            "digest": sha256_bytes(input_bytes),
            "byte_count": len(input_bytes),
        },
        "target": {
            "repository": context["repository"],
            "base_sha": context["base_sha"],
            "target_sha": context["target_sha"],
        },
        "producer": {
            "id": producer_id,
            "model_id": model_id,
            "source_sha": workflow["producer_source_sha"],
        },
        "workflow": {key: value for key, value in workflow.items() if key != "producer_source_sha"},
        "automatic_retry": False,
    }, issues


def finalize_receipt(
    receipt: dict,
    *,
    identity_issues: list[str],
    output: dict,
    raw_output_complete: bool,
    normalized_output_complete: bool,
    truncated: bool,
    overflow: bool,
    provider_failed: bool,
    issues: list[str],
) -> dict:
    all_issues = list(dict.fromkeys(identity_issues + issues))[:50]
    complete = (
        not all_issues
        and raw_output_complete
        and normalized_output_complete
        and not truncated
        and not overflow
        and not provider_failed
    )
    state = "FAILED" if provider_failed else ("SUCCEEDED" if complete else "INCOMPLETE")
    return {
        **receipt,
        "output": output,
        "completeness": {
            "complete": complete,
            "raw_output_complete": raw_output_complete,
            "normalized_output_complete": normalized_output_complete,
            "truncated": truncated,
            "overflow": overflow,
            "issues": all_issues,
        },
        "final": {
            "state": state,
            "success": complete,
        },
    }
