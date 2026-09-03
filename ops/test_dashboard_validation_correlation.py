#!/usr/bin/env python3
"""Static and executable contract checks for validator dispatch correlation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dashboard-current-validation.yml"
SOURCE = WORKFLOW.read_text(encoding="utf-8")
SHA = "a" * 40
OTHER_SHA = "b" * 40
CORRELATION_A = "promotion:action-149:candidate"
CORRELATION_B = "promotion:action-149:postmerge"
RUN_NAME_PREFIX = "dashboard-current-validation"
REQUIRED_EVIDENCE = {
    "schema",
    "targetSha",
    "actualSha",
    "targetResolution",
    "credentialGate",
    "pendingStatusPublish",
    "privateCheckout",
    "identityCheck",
    "npmCi",
    "npmRunCheck",
    "finalStatusPublish",
    "statusContext",
    "runId",
    "runAttempt",
}


def require(pattern: str, message: str, flags: int = 0) -> re.Match[str]:
    match = re.search(pattern, SOURCE, flags)
    if match is None:
        raise AssertionError(message)
    return match


def render_run_name(sha: str | None, correlation: str | None) -> str:
    return (
        f"{RUN_NAME_PREFIX}|sha={sha or 'target-file'}"
        f"|correlation={correlation or '-'}"
    )


def evidence_script() -> str:
    blocks = re.findall(r"^\s{10}python - <<'PY'\n(.*?)^\s{10}PY$", SOURCE, re.MULTILINE | re.DOTALL)
    if len(blocks) != 1:
        raise AssertionError("expected one evidence-generation Python heredoc")
    return textwrap.dedent(blocks[0])


def correlation_validation_script() -> str:
    blocks = re.findall(
        r'^\s{10}python - "\$correlation_id" <<\'PY\'\n(.*?)^\s{10}PY$',
        SOURCE,
        re.MULTILINE | re.DOTALL,
    )
    if len(blocks) != 1:
        raise AssertionError("expected one correlation-validation Python heredoc")
    return textwrap.dedent(blocks[0])


def correlation_is_valid(value: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", correlation_validation_script(), value],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def generate_evidence(correlation: str) -> tuple[dict[str, object], dict[str, object]]:
    env = {
        **os.environ,
        "TARGET_SHA": SHA,
        "ACTUAL_SHA": SHA,
        "TARGET_STATUS": "success",
        "CREDENTIAL_STATUS": "success",
        "PENDING_STATUS": "success",
        "CHECKOUT_STATUS": "success",
        "IDENTITY_STATUS": "success",
        "NPM_CI_STATUS": "success",
        "NPM_CHECK_STATUS": "success",
        "FINAL_STATUS": "success",
        "CORRELATION_ID": correlation,
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    with tempfile.TemporaryDirectory() as checkout, tempfile.TemporaryDirectory() as runner_temp:
        checkout_path = Path(checkout)
        (checkout_path / "ops").mkdir()
        env["RUNNER_TEMP"] = runner_temp
        subprocess.run(
            [sys.executable, "-c", evidence_script()],
            cwd=checkout_path,
            env=env,
            check=True,
        )
        committed = json.loads((checkout_path / "ops/dashboard-current-validation.json").read_text(encoding="utf-8"))
        artifact = json.loads((Path(runner_temp) / "dashboard-current-validation.json").read_text(encoding="utf-8"))
        return committed, artifact


def main() -> None:
    require(
        r"(?ms)^\s{6}sha:\n\s{8}description: Exact dashboard-control-center commit SHA to validate\n\s{8}required: true$",
        "legacy required sha input changed",
    )
    require(
        r"(?ms)^\s{6}correlation_id:\n\s{8}description: Optional opaque dispatch correlation identity\n"
        r"\s{8}required: false\n\s{8}type: string$",
        "optional string correlation_id input is missing",
    )

    run_name = require(r"(?m)^run-name: (.+)$", "run-name is missing").group(1)
    assert run_name == (
        "dashboard-current-validation|sha=${{ github.event_name == 'workflow_dispatch' && inputs.sha || 'target-file' }}"
        "|correlation=${{ github.event_name == 'workflow_dispatch' && inputs.correlation_id || '-' }}"
    )
    assert render_run_name(SHA, CORRELATION_A) != render_run_name(SHA, CORRELATION_B)
    assert render_run_name(SHA, None) not in {
        render_run_name(SHA, CORRELATION_A),
        render_run_name(SHA, CORRELATION_B),
    }

    correlation_pattern = require(
        r're\.fullmatch\(r"([^"]+)", value\)',
        "correlation validation pattern is missing",
    ).group(1)
    correlation_re = re.compile(rf"^(?:{correlation_pattern})$")
    for value in (
        "",
        "a",
        "A-1",
        "promotion:149:candidate",
        "action.id_with-dots:phase-1",
        "z" * 128,
    ):
        assert value == "" or correlation_re.fullmatch(value), value
        assert correlation_is_valid(value), value
    for value in (
        "-starts-with-punctuation",
        "_starts-with-punctuation",
        "contains space",
        "contains/slash",
        "contains|delimiter",
        "contains\nnewline",
        "contains\"quote",
        "z" * 129,
    ):
        assert correlation_re.fullmatch(value) is None, repr(value)
        assert not correlation_is_valid(value), repr(value)

    require(r'test "\$actual" = "\$TARGET_SHA"', "TARGET/ACTUAL exact equality check changed")
    assert SOURCE.count('\\"context\\":\\"dashboard-public-validation\\"') == 2
    require(
        r"name: dashboard-current-validation-evidence-\$\{\{ github\.run_id \}\}-\$\{\{ github\.run_attempt \}\}",
        "per-run evidence artifact name is missing",
    )
    require(r"path: \$\{\{ runner\.temp \}\}/dashboard-current-validation\.json", "per-run artifact path is missing")
    require(r"if-no-files-found: error", "missing evidence artifact must fail closed")

    correlated, artifact = generate_evidence(CORRELATION_A)
    assert correlated == artifact
    assert correlated["schema"] == 2
    assert REQUIRED_EVIDENCE <= correlated.keys()
    assert correlated["targetSha"] == SHA
    assert correlated["actualSha"] == SHA
    assert correlated["statusContext"] == "dashboard-public-validation"
    assert correlated["runId"] == "123456789"
    assert correlated["runAttempt"] == "1"
    assert correlated["correlationId"] == CORRELATION_A
    for key in REQUIRED_EVIDENCE - {"schema", "targetSha", "actualSha", "statusContext", "runId", "runAttempt"}:
        assert correlated[key] == "success", key

    legacy, legacy_artifact = generate_evidence("")
    assert legacy == legacy_artifact
    assert legacy["correlationId"] is None
    assert REQUIRED_EVIDENCE <= legacy.keys()

    unsafe = 'break"}\n{"injected":true}'
    encoded, encoded_artifact = generate_evidence(unsafe)
    assert encoded == encoded_artifact
    assert encoded["correlationId"] == unsafe
    assert encoded["targetSha"] == SHA
    assert encoded["actualSha"] == SHA
    assert set(encoded) == REQUIRED_EVIDENCE | {"correlationId"}

    assert OTHER_SHA not in SOURCE
    print("dashboard validation correlation contract: PASS")


if __name__ == "__main__":
    main()
