#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


@dataclass
class ServiceContext:
    root: Path
    token: str
    condition: str
    source_commit: str

    @property
    def status_path(self) -> Path:
        return self.root / ".vela-proxy-status.json"

    @property
    def result_path(self) -> Path:
        return self.root / f"condition_result_{self.condition}.json"

    @property
    def started_path(self) -> Path:
        return self.root / f".vela-science-started-{self.condition}"

    @property
    def complete_path(self) -> Path:
        return self.root / f".vela-science-complete-{self.condition}"


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def validated_result(ctx: ServiceContext) -> tuple[dict[str, Any], bytes] | None:
    try:
        raw = ctx.result_path.read_bytes()
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("status") != "P-R2-02_CONDITION_RESULT":
        return None
    if obj.get("scientific_evidence") is not True:
        return None
    if obj.get("condition") != ctx.condition:
        return None
    if obj.get("source_commit") != ctx.source_commit:
        return None
    return obj, raw


def status_payload(ctx: ServiceContext, state: str, phase: str, boot_id: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "vela-runpod-science-proxy-status:v1",
        "state": state,
        "phase": phase,
        "condition": ctx.condition,
        "source_commit": ctx.source_commit,
        "boot_id": boot_id,
        "updated_unix": int(time.time()),
    }
    payload.update(extra)
    return payload


def update_status(ctx: ServiceContext, state: str, phase: str, boot_id: str, **extra: Any) -> None:
    atomic_write_json(ctx.status_path, status_payload(ctx, state, phase, boot_id, **extra))


def handler_for(ctx: ServiceContext):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VelaResultProxy/1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            status_path = f"/v1/status/{ctx.token}"
            result_path = f"/v1/result/{ctx.token}"
            if self.path == status_path:
                if not ctx.status_path.exists():
                    self._send(503, b'{"state":"STARTING"}\n')
                    return
                self._send(200, ctx.status_path.read_bytes())
                return
            if self.path == result_path:
                status = load_json(ctx.status_path) or {}
                if status.get("state") != "COMPLETE" or not ctx.result_path.exists():
                    self._send(409, b'{"error":"RESULT_NOT_READY"}\n')
                    return
                self._send(200, ctx.result_path.read_bytes())
                return
            self._send(404, b'{"error":"NOT_FOUND"}\n')

    return Handler


def read_tail(path: Path, limit: int = 4000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return data[-limit:]


def run_subprocess(
    ctx: ServiceContext,
    boot_id: str,
    deadline: float,
    phase: str,
    cmd: list[str],
    env: dict[str, str] | None = None,
) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"deadline exhausted before {phase}")
    stdout_path = ctx.root / f"supervisor-{phase.lower()}.stdout"
    stderr_path = ctx.root / f"supervisor-{phase.lower()}.stderr"
    update_status(ctx, "RUNNING", phase, boot_id)
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        try:
            proc = subprocess.run(cmd, cwd=ctx.root, env=env, stdout=out, stderr=err, timeout=remaining, check=False)
            return int(proc.returncode)
        except subprocess.TimeoutExpired:
            update_status(
                ctx,
                "FAILED",
                phase,
                boot_id,
                failure_class="WORKLOAD_TIMEOUT",
                failure_detail=f"deadline expired in {phase}",
            )
            return 124


def download_sources(ctx: ServiceContext, boot_id: str, deadline: float, base_raw: str) -> None:
    files = [
        "runpod_condition_adapter_v1.py",
        "condition_runner.py",
        "science_common.py",
        "EVALUATOR.json",
        "EXTERNAL_TRACE.json",
        "MODEL_STIMULI_INDEX.json",
    ] + [f"stimuli/episode_{i}.json" for i in range(1, 7)]
    update_status(ctx, "RUNNING", "DOWNLOAD_SOURCES", boot_id)
    for rel in files:
        if time.monotonic() >= deadline:
            raise TimeoutError("deadline exhausted while downloading sources")
        dest = ctx.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(base_raw.rstrip("/") + "/" + rel, dest)


def workload(ctx: ServiceContext, boot_id: str) -> None:
    timeout_seconds = int(os.environ.get("VELA_WORKLOAD_TIMEOUT_SECONDS", "420"))
    deadline = time.monotonic() + timeout_seconds
    base_raw = os.environ["VELA_BASE_RAW"]

    try:
        recovered = validated_result(ctx)
        if recovered is not None:
            _, raw = recovered
            ctx.complete_path.touch()
            update_status(
                ctx,
                "COMPLETE",
                "RECOVERED_EXISTING_RESULT",
                boot_id,
                recovered=True,
                scientific_evidence=True,
                result_bytes=len(raw),
                result_sha256=hashlib.sha256(raw).hexdigest(),
            )
            return

        if ctx.started_path.exists():
            update_status(
                ctx,
                "FAILED",
                "RESTART_RECOVERY",
                boot_id,
                failure_class="CONTAINER_RESTART_AFTER_SCIENCE_START_NO_VALID_RESULT",
                scientific_evidence=False,
            )
            return

        ctx.started_path.write_text(boot_id + "\n", encoding="utf-8")
        rc = run_subprocess(
            ctx,
            boot_id,
            deadline,
            "PIP_INSTALL",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu126",
                "torch==2.7.1+cu126",
                "rwkv==0.8.32",
                "tokenizers==0.21.4",
                "numpy==2.4.6",
                "huggingface_hub==0.36.0",
            ],
        )
        if rc != 0:
            update_status(
                ctx,
                "FAILED",
                "PIP_INSTALL",
                boot_id,
                failure_class="PIP_INSTALL_FAILED",
                exit_code=rc,
                scientific_evidence=False,
                failure_detail=read_tail(ctx.root / "supervisor-pip_install.stderr"),
            )
            return

        download_sources(ctx, boot_id, deadline, base_raw)

        env = os.environ.copy()
        env["VELA_CONDITION"] = ctx.condition
        env["VELA_SOURCE_COMMIT"] = ctx.source_commit
        rc = run_subprocess(ctx, boot_id, deadline, "RUN_CONDITION", [sys.executable, "runpod_condition_adapter_v1.py"], env=env)
        result = validated_result(ctx)
        if rc != 0 or result is None:
            update_status(
                ctx,
                "FAILED",
                "RUN_CONDITION",
                boot_id,
                failure_class="CONDITION_RUNNER_FAILED_OR_INVALID_RESULT",
                exit_code=rc,
                scientific_evidence=False,
                failure_detail=read_tail(ctx.root / "supervisor-run_condition.stderr"),
                result_file_present=ctx.result_path.exists(),
            )
            return

        _, raw = result
        ctx.complete_path.touch()
        update_status(
            ctx,
            "COMPLETE",
            "RESULT_READY",
            boot_id,
            scientific_evidence=True,
            result_bytes=len(raw),
            result_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except Exception as exc:
        update_status(
            ctx,
            "FAILED",
            "SUPERVISOR_EXCEPTION",
            boot_id,
            failure_class=type(exc).__name__,
            scientific_evidence=False,
            failure_detail=(str(exc) + "\n" + traceback.format_exc())[-4000:],
        )


def main() -> None:
    token = os.environ.get("VELA_RESULT_TOKEN", "")
    condition = os.environ.get("VELA_CONDITION", "")
    source_commit = os.environ.get("VELA_SOURCE_COMMIT", "")
    if len(token) < 32 or condition not in {"B", "C", "G", "M", "X"} or len(source_commit) != 40:
        raise SystemExit("invalid supervisor identity environment")

    root = Path(os.environ.get("VELA_WORK_ROOT", "/workspace/vela"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "stimuli").mkdir(parents=True, exist_ok=True)
    ctx = ServiceContext(root=root, token=token, condition=condition, source_commit=source_commit)
    boot_id = uuid.uuid4().hex
    update_status(ctx, "BOOTING", "SUPERVISOR_START", boot_id)

    worker = threading.Thread(target=workload, args=(ctx, boot_id), daemon=True)
    worker.start()

    port = int(os.environ.get("VELA_RESULT_PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler_for(ctx))
    server.daemon_threads = True
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
