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
class Ctx:
    root: Path
    token: str
    source: str
    model: str
    retrieval: str

    @property
    def status(self) -> Path:
        return self.root / ".vela-status.json"

    @property
    def result(self) -> Path:
        return self.root / "benchmark_result.json"

    @property
    def started(self) -> Path:
        return self.root / ".vela-started"


def atomic(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n")
    tmp.replace(path)


def emit(tag: str, **data: Any) -> None:
    print(json.dumps({"tag": tag, "unix": int(time.time()), **data}, sort_keys=True), flush=True)


def read_tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(errors="replace")[-limit:]
    except Exception:
        return ""


def load_status(ctx: Ctx) -> dict[str, Any]:
    try:
        obj = json.loads(ctx.status.read_text())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def set_status(ctx: Ctx, state: str, phase: str, boot: str, **extra: Any) -> None:
    payload: dict[str, Any] = {
        "schema": "vela-statebank-proxy-status:v1",
        "state": state,
        "phase": phase,
        "boot_id": boot,
        "source_commit": ctx.source,
        "model_profile_id": ctx.model,
        "retrieval_mode": ctx.retrieval,
        "updated_unix": int(time.time()),
    }
    payload.update(extra)
    atomic(ctx.status, payload)
    emit("VELA_STATUS", state=state, phase=phase, **extra)


def valid_result(ctx: Ctx) -> tuple[dict[str, Any], bytes] | None:
    try:
        raw = ctx.result.read_bytes()
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("schema") != "vela-statebank-benchmark-result:v1":
        return None
    if obj.get("status") != "COMPLETE" or obj.get("scientific_evidence") is not True:
        return None
    if obj.get("source_commit") != ctx.source:
        return None
    if obj.get("model_profile", {}).get("id") != ctx.model:
        return None
    if obj.get("retrieval_mode") != ctx.retrieval:
        return None
    return obj, raw


def handler(ctx: Ctx):
    class H(BaseHTTPRequestHandler):
        server_version = "VelaStateBankProxy/1"

        def log_message(self, *_args: Any) -> None:
            return

        def sendb(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == f"/v1/status/{ctx.token}":
                if not ctx.status.exists():
                    self.sendb(503, b'{"state":"STARTING"}\n')
                    return
                self.sendb(200, ctx.status.read_bytes())
                return
            if self.path == f"/v1/result/{ctx.token}":
                status = load_status(ctx)
                result = valid_result(ctx)
                if status.get("state") != "COMPLETE" or result is None:
                    self.sendb(409, b'{"error":"RESULT_NOT_READY"}\n')
                    return
                self.sendb(200, ctx.result.read_bytes())
                return
            self.sendb(404, b'{"error":"NOT_FOUND"}\n')

    return H


def heartbeat(ctx: Ctx, stop: threading.Event) -> None:
    while not stop.wait(15):
        sizes: dict[str, int] = {}
        for name in ("pip_install.stdout", "pip_install.stderr", "run_benchmark.stdout", "run_benchmark.stderr"):
            path = ctx.root / name
            if path.exists():
                sizes[name] = path.stat().st_size
        status = load_status(ctx) or {"state": "STARTING", "phase": "none"}
        emit("VELA_HEARTBEAT", state=status.get("state"), phase=status.get("phase"), file_sizes=sizes)


def run(
    ctx: Ctx,
    boot: str,
    deadline: float,
    phase: str,
    cmd: list[str],
    env: dict[str, str] | None = None,
) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 124
    set_status(ctx, "RUNNING", phase, boot)
    stdout_path = ctx.root / f"{phase}.stdout"
    stderr_path = ctx.root / f"{phase}.stderr"
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        try:
            rc = subprocess.run(
                cmd,
                cwd=ctx.root,
                env=env,
                stdout=out,
                stderr=err,
                timeout=remaining,
                check=False,
            ).returncode
            emit("VELA_PHASE_EXIT", phase=phase, exit_code=rc)
            return rc
        except subprocess.TimeoutExpired:
            emit("VELA_PHASE_TIMEOUT", phase=phase)
            return 124


def dl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    emit("VELA_DOWNLOAD_START", name=dest.name)
    urllib.request.urlretrieve(url, dest)
    emit("VELA_DOWNLOAD_DONE", name=dest.name, bytes=dest.stat().st_size)


def download_pack(base: str, rel: str, dest: Path) -> Path:
    target = dest / "PACK_INPUT"
    dl(base + "/" + rel, target)
    try:
        descriptor = json.loads(target.read_text())
    except Exception:
        return target
    if descriptor.get("schema") != "vela-base64-zip-chunks:v1":
        return target
    packdir = dest / "pack"
    packdir.mkdir()
    desc = packdir / "PACK_V1.json"
    desc.write_text(json.dumps(descriptor, indent=2) + "\n")
    for chunk in descriptor["chunks"]:
        dl(base + "/" + str(Path(rel).parent / chunk), packdir / chunk)
    return desc


def work(ctx: Ctx, boot: str) -> None:
    deadline = time.monotonic() + int(os.environ.get("VELA_WORKLOAD_TIMEOUT_SECONDS", "840"))
    base = os.environ["VELA_BASE_RAW"].rstrip("/")
    try:
        recovered = valid_result(ctx)
        if recovered is not None:
            _, raw = recovered
            set_status(
                ctx,
                "COMPLETE",
                "recovered",
                boot,
                scientific_evidence=True,
                result_bytes=len(raw),
                result_sha256=hashlib.sha256(raw).hexdigest(),
            )
            return
        if ctx.started.exists():
            set_status(
                ctx,
                "FAILED",
                "restart_recovery",
                boot,
                failure_class="RESTART_WITHOUT_VALID_RESULT",
                scientific_evidence=False,
            )
            return
        ctx.started.write_text(boot + "\n")

        pip_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--no-input",
            "rwkv==0.8.32",
            "tokenizers==0.21.4",
            "numpy==2.4.6",
            "huggingface_hub==0.36.0",
        ]
        rc = run(ctx, boot, deadline, "pip_install", pip_cmd)
        if rc != 0:
            set_status(
                ctx,
                "FAILED",
                "pip_install",
                boot,
                failure_class="PIP_INSTALL_FAILED" if rc != 124 else "PIP_INSTALL_TIMEOUT",
                exit_code=rc,
                scientific_evidence=False,
                failure_detail=read_tail(ctx.root / "pip_install.stderr"),
            )
            return

        set_status(ctx, "RUNNING", "download_sources", boot)
        for name in ("statebank_runner.py", "benchmark_contract.py", "benchmark_metrics.py", "rwkv_state_engine.py"):
            dl(base + "/" + name, ctx.root / name)
        dl(base + "/" + os.environ["VELA_EXECUTION_CONFIG_REL"], ctx.root / "execution_config.json")
        pack = download_pack(base, os.environ["VELA_PROBLEM_PACK_REL"], ctx.root / "problem_pack")

        env = os.environ.copy()
        env["VELA_SOURCE_COMMIT"] = ctx.source
        rc = run(
            ctx,
            boot,
            deadline,
            "run_benchmark",
            [
                sys.executable,
                "statebank_runner.py",
                "--execution-config",
                "execution_config.json",
                "--problem-pack",
                str(pack),
                "--model-profile",
                ctx.model,
                "--retrieval-mode",
                ctx.retrieval,
                "--output",
                "benchmark_result.json",
            ],
            env,
        )
        result = valid_result(ctx)
        if rc != 0 or result is None:
            set_status(
                ctx,
                "FAILED",
                "run_benchmark",
                boot,
                failure_class="RUNNER_FAILED_OR_INVALID_RESULT" if rc != 124 else "RUNNER_TIMEOUT",
                exit_code=rc,
                scientific_evidence=False,
                result_present=ctx.result.exists(),
                stdout_tail=read_tail(ctx.root / "run_benchmark.stdout"),
                failure_detail=read_tail(ctx.root / "run_benchmark.stderr"),
            )
            return

        _, raw = result
        set_status(
            ctx,
            "COMPLETE",
            "result_ready",
            boot,
            scientific_evidence=True,
            result_bytes=len(raw),
            result_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except Exception as exc:
        set_status(
            ctx,
            "FAILED",
            "supervisor_exception",
            boot,
            failure_class=type(exc).__name__,
            scientific_evidence=False,
            failure_detail=(str(exc) + "\n" + traceback.format_exc())[-4000:],
        )


def main() -> None:
    token = os.environ.get("VELA_RESULT_TOKEN", "")
    source = os.environ.get("VELA_SOURCE_COMMIT", "")
    model = os.environ.get("VELA_MODEL_PROFILE_ID", "")
    retrieval = os.environ.get("VELA_RETRIEVAL_MODE", "")
    if len(token) < 32 or len(source) != 40 or not model or not retrieval:
        raise SystemExit("invalid identity environment")
    root = Path(os.environ.get("VELA_WORK_ROOT", "/workspace/vela"))
    root.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(root, token, source, model, retrieval)
    boot = uuid.uuid4().hex
    set_status(ctx, "BOOTING", "supervisor_start", boot)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(ctx, stop), daemon=True).start()
    threading.Thread(target=work, args=(ctx, boot), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("VELA_RESULT_PORT", "8000"))), handler(ctx))
    server.daemon_threads = True
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stop.set()


if __name__ == "__main__":
    main()
