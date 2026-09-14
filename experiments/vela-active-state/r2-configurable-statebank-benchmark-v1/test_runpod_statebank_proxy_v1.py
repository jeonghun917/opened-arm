#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("vela_statebank_supervisor_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    supervisor = Path(__file__).with_name("runpod_statebank_supervisor_v1.py")
    mod = load_module(supervisor)
    root = Path(tempfile.mkdtemp(prefix="vela-statebank-proxy-test-"))
    token = "a" * 64
    source = "d" * 40
    model = "rwkv7-g1i-2.9b-20260805"
    retrieval = "OFF"
    ctx = mod.Ctx(root, token, source, model, retrieval)
    mod.atomic(
        ctx.status,
        {
            "schema": "vela-statebank-proxy-status:v1",
            "state": "RUNNING",
            "phase": "run_benchmark",
            "boot_id": "boot",
            "source_commit": source,
            "model_profile_id": model,
            "retrieval_mode": retrieval,
        },
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.handler(ctx))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    status = json.load(urllib.request.urlopen(f"{base}/v1/status/{token}"))
    assert status["state"] == "RUNNING"
    assert status["source_commit"] == source
    try:
        urllib.request.urlopen(f"{base}/v1/status/wrong")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("wrong token was accepted")

    result = {
        "schema": "vela-statebank-benchmark-result:v1",
        "status": "COMPLETE",
        "scientific_evidence": True,
        "source_commit": source,
        "model_profile": {"id": model},
        "retrieval_mode": retrieval,
    }
    raw = json.dumps(result, separators=(",", ":"), sort_keys=True).encode()
    ctx.result.write_bytes(raw)

    try:
        urllib.request.urlopen(f"{base}/v1/result/{token}")
    except urllib.error.HTTPError as exc:
        assert exc.code == 409
    else:
        raise AssertionError("result was served before COMPLETE status")

    mod.atomic(
        ctx.status,
        {
            "schema": "vela-statebank-proxy-status:v1",
            "state": "COMPLETE",
            "phase": "result_ready",
            "boot_id": "boot",
            "source_commit": source,
            "model_profile_id": model,
            "retrieval_mode": retrieval,
            "result_bytes": len(raw),
            "result_sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    assert urllib.request.urlopen(f"{base}/v1/result/{token}").read() == raw
    server.shutdown()
    print("STATEBANK_PROXY_SYNTHETIC_PASS")


if __name__ == "__main__":
    main()
