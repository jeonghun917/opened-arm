#!/usr/bin/env python3
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


def main() -> None:
    supervisor = Path(__file__).with_name("runpod_science_proxy_supervisor_v1.py")
    spec = importlib.util.spec_from_file_location("vela_proxy_supervisor", supervisor)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    root = Path(tempfile.mkdtemp())
    token = "a" * 64
    source = "d" * 40
    ctx = mod.ServiceContext(root, token, "M", source)
    mod.atomic_write_json(ctx.status_path, {
        "schema": "vela-runpod-science-proxy-status:v1",
        "state": "RUNNING",
        "phase": "RUN_CONDITION",
        "condition": "M",
        "source_commit": source,
    })
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.handler_for(ctx))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    status = json.load(urllib.request.urlopen(f"{base}/v1/status/{token}"))
    assert status["state"] == "RUNNING"
    try:
        urllib.request.urlopen(f"{base}/v1/status/wrong")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("wrong token was accepted")

    result = {
        "status": "P-R2-02_CONDITION_RESULT",
        "scientific_evidence": True,
        "condition": "M",
        "source_commit": source,
        "scales": {"0.4B": {}, "2.9B": {}},
    }
    raw = json.dumps(result, separators=(",", ":"), sort_keys=True).encode()
    ctx.result_path.write_bytes(raw)
    mod.atomic_write_json(ctx.status_path, {
        "schema": "vela-runpod-science-proxy-status:v1",
        "state": "COMPLETE",
        "phase": "RESULT_READY",
        "condition": "M",
        "source_commit": source,
        "result_bytes": len(raw),
        "result_sha256": hashlib.sha256(raw).hexdigest(),
    })
    assert urllib.request.urlopen(f"{base}/v1/result/{token}").read() == raw
    server.shutdown()
    print("RUNPOD_SCIENCE_PROXY_SYNTHETIC_PASS")


if __name__ == "__main__":
    main()
