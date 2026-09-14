#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

PINNED_IMAGE = "pytorch/pytorch@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3"
PROXY_STARTUP_GRACE_SECONDS=120
POLL_INTERVAL_SECONDS=3


class ControllerSignal(Exception):
    def __init__(self, signum: int):
        super().__init__(f"controller signal {signum}")
        self.signum = signum


def sh(cmd, **kwargs):
    return subprocess.run(cmd, text=True, **kwargs)


def emit(tag: str, **data) -> None:
    print(json.dumps({"tag": tag, "unix": int(time.time()), **data}, sort_keys=True), flush=True)


def safe_rel(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe relative path {value}")
    return path.as_posix()


def request_json(url: str, timeout: int = 6) -> tuple[int, dict]:
    """Read the RunPod HTTPS proxy with curl.

    Keep this transport aligned with the previously hardened v4 controller.
    RunPod's public proxy is fronted by Cloudflare, and Python urllib's default
    HTTP fingerprint can receive a proxy-layer 403 before the pod service.
    """
    cmd = [
        "curl",
        "--silent",
        "--show-error",
        "--location",
        "--connect-timeout",
        "3",
        "--max-time",
        str(timeout),
        "--output",
        "-",
        "--write-out",
        "\n%{http_code}",
        url,
    ]
    try:
        proc = sh(cmd, capture_output=True, timeout=timeout + 5)
    except Exception:
        return 0, {}

    output = proc.stdout or ""
    body, sep, status_text = output.rpartition("\n")
    if not sep:
        return 0, {}
    try:
        status = int(status_text.strip())
    except ValueError:
        status = 0
    try:
        obj = json.loads(body) if body else {}
    except Exception:
        obj = {}
    return status, obj if isinstance(obj, dict) else {}


def fetch_result(url: str, path: Path, timeout: int = 15) -> tuple[int, bytes]:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        proc = sh(
            [
                "curl",
                "--silent",
                "--show-error",
                "--location",
                "--connect-timeout",
                "3",
                "--max-time",
                str(timeout),
                "--output",
                str(tmp),
                "--write-out",
                "%{http_code}",
                url,
            ],
            capture_output=True,
            timeout=timeout + 5,
        )
        try:
            status = int((proc.stdout or "").strip())
        except ValueError:
            status = 0
        if status == 200 and tmp.exists():
            raw = tmp.read_bytes()
            tmp.replace(path)
            return status, raw
    except Exception:
        status = 0
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    return status, b""


def image_ref(evidence: Path) -> str:
    (evidence / "image-ref.txt").write_text(PINNED_IMAGE + "\n")
    return PINNED_IMAGE


def redact(text: str, token: str, supervisor_b64: str) -> str:
    return text.replace(token, "<redacted-result-token>").replace(supervisor_b64, "<redacted-supervisor-b64>")


def sanitize(obj, token: str, supervisor_b64: str):
    if isinstance(obj, dict):
        return {
            key: (
                "<redacted>"
                if key in {"VELA_RESULT_TOKEN", "VELA_SUPERVISOR_B64"}
                else sanitize(value, token, supervisor_b64)
            )
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [
            (
                "VELA_RESULT_TOKEN=<redacted>"
                if isinstance(value, str) and value.startswith("VELA_RESULT_TOKEN=")
                else "VELA_SUPERVISOR_B64=<redacted>"
                if isinstance(value, str) and value.startswith("VELA_SUPERVISOR_B64=")
                else sanitize(value, token, supervisor_b64)
            )
            for value in obj
        ]
    return redact(obj, token, supervisor_b64) if isinstance(obj, str) else obj


def save_obj(raw: str, path: Path, token: str, supervisor_b64: str) -> None:
    try:
        path.write_text(json.dumps(sanitize(json.loads(raw), token, supervisor_b64), indent=2, sort_keys=True) + "\n")
    except Exception:
        path.write_text(
            json.dumps({"unparsed_sha256": hashlib.sha256(raw.encode()).hexdigest(), "bytes": len(raw)}) + "\n"
        )


def logs(pod: str, evidence: Path, token: str, supervisor_b64: str, label: str) -> None:
    for source in ("system", "container"):
        try:
            proc = sh(
                ["runpodctl", "pod", "logs", pod, "--since", "3m", "--source", source],
                capture_output=True,
                timeout=20,
            )
            body = redact(
                (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else ""),
                token,
                supervisor_b64,
            )
            exit_code = proc.returncode
        except Exception as exc:
            body = f"LOG_CAPTURE_EXCEPTION:{type(exc).__name__}:{exc}\n"
            exit_code = -1
        (evidence / f"{label}-{source}.log").write_text(body)
        emit("VELA_POD_LOG_SNAPSHOT", source=source, exit_code=exit_code, tail=body[-1600:].replace("\n", " | "))


def pod_get(pod: str, evidence: Path, token: str, supervisor_b64: str):
    try:
        proc = sh(["runpodctl", "pod", "get", pod, "--include-machine"], capture_output=True, timeout=20)
    except Exception as exc:
        (evidence / "last-pod-get.stderr").write_text(f"POD_GET_EXCEPTION:{type(exc).__name__}:{exc}\n")
        return None, None, None

    err = redact(proc.stderr or "", token, supervisor_b64)
    (evidence / "last-pod-get.stderr").write_text(err)
    if proc.returncode:
        if '"code":"not_found"' in err or '"code": "not_found"' in err:
            return None, None, "POD_NOT_FOUND"
        return None, None, None
    try:
        obj = json.loads(proc.stdout or "")
    except Exception:
        save_obj(proc.stdout or "", evidence / "last-pod-get.json", token, supervisor_b64)
        return None, None, "POD_GET_INVALID_JSON"

    (evidence / "last-pod-get.json").write_text(
        json.dumps(sanitize(obj, token, supervisor_b64), indent=2, sort_keys=True) + "\n"
    )
    runtime_status = str(obj.get("runtimeStatus")).lower() if obj.get("runtimeStatus") is not None else None
    uptime = obj.get("uptimeSeconds")
    return runtime_status, float(uptime) if isinstance(uptime, (int, float)) else None, None


def status_error(status: dict, source: str, model: str, retrieval: str):
    for key, value in {
        "schema": "vela-statebank-proxy-status:v1",
        "source_commit": source,
        "model_profile_id": model,
        "retrieval_mode": retrieval,
    }.items():
        if status.get(key) != value:
            return f"{key}:{status.get(key)!r}!={value!r}"
    return None if status.get("state") in {"BOOTING", "RUNNING", "FAILED", "COMPLETE"} else f"state:{status.get('state')!r}"


def integrity_error(status: dict, raw: bytes):
    if not isinstance(status.get("result_bytes"), int) or status["result_bytes"] != len(raw):
        return "result_bytes_mismatch"
    digest = hashlib.sha256(raw).hexdigest()
    return None if status.get("result_sha256") == digest else "result_sha256_mismatch"


def delete_pod(pod: str, evidence: Path, token: str, supervisor_b64: str) -> bool:
    for attempt in range(1, 4):
        emit("VELA_POD_DELETE_ATTEMPT", pod_id=pod, attempt=attempt)
        try:
            proc = sh(["runpodctl", "pod", "delete", pod], capture_output=True, timeout=30)
            out = redact(proc.stdout or "", token, supervisor_b64)
            err = redact(proc.stderr or "", token, supervisor_b64)
            (evidence / "pod-delete.stdout").write_text(out)
            (evidence / "pod-delete.stderr").write_text(err)
        except Exception as exc:
            proc = None
            err = ""
            (evidence / "pod-delete.stderr").write_text(f"DELETE_EXCEPTION:{type(exc).__name__}:{exc}\n")
        if proc is not None and (
            proc.returncode == 0 or '"code":"not_found"' in err or '"code": "not_found"' in err
        ):
            emit("VELA_POD_DELETE_DONE", pod_id=pod, deleted=True, exit_code=proc.returncode)
            return True
        if attempt < 3:
            time.sleep(2)
    emit("VELA_POD_DELETE_DONE", pod_id=pod, deleted=False, exit_code=-1)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    required = (
        "VELA_SOURCE_COMMIT",
        "VELA_MODEL_PROFILE_ID",
        "VELA_RETRIEVAL_MODE",
        "VELA_PROBLEM_PACK_REL",
        "VELA_EXECUTION_CONFIG_REL",
    )
    for key in required:
        if not os.environ.get(key):
            raise SystemExit(f"{key} required")

    source = os.environ["VELA_SOURCE_COMMIT"]
    model = os.environ["VELA_MODEL_PROFILE_ID"]
    retrieval = os.environ["VELA_RETRIEVAL_MODE"]
    if len(source) != 40:
        raise SystemExit("exact source SHA required")

    pack = safe_rel(os.environ["VELA_PROBLEM_PACK_REL"])
    config = safe_rel(os.environ["VELA_EXECUTION_CONFIG_REL"])
    evidence = Path(os.environ.get("VELA_EVIDENCE_DIR", "/tmp/vela-statebank"))
    evidence.mkdir(parents=True, exist_ok=True)
    plan = {
        "source_commit": source,
        "model_profile_id": model,
        "retrieval_mode": retrieval,
        "problem_pack_rel": pack,
        "execution_config_rel": config,
        "max_pod_count": 1,
        "model_selection_cardinality": 1,
        "provider": "RUNPOD",
        "gpu": "NVIDIA A40",
        "image": PINNED_IMAGE,
    }
    (evidence / "execution-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return

    if os.environ.get("VELA_PAID_EXECUTION_AUTHORIZED") != "true":
        raise SystemExit("paid execution not authorized")
    if not os.environ.get("RUNPOD_API_KEY"):
        raise SystemExit("RUNPOD_API_KEY required")

    max_runtime = int(os.environ.get("VELA_MAX_RUNTIME_SECONDS", "900"))
    result_port = 8000
    sh(["runpodctl", "user"], stdout=subprocess.DEVNULL, check=True)

    gpu_list = evidence / "gpu-list.json"
    gpu_list.write_text(sh(["runpodctl", "gpu", "list", "--include-unavailable"], capture_output=True, check=True).stdout)
    selection_path = evidence / "selection.json"
    sh(
        [sys.executable, str(Path(__file__).with_name("runpod_select_a40_capacity_v1.py")), str(gpu_list), str(selection_path)],
        check=True,
    )
    selection = json.loads(selection_path.read_text())
    data_center = selection["selectedDataCenter"]["dataCenterId"]
    price = float(selection["securePricePerHr"])
    emit(
        "VELA_PROVIDER_SELECTION",
        data_center=data_center,
        stock=selection["selectedDataCenter"].get("stockStatus"),
        price_per_hr=price,
    )

    supervisor_b64 = base64.b64encode(Path(__file__).with_name("runpod_statebank_supervisor_v1.py").read_bytes()).decode()
    token = secrets.token_hex(32)
    (evidence / "result-token-sha256.txt").write_text(hashlib.sha256(token.encode()).hexdigest() + "\n")
    env = {
        "VELA_SUPERVISOR_B64": supervisor_b64,
        "VELA_RESULT_TOKEN": token,
        "VELA_SOURCE_COMMIT": source,
        "VELA_BASE_RAW": f"https://raw.githubusercontent.com/jeonghun917/opened-arm/{source}/experiments/vela-active-state/r2-configurable-statebank-benchmark-v1",
        "VELA_MODEL_PROFILE_ID": model,
        "VELA_RETRIEVAL_MODE": retrieval,
        "VELA_PROBLEM_PACK_REL": pack,
        "VELA_EXECUTION_CONFIG_REL": config,
        "VELA_RESULT_PORT": str(result_port),
        "VELA_WORKLOAD_TIMEOUT_SECONDS": str(max_runtime - 60),
        "PYTHONUNBUFFERED": "1",
    }
    scale = model.split("-")[-2]
    name = f"vela-statebank-{scale}-{source[:10]}"
    image_ref(evidence)
    create_cmd = [
        "runpodctl",
        "pod",
        "create",
        "--name",
        name,
        "--image",
        PINNED_IMAGE,
        "--gpu-id",
        "NVIDIA A40",
        "--gpu-count",
        "1",
        "--cloud-type",
        "SECURE",
        "--data-center-ids",
        data_center,
        "--container-disk-in-gb",
        "50",
        "--ports",
        f"{result_port}/http",
        "--ssh=false",
        "--min-cuda-version",
        "12.6",
        "--env",
        json.dumps(env, separators=(",", ":")),
        "--docker-args",
        "sh -lc 'echo \"$VELA_SUPERVISOR_B64\" | base64 -d > /tmp/vela_supervisor.py && exec python /tmp/vela_supervisor.py'",
    ]
    emit("VELA_POD_CREATE_START", name=name, image=PINNED_IMAGE)
    create = sh(create_cmd, capture_output=True)
    save_obj(create.stdout or "", evidence / "pod-create.json", token, supervisor_b64)
    (evidence / "pod-create.stderr").write_text(redact(create.stderr or "", token, supervisor_b64))
    if create.returncode:
        raise SystemExit(create.returncode)

    created = json.loads(create.stdout)
    pod = str(created.get("id") or created.get("podId") or "")
    if not pod:
        raise SystemExit("pod id missing")
    (evidence / "pod-id.txt").write_text(pod + "\n")

    proxy = f"https://{pod}-{result_port}.proxy.runpod.net"
    start = time.time()
    found = False
    terminal = ""
    deleted = False
    first_running = None
    last_uptime = None
    last_report = 0.0
    last_phase = None
    last_diag = 0.0
    controller_signal = None
    old_handlers = {}

    def on_signal(signum, _frame):
        raise ControllerSignal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, on_signal)

    emit("VELA_POD_CREATED", pod_id=pod, proxy_host=f"{pod}-{result_port}.proxy.runpod.net")
    try:
        while time.time() - start < max_runtime:
            now = time.time()
            runtime_status, uptime, control_failure = pod_get(pod, evidence, token, supervisor_b64)
            if control_failure:
                terminal = control_failure
                logs(pod, evidence, token, supervisor_b64, "pod-control-failure")
                break
            if runtime_status == "running" and first_running is None:
                first_running = now
                emit("VELA_POD_RUNNING", pod_id=pod, elapsed_seconds=int(now - start))
            if runtime_status in {"failed", "exited", "stopped", "terminated"}:
                terminal = f"POD_RUNTIME_TERMINAL_{runtime_status.upper()}"
                logs(pod, evidence, token, supervisor_b64, "pod-runtime-terminal")
                break
            if uptime is not None:
                if last_uptime is not None and last_uptime >= 20 and uptime + 10 < last_uptime:
                    terminal = "CONTAINER_RESTART_DETECTED_BY_CONTROL_PLANE"
                    logs(pod, evidence, token, supervisor_b64, "container-restart")
                    break
                last_uptime = uptime

            code, status = request_json(f"{proxy}/v1/status/{token}")
            phase = status.get("phase") if isinstance(status, dict) else None
            state = status.get("state") if isinstance(status, dict) else None
            if code == 200:
                bad = status_error(status, source, model, retrieval)
                if bad:
                    terminal = "PROXY_STATUS_IDENTITY_MISMATCH"
                    (evidence / "terminal-status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
                    emit("VELA_STATUS_IDENTITY_FAILURE", detail=bad)
                    break
                (evidence / "last-status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
                if phase != last_phase or now - last_report >= 15:
                    emit(
                        "VELA_POLL",
                        elapsed_seconds=int(now - start),
                        http=code,
                        state=state,
                        phase=phase,
                        runtime_status=runtime_status,
                        uptime_seconds=uptime,
                    )
                    last_report = now
                    last_phase = phase
                if state == "FAILED":
                    terminal = "SUPERVISOR_FAILED"
                    logs(pod, evidence, token, supervisor_b64, "failure")
                    break
                if state == "COMPLETE":
                    result_code, raw = fetch_result(f"{proxy}/v1/result/{token}", evidence / "benchmark_result.json")
                    bad = integrity_error(status, raw) if result_code == 200 else f"http_{result_code}"
                    if bad:
                        terminal = "RESULT_INTEGRITY_OR_FETCH_FAILURE"
                        emit("VELA_RESULT_INTEGRITY_FAILURE", detail=bad)
                        logs(pod, evidence, token, supervisor_b64, "result-failure")
                        break
                    result = json.loads(raw)
                    if (
                        result.get("schema") != "vela-statebank-benchmark-result:v1"
                        or result.get("source_commit") != source
                        or result.get("model_profile", {}).get("id") != model
                        or result.get("retrieval_mode") != retrieval
                    ):
                        terminal = "RESULT_IDENTITY_MISMATCH"
                        break
                    found = True
                    terminal = "RESULT_FETCHED"
                    break
            else:
                if now - last_report >= 15:
                    emit(
                        "VELA_POLL",
                        elapsed_seconds=int(now - start),
                        http=code,
                        state="UNREACHABLE",
                        phase="proxy_wait",
                        runtime_status=runtime_status,
                        uptime_seconds=uptime,
                    )
                    last_report = now
                if first_running is not None and now - first_running >= 30 and now - last_diag >= 60:
                    logs(pod, evidence, token, supervisor_b64, f"proxy-debug-{int(now-start)}s")
                    last_diag = now
                if first_running is not None and now - first_running >= PROXY_STARTUP_GRACE_SECONDS:
                    terminal = "PROXY_UNREACHABLE_AFTER_RUNNING_GRACE"
                    logs(pod, evidence, token, supervisor_b64, "proxy-start-timeout")
                    break

            time.sleep(POLL_INTERVAL_SECONDS)

        if not terminal:
            terminal = "MAX_RUNTIME_EXPIRED"
            logs(pod, evidence, token, supervisor_b64, "max-runtime")
    except ControllerSignal as exc:
        controller_signal = exc.signum
        terminal = "CONTROLLER_CANCELLED"
        logs(pod, evidence, token, supervisor_b64, "controller-cancelled")
    finally:
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            emit("VELA_POD_DELETE_START", pod_id=pod, terminal_reason=terminal)
            deleted = delete_pod(pod, evidence, token, supervisor_b64)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)

    elapsed = min(max_runtime, max(0, int(time.time() - start)))
    envelope = {
        "status": "PASS" if found and deleted else "FAIL",
        "scientific_evidence": bool(found),
        "source_commit": source,
        "model_profile_id": model,
        "retrieval_mode": retrieval,
        "pod_id": pod,
        "pod_deleted": deleted,
        "terminal_reason": terminal,
        "elapsed_seconds": elapsed,
        "max_runtime_seconds": max_runtime,
        "secure_price_per_hr_usd": price,
        "estimated_gpu_cost_upper_bound_usd": round(elapsed * price / 3600, 6),
        "data_center_id": data_center,
        "max_pod_count": 1,
        "model_selection_cardinality": 1,
        "controller_signal": controller_signal,
    }
    (evidence / "provider-envelope.json").write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    print(json.dumps(envelope, sort_keys=True), flush=True)
    if controller_signal is not None:
        raise SystemExit(128 + controller_signal)
    raise SystemExit(0 if envelope["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
