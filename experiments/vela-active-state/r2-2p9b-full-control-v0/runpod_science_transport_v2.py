#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
from pathlib import Path


def iter_lines(path: Path):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        line = obj.get("line")
        if isinstance(line, str):
            yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("output")
    ap.add_argument("condition")
    ap.add_argument("source_commit")
    args = ap.parse_args()

    logs = Path(args.logs)
    output = Path(args.output)
    segments = []
    current = None
    ready = []
    bootstrap_exits = []

    for line in iter_lines(logs):
        if line.startswith("VELA_CONDITION_RESULT_META="):
            try:
                meta = json.loads(line.split("=", 1)[1])
            except Exception:
                current = None
                continue
            current = {"meta": meta, "parts": {}}
            continue

        if line.startswith("VELA_CONDITION_RESULT_CHUNK=") and current is not None:
            try:
                head, data = line.split(":", 1)
                seq = head.split("=", 1)[1]
                idx, total = map(int, seq.split("/"))
                if total != int(current["meta"].get("chunks", -1)):
                    continue
                current["parts"][idx] = data
                if len(current["parts"]) == total:
                    b64 = "".join(current["parts"][i] for i in range(1, total + 1))
                    blob = base64.b64decode(b64)
                    meta = current["meta"]
                    if len(blob) == int(meta.get("bytes", -1)) and hashlib.sha256(blob).hexdigest() == meta.get("sha256"):
                        try:
                            result = json.loads(blob)
                        except Exception:
                            result = None
                        if isinstance(result, dict):
                            segments.append((meta, blob, result))
                    current = None
            except Exception:
                continue
            continue

        if line.startswith("VELA_SCIENCE_RESULT_READY="):
            try:
                ready.append(json.loads(line.split("=", 1)[1]))
            except Exception:
                ready.append({"raw": line.split("=", 1)[1]})
            continue

        if line.startswith("VELA_SCIENCE_BOOTSTRAP_EXIT="):
            try:
                bootstrap_exits.append(int(line.split("=", 1)[1]))
            except Exception:
                bootstrap_exits.append(None)

    valid = []
    validation_errors = []
    for meta, blob, result in segments:
        if meta.get("condition") != args.condition or result.get("condition") != args.condition:
            continue
        if result.get("status") == "P-R2-02_CONDITION_RESULT" and result.get("scientific_evidence") is True:
            if result.get("source_commit") != args.source_commit:
                validation_errors.append("SOURCE_COMMIT_MISMATCH")
                continue
        valid.append((meta, blob, result))

    reconstructed = bool(valid)
    if reconstructed:
        output.write_bytes(valid[-1][1])

    status = {
        "ready_seen": bool(ready),
        "ready": ready[-1] if ready else None,
        "bootstrap_exit_seen": bool(bootstrap_exits),
        "bootstrap_exit": bootstrap_exits[-1] if bootstrap_exits else None,
        "result_reconstructed": reconstructed,
        "valid_segments": len(valid),
        "validation_errors": validation_errors,
    }
    print(json.dumps(status, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
