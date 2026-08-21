from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import kaggle_cori_kernel as legacy
from kaggle_offline_runtime import prepare_offline_environment

WORK = Path("/kaggle/working")
MATCHA = WORK / "Matcha-TTS"
OUTPUT = WORK / "c3-cori-kaggle-runs"


def main() -> None:
    started = time.monotonic()

    # Private C3 assets are discovered and SHA-gated by the already-tested mount
    # normalization path. No source audio or checkpoint bytes are logged.
    source = legacy.discover_private_input()
    print("C3_KAGGLE_PRIVATE_INPUT_DISCOVERED", source.name, flush=True)

    # This bootstrap performs no apt/pip/git network access. Public Matcha source,
    # eSpeak-NG packages and small Python dependency wheels come from the attached
    # private runtime Dataset. Kaggle's own CUDA-matched torch/numpy/scipy remain in place.
    offline_runtime = prepare_offline_environment(MATCHA)

    env = legacy.environment_report()
    if env["gpu_count"] != 1 or "T4" not in str(env["gpu"]).upper():
        raise RuntimeError(f"first benchmark requires exactly one T4; environment={env}")
    if env["matcha_commit"] != legacy.MATCHA_COMMIT:
        raise RuntimeError("Matcha commit mismatch after offline environment setup")
    print("C3_KAGGLE_ENVIRONMENT", json.dumps(env, ensure_ascii=False), flush=True)

    here = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(here / "kaggle_cori_segment_entry.py"),
        "--handoff-dir",
        str(source / "c3-cori-handoff"),
        "--dataset-root",
        str(source / "c3-cori-dataset"),
        "--resume-checkpoint",
        str(source / "c3-cori-e280" / "checkpoint_epoch=279.ckpt"),
        "--output-root",
        str(OUTPUT),
        "--matcha-checkout",
        str(MATCHA),
        "--target-max-epochs",
        "290",
    ]
    legacy.run(cmd)

    elapsed = time.monotonic() - started
    report = {
        "schema": "c3-cori-kaggle-t4-offline-wrapper-v1",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "wall_seconds_including_offline_environment_setup": elapsed,
        "environment": env,
        "offline_runtime": offline_runtime,
        "target_epoch": 290,
        "batch_size": 16,
        "e280_sha256": legacy.E280_SHA256,
        "network_enabled": False,
    }
    (WORK / "C3_KAGGLE_T4_BENCHMARK_WRAPPER_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("C3_KAGGLE_T4_OFFLINE_WRAPPER_PASS")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
