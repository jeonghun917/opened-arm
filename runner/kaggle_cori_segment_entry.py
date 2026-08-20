from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightning_cori_continue_to550 as runner

EXPECTED_E280_SHA256 = "081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9"
EXPECTED_E280_EPOCH = 280
EXPECTED_E280_GLOBAL_STEP = 140560
DEFAULT_TARGET_EPOCH = 290


def print_gpu_preflight() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; enable a Kaggle GPU accelerator first")

    props = torch.cuda.get_device_properties(0)
    info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(props.total_memory),
        "gpu_total_memory_gib": round(props.total_memory / (1024 ** 3), 3),
        "gpu_count_visible": int(torch.cuda.device_count()),
    }
    print("C3_KAGGLE_GPU_PREFLIGHT", json.dumps(info, ensure_ascii=False), flush=True)
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("C3_KAGGLE_NVIDIA_SMI unavailable", flush=True)
    return info


def verify_resume(path: Path) -> tuple[int, int, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint missing: {path}")

    actual_sha = runner.sha256_file(path)
    if actual_sha != EXPECTED_E280_SHA256:
        raise RuntimeError(
            "E280 checkpoint SHA-256 mismatch: "
            f"actual={actual_sha} expected={EXPECTED_E280_SHA256}"
        )

    epoch, step = runner.load_checkpoint_meta(path)
    if epoch != EXPECTED_E280_EPOCH or step != EXPECTED_E280_GLOBAL_STEP:
        raise RuntimeError(
            "E280 checkpoint semantic metadata mismatch: "
            f"epoch={epoch} step={step}; "
            f"expected epoch={EXPECTED_E280_EPOCH} step={EXPECTED_E280_GLOBAL_STEP}"
        )
    return epoch, step, actual_sha


def main() -> None:
    p = argparse.ArgumentParser(
        description="Port the frozen Cori E280 continuation recipe to a Kaggle single-GPU session."
    )
    p.add_argument("--handoff-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--resume-checkpoint", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("/kaggle/working/c3-cori-kaggle-runs"))
    p.add_argument("--matcha-checkout", type=Path, default=Path("/kaggle/working/Matcha-TTS"))
    p.add_argument("--target-max-epochs", type=int, default=DEFAULT_TARGET_EPOCH)
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify GPU, E280 identity, and input paths without starting training.",
    )
    args = p.parse_args()

    gpu_info = print_gpu_preflight()

    handoff = args.handoff_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    resume = args.resume_checkpoint.resolve()
    if not handoff.is_dir():
        raise FileNotFoundError(f"handoff directory missing: {handoff}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root missing: {dataset_root}")

    epoch, step, resume_sha = verify_resume(resume)

    required_handoff = [
        handoff / "cori_matcha_epoch100.ckpt",
        handoff / "HANDOFF_MANIFEST.json",
        handoff / "metadata" / "FREEZE.json",
        handoff / "metadata" / "PREPARED.json",
        handoff / "metadata" / "STATS.json",
        handoff / "filelists" / "train.txt",
        handoff / "filelists" / "valid.txt",
    ]
    missing = [str(path) for path in required_handoff if not path.is_file()]
    if missing:
        raise FileNotFoundError("handoff bundle incomplete: " + ", ".join(missing))

    preflight = {
        "schema": "c3-cori-kaggle-preflight-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_epoch": epoch,
        "resume_global_step": step,
        "resume_sha256": resume_sha,
        "target_epoch": int(args.target_max_epochs),
        "batch_size_frozen": runner.BATCH_SIZE,
        "gpu": gpu_info,
        "handoff_dir": str(handoff),
        "dataset_root": str(dataset_root),
        "resume_checkpoint": str(resume),
    }
    print("C3_KAGGLE_PREFLIGHT_PASS")
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)

    if args.preflight_only:
        return

    if int(args.target_max_epochs) != DEFAULT_TARGET_EPOCH:
        raise RuntimeError(
            f"first Kaggle benchmark is frozen to E280->E{DEFAULT_TARGET_EPOCH}; "
            "do not extend until the benchmark is reviewed"
        )

    stable_run_dir = args.output_root.resolve() / "cori-e100-to-e550-b16"
    import_dir = stable_run_dir / "imported_resume"
    import_dir.mkdir(parents=True, exist_ok=True)
    imported = import_dir / f"resume_e{epoch:03d}_s{step}.ckpt"
    if not imported.exists() or runner.sha256_file(imported) != resume_sha:
        shutil.copy2(resume, imported)
    if runner.sha256_file(imported) != resume_sha:
        raise RuntimeError("imported E280 checkpoint SHA-256 mismatch after copy")

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("lightning_cori_continue_to550.py")),
        "--handoff-dir",
        str(handoff),
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(args.output_root.resolve()),
        "--matcha-checkout",
        str(args.matcha_checkout.resolve()),
        "--target-max-epochs",
        str(DEFAULT_TARGET_EPOCH),
    ]
    print("C3_KAGGLE_BENCHMARK_BEGIN", flush=True)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("C3_KAGGLE_BENCHMARK_PASS", flush=True)


if __name__ == "__main__":
    main()
