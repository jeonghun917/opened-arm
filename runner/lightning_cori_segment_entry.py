from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import lightning_cori_continue_to550 as runner

JOB_PREFIX = "c3-cori-e"


def newest_teamspace_checkpoint(jobs_root: Path) -> tuple[Path | None, int, int]:
    candidates: list[tuple[int, int, Path]] = []
    if not jobs_root.exists():
        return None, -1, -1
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir() or not job_dir.name.startswith(JOB_PREFIX):
            continue
        artifacts = job_dir / "artifacts"
        if not artifacts.exists():
            continue
        for ckpt in artifacts.rglob("*.ckpt"):
            try:
                epoch, step = runner.load_checkpoint_meta(ckpt)
            except Exception as exc:
                print(f"SKIP_TEAMSPACE_CKPT path={ckpt} error={exc!r}", flush=True)
                continue
            if epoch >= runner.BASELINE_TOTAL_EPOCHS and step >= 0:
                candidates.append((step, epoch, ckpt))
    if not candidates:
        return None, -1, -1
    step, epoch, ckpt = max(candidates, key=lambda item: (item[0], item[1]))
    return ckpt.resolve(), epoch, step


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--handoff-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--matcha-checkout", type=Path, required=True)
    p.add_argument("--target-max-epochs", type=int, required=True)
    p.add_argument("--teamspace-jobs-root", type=Path, default=Path("/teamspace/jobs"))
    args = p.parse_args()

    stable_run_dir = args.output_root.resolve() / "cori-e100-to-e550-b16"
    import_dir = stable_run_dir / "imported_resume"
    import_dir.mkdir(parents=True, exist_ok=True)

    prior, epoch, step = newest_teamspace_checkpoint(args.teamspace_jobs_root)
    if prior is not None:
        local = import_dir / f"resume_e{epoch:03d}_s{step}.ckpt"
        if not local.exists() or runner.sha256_file(local) != runner.sha256_file(prior):
            shutil.copy2(prior, local)
        print(f"C3_IMPORTED_TEAMSPACE_RESUME source={prior} local={local} epoch={epoch} step={step}")
    else:
        print("C3_IMPORTED_TEAMSPACE_RESUME none; baseline handoff will be used")

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("lightning_cori_continue_to550.py")),
        "--handoff-dir",
        str(args.handoff_dir),
        "--dataset-root",
        str(args.dataset_root),
        "--output-root",
        str(args.output_root),
        "--matcha-checkout",
        str(args.matcha_checkout),
        "--target-max-epochs",
        str(args.target_max_epochs),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
