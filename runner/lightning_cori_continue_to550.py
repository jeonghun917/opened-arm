from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_CHECKPOINT_SHA256 = "f4409103780820e356b609ec79c425cb1cffd3059fed163e1f60bfe926438273"
MATCHA_REPO = "https://github.com/shivammehta25/Matcha-TTS.git"
MATCHA_COMMIT = "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
MATCHA_TEXT_PATCH_VERSION = "c3-strip-unsupported-combining-v1"
MATCHA_TEXT_PATCH_MARKER = "# C3_MATCHA_TEXT_PATCH_V1"
EXPECTED_ROWS = 8646
SOURCE_DATASET_PREFIX = "/vol/datasets/cori/arm-31926500912"
BATCH_SIZE = 16
BASELINE_TOTAL_EPOCHS = 100
FINAL_TARGET_EPOCHS = 550
SEGMENT_EPOCHS = 10
CHECKPOINT_EVERY_EPOCHS = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint_meta(path: Path) -> tuple[int, int]:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    internal_epoch = int(payload.get("epoch", -1))
    global_step = int(payload.get("global_step", -1))
    return internal_epoch + 1, global_step


def prepare_matcha(checkout: Path) -> None:
    if not checkout.exists():
        run(["git", "clone", MATCHA_REPO, str(checkout)])
    run(["git", "fetch", "--all", "--tags"], cwd=checkout)
    run(["git", "checkout", "--detach", MATCHA_COMMIT], cwd=checkout)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if actual != MATCHA_COMMIT:
        raise RuntimeError(f"Matcha commit mismatch: {actual}")


def install_c3_matcha_text_patch(checkout: Path) -> str:
    cleaners_path = checkout / "matcha" / "text" / "cleaners.py"
    source = cleaners_path.read_text(encoding="utf-8")
    if MATCHA_TEXT_PATCH_MARKER in source:
        return sha256_file(cleaners_path)
    if "import unicodedata\n" not in source:
        source = source.replace("import logging\nimport re\n", "import logging\nimport re\nimport unicodedata\n", 1)
    needle = "    phonemes = collapse_whitespace(phonemes)\n    return phonemes\n"
    replacement = """    phonemes = collapse_whitespace(phonemes)\n    # C3_MATCHA_TEXT_PATCH_V1\n    from matcha.text.symbols import symbols as _c3_symbols\n    _c3_allowed = set(_c3_symbols)\n    _c3_unknown_noncombining = sorted(\n        {ch for ch in phonemes if ch not in _c3_allowed and not unicodedata.combining(ch)}\n    )\n    if _c3_unknown_noncombining:\n        raise ValueError(\n            f\"unsupported non-combining Matcha symbols: {_c3_unknown_noncombining}\"\n        )\n    phonemes = \"\".join(\n        ch for ch in phonemes if ch in _c3_allowed or not unicodedata.combining(ch)\n    )\n    return phonemes\n"""
    if needle not in source:
        raise RuntimeError("could not locate english_cleaners2 return block for C3 patch")
    cleaners_path.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
    return sha256_file(cleaners_path)


def rewrite_filelist(src: Path, dst: Path, dataset_root: Path) -> int:
    dataset_root = dataset_root.resolve()
    rows = 0
    out_lines: list[str] = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("|", 1)
        if len(parts) != 2:
            raise RuntimeError(f"unexpected filelist row: {raw[:120]!r}")
        old_audio, rest = parts
        if not old_audio.startswith(SOURCE_DATASET_PREFIX + "/"):
            raise RuntimeError(f"audio path outside frozen Cori dataset root: {old_audio}")
        relative = old_audio[len(SOURCE_DATASET_PREFIX) + 1 :]
        new_audio = dataset_root / relative
        if not new_audio.is_file():
            raise RuntimeError(f"Lightning dataset audio missing: {new_audio}")
        out_lines.append(f"{new_audio}|{rest}")
        rows += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return rows


def render_data_yaml(train_filelist: Path, valid_filelist: Path, stats: dict) -> str:
    return f'''_target_: matcha.data.text_mel_datamodule.TextMelDataModule
name: c3_cori_lightning
train_filelist_path: {train_filelist}
valid_filelist_path: {valid_filelist}
batch_size: {BATCH_SIZE}
num_workers: 8
pin_memory: True
cleaners: [english_cleaners2]
add_blank: True
n_spks: 1
n_fft: 1024
n_feats: 80
sample_rate: 22050
hop_length: 256
win_length: 1024
f_min: 0
f_max: 8000
data_statistics:
  mel_mean: {float(stats["mel_mean"]):.12f}
  mel_std: {float(stats["mel_std"]):.12f}
seed: ${{seed}}
load_durations: false
'''


def newest_resume_checkpoint(run_dir: Path, baseline: Path) -> tuple[Path, int, int]:
    candidates: list[tuple[int, int, Path]] = []
    if run_dir.exists():
        for path in run_dir.rglob("*.ckpt"):
            try:
                total_epoch, global_step = load_checkpoint_meta(path)
            except Exception as exc:
                print(f"SKIP_BAD_CKPT path={path} error={exc!r}", flush=True)
                continue
            if total_epoch >= BASELINE_TOTAL_EPOCHS and global_step >= 0:
                candidates.append((global_step, total_epoch, path))
    if candidates:
        global_step, total_epoch, path = max(candidates, key=lambda item: (item[0], item[1]))
        return path.resolve(), total_epoch, global_step
    total_epoch, global_step = load_checkpoint_meta(baseline)
    if total_epoch != BASELINE_TOTAL_EPOCHS:
        raise RuntimeError(f"baseline semantic epoch mismatch: {total_epoch}")
    return baseline.resolve(), total_epoch, global_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path.home() / "c3-cori-lightning-runs")
    parser.add_argument("--matcha-checkout", type=Path, default=Path.home() / "src" / "Matcha-TTS")
    parser.add_argument("--target-max-epochs", type=int, default=BASELINE_TOTAL_EPOCHS + SEGMENT_EPOCHS)
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for continuation training")
    target = int(args.target_max_epochs)
    if target < BASELINE_TOTAL_EPOCHS + SEGMENT_EPOCHS or target > FINAL_TARGET_EPOCHS:
        raise RuntimeError(f"target-max-epochs must be between 110 and {FINAL_TARGET_EPOCHS}")
    if target % SEGMENT_EPOCHS != 0:
        raise RuntimeError(f"target-max-epochs must be a multiple of {SEGMENT_EPOCHS}")

    handoff = args.handoff_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    baseline_checkpoint = handoff / "cori_matcha_epoch100.ckpt"
    manifest = load_json(handoff / "HANDOFF_MANIFEST.json")
    freeze = load_json(handoff / "metadata" / "FREEZE.json")
    prepared = load_json(handoff / "metadata" / "PREPARED.json")
    stats = load_json(handoff / "metadata" / "STATS.json")

    if sha256_file(baseline_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("epoch100 checkpoint SHA256 mismatch")
    if manifest.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("handoff manifest checkpoint identity mismatch")
    if int(prepared.get("rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("PREPARED row count mismatch")
    if freeze.get("canonical_dataset_freeze") is not True:
        raise RuntimeError("dataset is not marked canonical frozen")
    if freeze.get("matcha_upstream_commit") != MATCHA_COMMIT:
        raise RuntimeError("frozen Matcha commit mismatch")
    if freeze.get("matcha_text_patch_version") != MATCHA_TEXT_PATCH_VERSION:
        raise RuntimeError("frozen text patch version mismatch")

    prepare_matcha(args.matcha_checkout)
    patched_sha = install_c3_matcha_text_patch(args.matcha_checkout)
    expected_cleaner_sha = str(stats.get("matcha_patched_cleaners_sha256", ""))
    if not expected_cleaner_sha or patched_sha != expected_cleaner_sha:
        raise RuntimeError(f"patched cleaner SHA mismatch: actual={patched_sha} expected={expected_cleaner_sha}")

    run_dir = args.output_root.resolve() / "cori-e100-to-e550-b16"
    run_dir.mkdir(parents=True, exist_ok=True)
    milestones_dir = run_dir / "milestones"
    milestones_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint, resume_total_epoch, resume_global_step = newest_resume_checkpoint(run_dir, baseline_checkpoint)
    print(f"C3_RESUME checkpoint={resume_checkpoint} total_epoch={resume_total_epoch} global_step={resume_global_step} target={target}", flush=True)
    if resume_total_epoch >= target:
        print(f"C3_SEGMENT_ALREADY_COMPLETE target={target} resume_epoch={resume_total_epoch}")
        return
    if target - resume_total_epoch > SEGMENT_EPOCHS:
        raise RuntimeError(f"refusing segment longer than {SEGMENT_EPOCHS} epochs: resume={resume_total_epoch} target={target}")

    train_filelist = run_dir / "filelists" / "train.txt"
    valid_filelist = run_dir / "filelists" / "valid.txt"
    train_rows = rewrite_filelist(handoff / "filelists" / "train.txt", train_filelist, dataset_root)
    valid_rows = rewrite_filelist(handoff / "filelists" / "valid.txt", valid_filelist, dataset_root)
    if train_rows + valid_rows != EXPECTED_ROWS:
        raise RuntimeError(f"rewritten split row total mismatch: {train_rows}+{valid_rows}")

    data_cfg = args.matcha_checkout / "configs" / "data" / "c3_cori_lightning.yaml"
    data_cfg.write_text(render_data_yaml(train_filelist, valid_filelist, stats), encoding="utf-8")

    segment_started = datetime.now(timezone.utc).isoformat()
    provenance = {
        "schema": "c3-cori-lightning-continuation-v2-restart-safe",
        "created_at_utc": segment_started,
        "speaker": "Cori Samuel",
        "role": "Tutor-F",
        "baseline_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "resume_checkpoint": str(resume_checkpoint),
        "resume_checkpoint_sha256": sha256_file(resume_checkpoint),
        "resume_semantic_epoch_total": resume_total_epoch,
        "resume_global_step": resume_global_step,
        "segment_target_total_epochs": target,
        "final_experiment_target_total_epochs": FINAL_TARGET_EPOCHS,
        "segment_epochs_max": SEGMENT_EPOCHS,
        "batch_size": BATCH_SIZE,
        "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "matcha_commit": MATCHA_COMMIT,
        "text_patch_version": MATCHA_TEXT_PATCH_VERSION,
        "patched_cleaner_sha256": patched_sha,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "dataset_root": str(dataset_root),
        "vocoder_not_trained_here": True,
        "comparison_vocoder": "Cori-adapted BigVGAN",
        "restart_contract": "10-epoch max segment; checkpoint callback every epoch; persistent Studio home; auto-resume highest global_step",
    }
    (run_dir / f"SEGMENT_E{target:03d}_START.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env["PROJECT_ROOT"] = str(args.matcha_checkout)
    env["PYTHONUNBUFFERED"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"

    command = [
        sys.executable, "-m", "matcha.train",
        "data=c3_cori_lightning", "run_name=c3-cori-lightning-continuation", "test=false",
        f"data.batch_size={BATCH_SIZE}", f"trainer.max_epochs={target}",
        f"callbacks.model_checkpoint.every_n_epochs={CHECKPOINT_EVERY_EPOCHS}",
        "callbacks.model_checkpoint.save_top_k=1", "++callbacks.model_checkpoint.save_last=true",
        f"hydra.run.dir={run_dir}", f"ckpt_path={resume_checkpoint}",
    ]
    run(command, cwd=args.matcha_checkout, env=env)

    latest_checkpoint, latest_total_epoch, latest_global_step = newest_resume_checkpoint(run_dir, baseline_checkpoint)
    if latest_total_epoch < target:
        raise RuntimeError(f"segment ended below target: latest_epoch={latest_total_epoch} target={target}")

    milestone = milestones_dir / f"cori_matcha_e{target:03d}.ckpt"
    if latest_checkpoint.resolve() != milestone.resolve():
        shutil.copy2(latest_checkpoint, milestone)
    milestone_epoch, milestone_step = load_checkpoint_meta(milestone)
    if milestone_epoch < target:
        raise RuntimeError(f"milestone epoch mismatch: {milestone_epoch} < {target}")

    result = {
        **provenance,
        "ok": True,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "milestone_checkpoint": str(milestone),
        "milestone_checkpoint_sha256": sha256_file(milestone),
        "milestone_semantic_epoch": milestone_epoch,
        "milestone_global_step": milestone_step,
    }
    (run_dir / f"SEGMENT_E{target:03d}_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("C3_RESTART_SAFE_SEGMENT_PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
