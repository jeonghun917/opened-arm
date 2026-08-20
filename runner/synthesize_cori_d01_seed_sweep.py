from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import modal

APP_NAME = "c3-cori-d01-seed-sweep-cpu"
VOLUME_NAME = "c3-speech-en-v1"
VOLUME_MOUNT = Path("/vol")
MATCHA_CHECKOUT = Path("/opt/Matcha-TTS")
BIGVGAN_CHECKOUT = Path("/opt/BigVGAN")
MATCHA_COMMIT = "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
EXPECTED_CLEANER_SHA256 = "8f42115f89560604caf0643f270f930d6d9ef462cf55e51e8a9ffc0a7c1c962a"
SAMPLE_RATE = 22050
D01_TEXT = "The sixth street shuttle stopped beside three freshly painted shops."
SEEDS = [1004, 2004, 3004, 4004, 5004]

CHECKPOINTS = {
    "E200": {
        "path": Path("/vol/training/cori/diagnostics/d01_seed_sweep/E200/checkpoint_epoch=199.ckpt"),
        "sha256": "b3235e8bff23c6241119add85e57dccfa1e88ed2cf2ed51bed8a3c305dee5c54",
        "epoch": 200,
        "global_step": 100400,
    },
    "E280": {
        "path": Path("/vol/training/cori/diagnostics/d01_seed_sweep/E280/checkpoint_epoch=279.ckpt"),
        "sha256": "081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9",
        "epoch": 280,
        "global_step": 140560,
    },
}

ADAPT_RUN = Path("/vol/training/cori/vocoder_adaptation/bigvgan_base_cori_22k80/20260817T022729Z")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "espeak-ng", "ffmpeg", "libsndfile1", "ca-certificates")
    .pip_install(
        "numpy<2", "Cython", "torch==2.5.1", "torchaudio==2.5.1", "torchvision==0.20.1",
        "matplotlib<3.10", "huggingface_hub>=0.23,<1", "soundfile", "librosa==0.10.2.post1",
        "scipy", "pyyaml", "einops", "tqdm", "ninja",
    )
    .run_commands(
        "git clone https://github.com/shivammehta25/Matcha-TTS.git /opt/Matcha-TTS",
        f"cd /opt/Matcha-TTS && git checkout {MATCHA_COMMIT}",
        "cd /opt/Matcha-TTS && python -m pip install -e .",
        "git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git /opt/BigVGAN",
    )
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def install_c3_matcha_text_patch() -> str:
    cleaners_path = MATCHA_CHECKOUT / "matcha" / "text" / "cleaners.py"
    source = cleaners_path.read_text(encoding="utf-8")
    marker = "# C3_MATCHA_TEXT_PATCH_V1"
    if marker not in source:
        if "import unicodedata\n" not in source:
            source = source.replace("import logging\nimport re\n", "import logging\nimport re\nimport unicodedata\n", 1)
        needle = "    phonemes = collapse_whitespace(phonemes)\n    return phonemes\n"
        replacement = """    phonemes = collapse_whitespace(phonemes)\n    # C3_MATCHA_TEXT_PATCH_V1\n    from matcha.text.symbols import symbols as _c3_symbols\n    _c3_allowed = set(_c3_symbols)\n    _c3_unknown_noncombining = sorted(\n        {ch for ch in phonemes if ch not in _c3_allowed and not unicodedata.combining(ch)}\n    )\n    if _c3_unknown_noncombining:\n        raise ValueError(\n            f\"unsupported non-combining Matcha symbols: {_c3_unknown_noncombining}\"\n        )\n    phonemes = \"\".join(\n        ch for ch in phonemes if ch in _c3_allowed or not unicodedata.combining(ch)\n    )\n    return phonemes\n"""
        if needle not in source:
            raise RuntimeError("could not locate english_cleaners2 return block for C3 patch")
        cleaners_path.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
    return sha256_file(cleaners_path)


def wav_bytes(audio, sf) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_24")
    return buf.getvalue()


def waveform_metrics(waveform, np) -> dict:
    x = waveform.astype(np.float64)
    n_fft = 1024
    hop = 256
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    window = np.hanning(n_fft)
    frames = []
    for start in range(0, max(1, x.size - n_fft + 1), hop):
        frame = x[start:start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        frames.append(frame * window)
    spec = np.abs(np.fft.rfft(np.stack(frames), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
    total = spec.sum(axis=1) + 1e-12
    centroid = (spec * freqs[None, :]).sum(axis=1) / total
    frame_rms = np.sqrt(np.mean(np.stack(frames) ** 2, axis=1))
    active = frame_rms > max(1e-5, float(np.percentile(frame_rms, 20)) * 1.5)
    if not np.any(active):
        active = np.ones_like(frame_rms, dtype=bool)
    hf = spec[:, freqs >= 6000.0].sum(axis=1)
    hf_ratio = hf / total
    return {
        "duration_seconds": float(waveform.size / SAMPLE_RATE),
        "peak_abs": float(np.max(np.abs(waveform))),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "spectral_centroid_mean_hz_active": float(np.mean(centroid[active])),
        "spectral_centroid_p95_hz_active": float(np.percentile(centroid[active], 95)),
        "centroid_gt_4khz_active_fraction": float(np.mean(centroid[active] > 4000.0)),
        "hf_energy_ratio_gt6khz_active_mean": float(np.mean(hf_ratio[active])),
        "hf_energy_ratio_gt6khz_active_p95": float(np.percentile(hf_ratio[active], 95)),
    }


def mel_metrics(mel, np) -> dict:
    arr = mel.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    top_n = max(1, arr.shape[0] // 5)
    top = arr[-top_n:, :]
    return {
        "mel_bins": int(arr.shape[0]),
        "mel_frames": int(arr.shape[1]),
        "mel_mean": float(np.mean(arr)),
        "mel_std": float(np.std(arr)),
        "mel_top20_mean": float(np.mean(top)),
        "mel_top20_std": float(np.std(top)),
        "mel_top20_minus_all_mean": float(np.mean(top) - np.mean(arr)),
        "mel_p01": float(np.percentile(arr, 1)),
        "mel_p99": float(np.percentile(arr, 99)),
    }


@app.function(image=image, volumes={str(VOLUME_MOUNT): volume}, cpu=16, memory=32768, timeout=5400)
def synthesize() -> dict:
    volume.reload()
    for label, meta in CHECKPOINTS.items():
        if not meta["path"].is_file():
            raise RuntimeError(f"missing {label} checkpoint: {meta['path']}")
        actual = sha256_file(meta["path"])
        if actual != meta["sha256"]:
            raise RuntimeError(f"{label} SHA mismatch: {actual} != {meta['sha256']}")

    cleaner_sha = install_c3_matcha_text_patch()
    if cleaner_sha != EXPECTED_CLEANER_SHA256:
        raise RuntimeError(f"cleaner SHA mismatch: {cleaner_sha} != {EXPECTED_CLEANER_SHA256}")

    generator_path = ADAPT_RUN / "generator_final.pt"
    config_path = ADAPT_RUN / "checkpoints" / "config.json"
    if not generator_path.is_file() or not config_path.is_file():
        raise RuntimeError(f"missing frozen adapted BigVGAN assets under {ADAPT_RUN}")

    import numpy as np
    import soundfile as sf
    import torch
    from matcha.cli import load_matcha, process_text

    sys.path.insert(0, str(BIGVGAN_CHECKOUT))
    from bigvgan import BigVGAN
    from env import AttrDict

    torch.set_num_threads(16)
    device = torch.device("cpu")
    h = AttrDict(json.loads(config_path.read_text(encoding="utf-8")))
    vocoder = BigVGAN(h, use_cuda_kernel=False).to(device)
    state = torch.load(generator_path, map_location="cpu")
    vocoder.load_state_dict(state["generator"])
    vocoder.remove_weight_norm()
    vocoder.eval()

    models = {
        label: load_matcha(f"c3_cori_{label.lower()}_d01_sweep", str(meta["path"]), device)
        for label, meta in CHECKPOINTS.items()
    }
    processed = process_text(1, D01_TEXT, device)

    rows = []
    audio_entries = []
    for label in ("E200", "E280"):
        model = models[label]
        meta = CHECKPOINTS[label]
        for seed in SEEDS:
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = model.synthesise(
                    processed["x"], processed["x_lengths"], n_timesteps=10,
                    temperature=0.667, spks=None, length_scale=1.0,
                )
                mel = output["mel"].to(device)
                waveform = vocoder(mel).squeeze().detach().cpu().float().numpy().astype(np.float32)
            if waveform.ndim != 1 or waveform.size == 0:
                raise RuntimeError(f"invalid waveform for {label} seed {seed}: {waveform.shape}")
            peak = float(np.max(np.abs(waveform)))
            if peak > 0.99:
                waveform = waveform / peak * 0.99
            row = {
                "checkpoint": label,
                "epoch_completed": meta["epoch"],
                "global_step": meta["global_step"],
                "checkpoint_sha256": meta["sha256"],
                "seed": seed,
                "prompt_id": "D01",
                "text": D01_TEXT,
                "n_timesteps": 10,
                "temperature": 0.667,
                "length_scale": 1.0,
                "vocoder_run": str(ADAPT_RUN),
            }
            row.update(waveform_metrics(waveform, np))
            row.update(mel_metrics(mel, np))
            rows.append(row)
            audio_entries.append((label, seed, waveform))

    fieldnames = list(rows[0].keys())
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    summary = {}
    numeric_keys = [
        "duration_seconds", "peak_abs", "rms", "spectral_centroid_mean_hz_active",
        "spectral_centroid_p95_hz_active", "centroid_gt_4khz_active_fraction",
        "hf_energy_ratio_gt6khz_active_mean", "hf_energy_ratio_gt6khz_active_p95",
        "mel_frames", "mel_mean", "mel_std", "mel_top20_mean", "mel_top20_std",
        "mel_top20_minus_all_mean", "mel_p01", "mel_p99",
    ]
    for label in ("E200", "E280"):
        subset = [r for r in rows if r["checkpoint"] == label]
        summary[label] = {}
        for key in numeric_keys:
            vals = np.array([float(r[key]) for r in subset], dtype=np.float64)
            summary[label][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    pack = io.BytesIO()
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for label, seed, audio in audio_entries:
            zf.writestr(f"audio/{label}_D01_seed{seed}.wav", wav_bytes(audio, sf))
        zf.writestr("diagnostic_metrics.csv", csv_buf.getvalue())
        zf.writestr("diagnostic_metrics.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        zf.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        zf.writestr(
            "README.txt",
            "Cori D01 descriptive diagnostic seed sweep.\n"
            "Prompt: The sixth street shuttle stopped beside three freshly painted shops.\n"
            "Checkpoints: E200 and E280.\n"
            "Seeds: 1004, 2004, 3004, 4004, 5004. Seed 1004 reproduces the seed used for D01 in the prior E280 five-item preview.\n"
            "Vocoder: exact frozen Cori-adapted BigVGAN.\n"
            "Inference: n_timesteps=10, temperature=0.667, length_scale=1.0.\n"
            "No EQ, reverb, compressor, pause surgery, or ending-duration patch.\n"
            "This is an ad-hoc descriptive diagnostic, not a formal blinded milestone evaluation.\n",
        )

    return {
        "ok": True,
        "gpu_allocated": False,
        "rows": rows,
        "summary": summary,
        "metrics_csv": csv_buf.getvalue(),
        "pack_bytes": pack.getvalue(),
    }


@app.local_entrypoint()
def main(output_dir: str = "cori_d01_seed_sweep") -> None:
    result = synthesize.remote()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cori-d01-seed-sweep.zip").write_bytes(result.pop("pack_bytes"))
    (out / "diagnostic_metrics.csv").write_text(result.pop("metrics_csv"), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
