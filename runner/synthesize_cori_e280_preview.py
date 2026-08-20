from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import modal

APP_NAME = "c3-cori-e280-preview-cpu"
VOLUME_NAME = "c3-speech-en-v1"
VOLUME_MOUNT = Path("/vol")
MATCHA_CHECKOUT = Path("/opt/Matcha-TTS")
BIGVGAN_CHECKOUT = Path("/opt/BigVGAN")
MATCHA_COMMIT = "bd4d90d93214b37f7a159cf205ae85762c2c10aa"
EXPECTED_CLEANER_SHA256 = "8f42115f89560604caf0643f270f930d6d9ef462cf55e51e8a9ffc0a7c1c962a"
SAMPLE_RATE = 22050

E280 = Path("/vol/training/cori/lightning_e280/checkpoint_epoch=279.ckpt")
E280_SHA256 = "081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9"
ADAPT_RUN = Path("/vol/training/cori/vocoder_adaptation/bigvgan_base_cori_22k80/20260817T022729Z")

EVAL_ITEMS = [
    {"id": "S01", "text": "The small lamp beside the window is still on."},
    {"id": "P01", "text": "After the rain stopped, the street grew quiet, and the clouds began to break."},
    {"id": "L02", "text": "The experiment seemed simple at first, but once we repeated it under the same conditions, small differences appeared in the timing, the rhythm, and the way each sentence ended."},
    {"id": "D01", "text": "The sixth street shuttle stopped beside three freshly painted shops."},
    {"id": "E04", "text": "No one noticed the difference until the recording ended."},
]

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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _install_c3_matcha_text_patch() -> str:
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
    return _sha256_file(cleaners_path)


def _wav_bytes(audio, sf) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_24")
    return buf.getvalue()


@app.function(
    image=image,
    volumes={str(VOLUME_MOUNT): volume},
    cpu=16,
    memory=32768,
    timeout=5400,
)
def synthesize() -> dict:
    volume.reload()
    if not E280.is_file():
        raise RuntimeError(f"missing E280 checkpoint: {E280}")
    actual = _sha256_file(E280)
    if actual != E280_SHA256:
        raise RuntimeError(f"E280 SHA mismatch: {actual} != {E280_SHA256}")

    cleaner_sha = _install_c3_matcha_text_patch()
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

    model = load_matcha("c3_cori_e280_preview", str(E280), device)
    rendered = []
    metrics = []
    for i, item in enumerate(EVAL_ITEMS, start=1):
        torch.manual_seed(1000 + i)
        processed = process_text(i, item["text"], device)
        with torch.inference_mode():
            output = model.synthesise(
                processed["x"], processed["x_lengths"], n_timesteps=10,
                temperature=0.667, spks=None, length_scale=1.0,
            )
            waveform = vocoder(output["mel"].to(device)).squeeze().detach().cpu().float().numpy().astype(np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise RuntimeError(f"invalid E280 waveform for {item['id']}: {waveform.shape}")
        peak = float(np.max(np.abs(waveform)))
        if peak > 0.99:
            waveform = waveform / peak * 0.99
        rendered.append((item, waveform))
        metrics.append({
            "id": item["id"],
            "duration_seconds": float(waveform.size / SAMPLE_RATE),
            "peak_abs": float(np.max(np.abs(waveform))),
            "rms": float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))),
        })

    silence = np.zeros(int(0.8 * SAMPLE_RATE), dtype=np.float32)
    preview_parts = []
    for _item, audio in rendered:
        preview_parts.extend([audio, silence])
    preview = np.concatenate(preview_parts[:-1])

    pack = io.BytesIO()
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for item, audio in rendered:
            zf.writestr(f"{item['id']}.wav", _wav_bytes(audio, sf))
            zf.writestr(f"{item['id']}.txt", item["text"] + "\n")
        zf.writestr("preview_E280.wav", _wav_bytes(preview, sf))
        zf.writestr("metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
        zf.writestr(
            "README.txt",
            "Cori Matcha E280 descriptive preview.\n"
            "Checkpoint: epoch 280 completed / global_step 140560.\n"
            "Vocoder: frozen Cori-adapted BigVGAN 22k/80 mel.\n"
            "Inference: n_timesteps=10, temperature=0.667, length_scale=1.0.\n"
            "No EQ, reverb, compressor, pause surgery, or ending-duration patch.\n"
            "This is not a formal milestone evaluation.\n",
        )

    return {
        "ok": True,
        "gpu_allocated": False,
        "checkpoint_sha256": actual,
        "epoch_completed": 280,
        "global_step": 140560,
        "vocoder_run": str(ADAPT_RUN),
        "items": len(rendered),
        "pack_bytes": pack.getvalue(),
        "preview_bytes": _wav_bytes(preview, sf),
        "metrics": metrics,
    }


@app.local_entrypoint()
def main(output_dir: str = "cori_e280_preview") -> None:
    result = synthesize.remote()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cori-e280-preview.zip").write_bytes(result.pop("pack_bytes"))
    (out / "preview_E280.wav").write_bytes(result.pop("preview_bytes"))
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
