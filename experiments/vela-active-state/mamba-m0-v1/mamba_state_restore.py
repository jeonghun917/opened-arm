from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

MODEL_ID = "state-spaces/mamba-130m-hf"


def run():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()

    text = (
        "VELA keeps a durable hypothesis and then receives a correction. "
        "The correction is causally active for the next computation."
    )
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.shape[1] < 3:
        raise RuntimeError("unexpectedly short tokenization")
    first = ids[:, :-1]
    second = ids[:, -1:]

    with torch.no_grad():
        pre = model(first, use_cache=True, return_dict=True)
        cache = pre.cache_params
        if cache is None:
            raise RuntimeError("Mamba model returned no causal cache")

        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "mamba_cache.pt"
            # Serialize before native continuation mutates the live cache.
            torch.save(cache, cp)
            native = model(
                second,
                cache_params=cache,
                use_cache=True,
                return_dict=True,
            ).logits.detach().float().cpu()
            restored_cache = torch.load(cp, map_location="cpu", weights_only=False)
            restored = model(
                second,
                cache_params=restored_cache,
                use_cache=True,
                return_dict=True,
            ).logits.detach().float().cpu()

        fresh = model(second, use_cache=True, return_dict=True).logits.detach().float().cpu()

    restore_diff = float((native - restored).abs().max())
    fresh_diff = float((native - fresh).abs().max())
    report = {
        "status": "M0_STATE_CHECKPOINT_PROBE",
        "model": MODEL_ID,
        "device": device,
        "prefix_tokens": int(first.shape[1]),
        "continuation_tokens": int(second.shape[1]),
        "cache_type": type(cache).__name__,
        "restore_max_abs_diff": restore_diff,
        "fresh_max_abs_diff": fresh_diff,
        "restore_equivalent": restore_diff <= 1e-5,
        "fresh_is_different": fresh_diff > 1e-5,
        "interpretation": (
            "This tests whether the Mamba inference cache can be serialized and restored across a fresh Python object "
            "load while preserving the next-token logits at a one-token cached-continuation boundary."
        ),
        "claim_boundary": (
            "Mamba-1 130M family probe only. It does not establish Mamba-3 behavior, VELA continuity, identity, "
            "or cognitive-engine superiority."
        ),
    }
    result_path = os.environ.get("VELA_RESULT_PATH")
    if result_path:
        p = Path(result_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not (report["restore_equivalent"] and report["fresh_is_different"]):
        raise SystemExit(1)


if __name__ == "__main__":
    run()
