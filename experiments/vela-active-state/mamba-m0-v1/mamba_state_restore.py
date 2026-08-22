from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path

import torch

MODEL_ID = "state-spaces/mamba-130m-hf"


def _write_report(report: dict) -> None:
    result_path = os.environ.get("VELA_RESULT_PATH")
    if result_path:
        p = Path(result_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

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
            cache = getattr(pre, "cache_params", None)
            if cache is None:
                cache = getattr(pre, "past_key_values", None)
            if cache is None:
                raise RuntimeError(
                    f"Mamba model returned no causal cache; output type={type(pre).__name__}, keys={list(pre.keys())}"
                )

            cache_arg = "cache_params" if getattr(pre, "cache_params", None) is not None else "past_key_values"
            # Transformers 4.57-era Mamba requires a nonzero cache_position when a populated cache is passed.
            # For a one-token continuation, the absolute prefix length is the correct manual decode position.
            cache_position = torch.tensor([int(first.shape[1])], dtype=torch.long, device=device)

            with tempfile.TemporaryDirectory() as td:
                cp = Path(td) / "mamba_cache.pt"
                torch.save(cache, cp)

                native_kwargs = {cache_arg: cache}
                if cache_arg == "cache_params":
                    native_kwargs["cache_position"] = cache_position
                native = model(
                    second,
                    **native_kwargs,
                    use_cache=True,
                    return_dict=True,
                ).logits.detach().float().cpu()

                restored_cache = torch.load(cp, map_location="cpu", weights_only=False)
                restored_kwargs = {cache_arg: restored_cache}
                if cache_arg == "cache_params":
                    restored_kwargs["cache_position"] = cache_position.clone()
                restored = model(
                    second,
                    **restored_kwargs,
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
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "prefix_tokens": int(first.shape[1]),
            "continuation_tokens": int(second.shape[1]),
            "cache_type": type(cache).__name__,
            "cache_argument": cache_arg,
            "cache_position": cache_position.cpu().tolist(),
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
        _write_report(report)
        if not (report["restore_equivalent"] and report["fresh_is_different"]):
            raise SystemExit(1)
        return report
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            raise
        report = {
            "status": "M0_PROBE_ERROR",
            "model": MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-18:],
            "claim_boundary": "Experiment/runtime failure only; do not interpret as a Mamba architecture failure.",
        }
        _write_report(report)
        raise


if __name__ == "__main__":
    run()
