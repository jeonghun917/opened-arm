from __future__ import annotations

import json
import math
import os
import tempfile
import traceback
from pathlib import Path
from typing import Iterable

import torch

MODEL_ID = "state-spaces/mamba-130m-hf"
SCALES = [0.0, 1e-4, 1e-3, 1e-2]


def write_report(report: dict) -> None:
    path = os.environ.get("VELA_RESULT_PATH")
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def iter_tensors(obj) -> Iterable[torch.Tensor]:
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_tensors(value)
    elif hasattr(obj, "__dict__"):
        for value in vars(obj).values():
            yield from iter_tensors(value)


def cache_signature(cache) -> dict:
    tensors = [t.detach().float().cpu() for t in iter_tensors(cache)]
    total_numel = sum(t.numel() for t in tensors)
    sq = sum(float((t * t).sum()) for t in tensors)
    return {
        "tensor_count": len(tensors),
        "numel": int(total_numel),
        "l2": math.sqrt(sq),
        "shapes": [list(t.shape) for t in tensors[:8]],
    }


def cache_distance(a, b) -> dict:
    ta = [t.detach().float().cpu() for t in iter_tensors(a)]
    tb = [t.detach().float().cpu() for t in iter_tensors(b)]
    if len(ta) != len(tb):
        return {"comparable": False, "reason": "tensor_count_mismatch", "a": len(ta), "b": len(tb)}
    sq = 0.0
    max_abs = 0.0
    numel = 0
    for xa, xb in zip(ta, tb):
        if xa.shape != xb.shape:
            return {"comparable": False, "reason": "shape_mismatch", "a": list(xa.shape), "b": list(xb.shape)}
        d = xa - xb
        sq += float((d * d).sum())
        max_abs = max(max_abs, float(d.abs().max()))
        numel += d.numel()
    return {
        "comparable": True,
        "l2": math.sqrt(sq),
        "rms": math.sqrt(sq / max(numel, 1)),
        "max_abs": max_abs,
        "numel": int(numel),
    }


def continue_one(model, token, cache, cache_position: int):
    pos = torch.tensor([cache_position], dtype=torch.long, device=token.device)
    out = model(
        token,
        cache_params=cache,
        cache_position=pos,
        use_cache=True,
        return_dict=True,
    )
    return out.logits[:, -1].detach().float().cpu(), out.cache_params


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()

        text = (
            "VELA preserves a live hypothesis across an engine upgrade. "
            "The next computation should depend on the pre-upgrade causal state."
        )
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if ids.shape[1] < 4:
            raise RuntimeError("unexpectedly short tokenization")
        prefix = ids[:, :-1]
        continuation = ids[:, -1:]
        prefix_len = int(prefix.shape[1])

        target = model.backbone.layers[0].mixer.x_proj.weight
        original = target.detach().clone()

        with torch.no_grad(), tempfile.TemporaryDirectory() as td:
            # W1 active state S1.
            target.copy_(original)
            old_pre = model(prefix, use_cache=True, return_dict=True)
            old_cache = old_pre.cache_params
            old_path = Path(td) / "old_cache.pt"
            torch.save(old_cache, old_path)
            old_sig = cache_signature(old_cache)

            rows = []
            for scale in SCALES:
                # Controlled same-architecture weight update W1 -> W2.
                target.copy_(original * (1.0 + scale))

                # Gold reference: replay prefix under W2 to obtain W2-native state S2.
                new_pre = model(prefix, use_cache=True, return_dict=True)
                new_cache_pre = new_pre.cache_params
                new_sig = cache_signature(new_cache_pre)

                # Measure pre-continuation S1 vs S2 before either is mutated.
                imported_cache = torch.load(old_path, map_location="cpu", weights_only=False)
                state_distance = cache_distance(imported_cache, new_cache_pre)

                # Both continuations use identical W2. The only difference is state provenance.
                new_native_logits, _ = continue_one(model, continuation, new_cache_pre, prefix_len)
                imported_logits, _ = continue_one(model, continuation, imported_cache, prefix_len)

                logit_diff = new_native_logits - imported_logits
                rows.append(
                    {
                        "weight_scale_delta": scale,
                        "new_state_signature": new_sig,
                        "old_vs_new_prefill_state": state_distance,
                        "new_native_vs_imported_state_logits": {
                            "max_abs": float(logit_diff.abs().max()),
                            "rms": float(torch.sqrt(torch.mean(logit_diff * logit_diff))),
                            "argmax_same": bool(torch.argmax(new_native_logits, dim=-1).item() == torch.argmax(imported_logits, dim=-1).item()),
                        },
                    }
                )

            target.copy_(original)

        control = rows[0]
        report = {
            "status": "WEIGHT_UPDATE_STATE_COMPAT_PROBE",
            "model": MODEL_ID,
            "device": device,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "prefix_tokens": prefix_len,
            "continuation_tokens": int(continuation.shape[1]),
            "updated_parameter": "backbone.layers[0].mixer.x_proj.weight",
            "old_state_signature": old_sig,
            "rows": rows,
            "control_exact": control["new_native_vs_imported_state_logits"]["max_abs"] <= 1e-5,
            "interpretation": (
                "For each nonzero row, W2-native continuation and W1-state-import continuation use the same updated weights W2. "
                "Their difference therefore isolates incompatibility between the old active state and the state W2 would have produced by replaying the same prefix."
            ),
            "claim_boundary": (
                "Controlled parameter perturbation, not capability training. This measures direct active-state portability across a same-architecture weight update; "
                "it does not show that a learned upgrade improves reasoning or that a state mapper is impossible."
            ),
        }
        write_report(report)
        if not report["control_exact"]:
            raise SystemExit(1)
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            raise
        report = {
            "status": "WEIGHT_UPDATE_STATE_COMPAT_ERROR",
            "model": MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-24:],
            "claim_boundary": "Experiment/runtime failure only; no architecture verdict.",
        }
        write_report(report)
        raise


if __name__ == "__main__":
    run()
