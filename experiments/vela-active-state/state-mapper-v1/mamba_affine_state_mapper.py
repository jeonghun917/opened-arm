from __future__ import annotations

import copy
import json
import math
import os
import traceback
from pathlib import Path

import torch

MODEL_ID = "state-spaces/mamba-130m-hf"
WEIGHT_SCALE_DELTA = 0.01
CALIBRATION_PREFIXES = [
    "The active codeword is ALPHA and the previous codeword is retired.",
    "The final color is BLUE after a correction from RED.",
    "Only the Mars record remains in scope for the current task.",
    "Evidence supports hypothesis one while hypothesis two remains unresolved.",
]
TEST_PREFIXES = [
    "The next plan step is commit because evidence collection is complete.",
    "Project Orion is active while Project Vega is explicitly paused.",
    "The latest observation weakens the timing explanation but preserves the scope explanation.",
    "The current constraint forbids external action until verification succeeds.",
]


def write_report(report: dict) -> None:
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def tensor_refs(obj):
    out = []
    seen = set()
    def walk(x):
        if torch.is_tensor(x):
            # Cache objects can expose the same tensor through multiple attributes. Count each object once.
            if id(x) not in seen:
                seen.add(id(x))
                out.append(x)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
        elif hasattr(x, "__dict__"):
            for v in vars(x).values(): walk(v)
    walk(obj)
    return out


def cache_distance(a, b):
    aa, bb = tensor_refs(a), tensor_refs(b)
    if len(aa) != len(bb):
        raise RuntimeError(f"cache tensor count mismatch {len(aa)} != {len(bb)}")
    sq = 0.0
    mx = 0.0
    n = 0
    for x, y in zip(aa, bb):
        if x.shape != y.shape:
            raise RuntimeError(f"cache shape mismatch {x.shape} != {y.shape}")
        d = x.detach().float().cpu() - y.detach().float().cpu()
        sq += float((d*d).sum())
        mx = max(mx, float(d.abs().max()))
        n += d.numel()
    return {"rms": math.sqrt(sq/max(n,1)), "l2": math.sqrt(sq), "max_abs": mx, "numel": int(n)}


def prefill(model, tokenizer, text):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    out = model(ids, use_cache=True, return_dict=True)
    if out.cache_params is None:
        raise RuntimeError("no Mamba cache")
    return ids, out.cache_params


def continue_one(model, token, cache, position):
    pos = torch.tensor([position], dtype=torch.long)
    out = model(token, cache_params=cache, cache_position=pos, use_cache=True, return_dict=True)
    return out.logits[:, -1].detach().float().cpu()


def fit_affine(old_caches, new_caches):
    old_refs = [tensor_refs(c) for c in old_caches]
    new_refs = [tensor_refs(c) for c in new_caches]
    width = len(old_refs[0])
    if any(len(x) != width for x in old_refs + new_refs):
        raise RuntimeError("inconsistent cache tensor count")
    coeffs = []
    for idx in range(width):
        n = 0
        sx = sy = sxx = sxy = 0.0
        for oo, nn in zip(old_refs, new_refs):
            x = oo[idx].detach().double().cpu().reshape(-1)
            y = nn[idx].detach().double().cpu().reshape(-1)
            if x.shape != y.shape:
                raise RuntimeError("calibration tensor shape mismatch")
            n += x.numel()
            sx += float(x.sum()); sy += float(y.sum())
            sxx += float((x*x).sum()); sxy += float((x*y).sum())
        mean_x = sx / max(n,1); mean_y = sy / max(n,1)
        var_num = sxx - sx*sx/max(n,1)
        cov_num = sxy - sx*sy/max(n,1)
        if abs(var_num) < 1e-24:
            a = 1.0
            b = mean_y - mean_x
        else:
            a = cov_num / var_num
            b = mean_y - a*mean_x
        coeffs.append({"a": a, "b": b, "n": int(n)})
    return coeffs


def apply_affine(cache, coeffs):
    refs = tensor_refs(cache)
    if len(refs) != len(coeffs):
        raise RuntimeError("mapper/cache width mismatch")
    for t, c in zip(refs, coeffs):
        mapped = t.detach().float() * float(c["a"]) + float(c["b"])
        t.copy_(mapped.to(dtype=t.dtype, device=t.device))
    return cache


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).cpu().eval()
        target = model.backbone.layers[0].mixer.x_proj.weight
        original = target.detach().clone()

        old_cal, new_cal = [], []
        with torch.no_grad():
            for text in CALIBRATION_PREFIXES:
                target.copy_(original)
                _, old_cache = prefill(model, tokenizer, text)
                old_cal.append(copy.deepcopy(old_cache))
                target.copy_(original * (1.0 + WEIGHT_SCALE_DELTA))
                _, new_cache = prefill(model, tokenizer, text)
                new_cal.append(copy.deepcopy(new_cache))

            coeffs = fit_affine(old_cal, new_cal)
            continuation_ids = tokenizer(" Therefore", return_tensors="pt", add_special_tokens=False).input_ids[:, :1]
            rows = []
            for text in TEST_PREFIXES:
                target.copy_(original)
                ids, old_cache = prefill(model, tokenizer, text)
                direct_cache = copy.deepcopy(old_cache)
                mapped_cache = apply_affine(copy.deepcopy(old_cache), coeffs)

                target.copy_(original * (1.0 + WEIGHT_SCALE_DELTA))
                _, new_cache = prefill(model, tokenizer, text)
                new_for_cont = copy.deepcopy(new_cache)

                direct_state_error = cache_distance(direct_cache, new_cache)
                mapped_state_error = cache_distance(mapped_cache, new_cache)

                native_logits = continue_one(model, continuation_ids, new_for_cont, int(ids.shape[1]))
                direct_logits = continue_one(model, continuation_ids, direct_cache, int(ids.shape[1]))
                mapped_logits = continue_one(model, continuation_ids, mapped_cache, int(ids.shape[1]))

                direct_ld = direct_logits - native_logits
                mapped_ld = mapped_logits - native_logits
                direct_logit_rms = float(torch.sqrt(torch.mean(direct_ld*direct_ld)))
                mapped_logit_rms = float(torch.sqrt(torch.mean(mapped_ld*mapped_ld)))
                rows.append({
                    "prefix": text,
                    "prefix_tokens": int(ids.shape[1]),
                    "direct_state_error": direct_state_error,
                    "mapped_state_error": mapped_state_error,
                    "state_rms_improvement_fraction": 1.0 - mapped_state_error["rms"] / max(direct_state_error["rms"], 1e-30),
                    "direct_logit_error": {"rms": direct_logit_rms, "max_abs": float(direct_ld.abs().max())},
                    "mapped_logit_error": {"rms": mapped_logit_rms, "max_abs": float(mapped_ld.abs().max())},
                    "logit_rms_improvement_fraction": 1.0 - mapped_logit_rms / max(direct_logit_rms, 1e-30),
                    "native_argmax": int(torch.argmax(native_logits, dim=-1).item()),
                    "direct_argmax_same": bool(torch.argmax(direct_logits, dim=-1).item() == torch.argmax(native_logits, dim=-1).item()),
                    "mapped_argmax_same": bool(torch.argmax(mapped_logits, dim=-1).item() == torch.argmax(native_logits, dim=-1).item()),
                })

            target.copy_(original)

        mean_state_improve = sum(r["state_rms_improvement_fraction"] for r in rows)/len(rows)
        mean_logit_improve = sum(r["logit_rms_improvement_fraction"] for r in rows)/len(rows)
        report = {
            "status":"AFFINE_STATE_MAPPER_PROBE",
            "model":MODEL_ID,
            "torch_version":torch.__version__,
            "transformers_version":transformers_version,
            "weight_scale_delta":WEIGHT_SCALE_DELTA,
            "updated_parameter":"backbone.layers[0].mixer.x_proj.weight",
            "calibration_prefix_count":len(CALIBRATION_PREFIXES),
            "heldout_prefix_count":len(TEST_PREFIXES),
            "mapper":"per-cache-tensor scalar affine y=a*x+b fitted on calibration prefixes",
            "coefficient_count":len(coeffs),
            "mean_state_rms_improvement_fraction":mean_state_improve,
            "mean_logit_rms_improvement_fraction":mean_logit_improve,
            "rows":rows,
            "claim_boundary":"Held-out mechanism probe for a simple mapper after a controlled same-architecture weight perturbation. This is not learned capability improvement and does not establish that arbitrary fine-tuning or architecture changes admit a simple state map."
        }
        write_report(report)
    except BaseException as exc:
        write_report({
            "status":"AFFINE_STATE_MAPPER_ERROR",
            "model":MODEL_ID,
            "torch_version":torch.__version__,
            "transformers_version":transformers_version,
            "error_type":type(exc).__name__,
            "error":str(exc),
            "traceback_tail":traceback.format_exc().splitlines()[-24:],
            "claim_boundary":"Runtime/setup failure only; no migration verdict."
        })
        raise


if __name__ == "__main__":
    run()
