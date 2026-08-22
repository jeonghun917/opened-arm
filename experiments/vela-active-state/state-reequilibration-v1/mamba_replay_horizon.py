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
HORIZONS = [0, 1, 2, 4, 8, 16, 32, 64]

PREFIX_TEXT = (
    "VELA is tracking two hypotheses. The timing hypothesis was weakened by the latest observation, "
    "while the scope hypothesis remains unresolved. The current plan is to collect one more piece of evidence, "
    "then verify the conflict before committing any external action."
)
CONTINUATION_TEXT = " Therefore"


def write_report(report: dict) -> None:
    path = os.environ.get("VELA_RESULT_PATH")
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def tensor_refs(obj):
    refs = []
    seen = set()
    def walk(x):
        if torch.is_tensor(x):
            if id(x) not in seen:
                seen.add(id(x)); refs.append(x)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
        elif hasattr(x, "__dict__"):
            for v in vars(x).values(): walk(v)
    walk(obj)
    return refs


def cache_distance(a, b):
    aa, bb = tensor_refs(a), tensor_refs(b)
    if len(aa) != len(bb):
        raise RuntimeError(f"cache tensor count mismatch {len(aa)} != {len(bb)}")
    sq = 0.0; mx = 0.0; n = 0
    for x, y in zip(aa, bb):
        if x.shape != y.shape:
            raise RuntimeError(f"cache shape mismatch {x.shape} != {y.shape}")
        d = x.detach().float().cpu() - y.detach().float().cpu()
        sq += float((d*d).sum()); mx = max(mx, float(d.abs().max())); n += d.numel()
    return {"rms": math.sqrt(sq/max(n,1)), "l2": math.sqrt(sq), "max_abs": mx, "numel": int(n)}


def prefill(model, ids):
    out = model(ids, use_cache=True, return_dict=True)
    if out.cache_params is None:
        raise RuntimeError("Mamba returned no cache")
    return out.cache_params


def feed_tokens(model, ids, cache, start_pos: int):
    for j in range(ids.shape[1]):
        tok = ids[:, j:j+1]
        pos = torch.tensor([start_pos+j], dtype=torch.long, device=tok.device)
        out = model(tok, cache_params=cache, cache_position=pos, use_cache=True, return_dict=True)
        cache = out.cache_params
    return cache


def continue_logits(model, token, cache, pos: int):
    p = torch.tensor([pos], dtype=torch.long, device=token.device)
    out = model(token, cache_params=cache, cache_position=p, use_cache=True, return_dict=True)
    return out.logits[:, -1].detach().float().cpu()


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).cpu().eval()
        prefix = tok(PREFIX_TEXT, return_tensors="pt", add_special_tokens=False).input_ids
        cont = tok(CONTINUATION_TEXT, return_tensors="pt", add_special_tokens=False).input_ids[:, :1]
        T = int(prefix.shape[1])
        target = model.backbone.layers[0].mixer.x_proj.weight
        original = target.detach().clone()

        horizons = sorted(set(min(h, T) for h in HORIZONS) | {T})
        with torch.no_grad():
            # Gold W2 state at t: full prefix processed by the upgraded engine.
            target.copy_(original * (1.0 + WEIGHT_SCALE_DELTA))
            gold_state = prefill(model, prefix)
            gold_logits = continue_logits(model, cont, copy.deepcopy(gold_state), T)

            rows = []
            for h in horizons:
                if h == 0:
                    # Direct import: W1 processes all history; W2 immediately inherits S1_t.
                    target.copy_(original)
                    hybrid_state = prefill(model, prefix)
                elif h == T:
                    # Full replay under W2: exact gold control.
                    target.copy_(original * (1.0 + WEIGHT_SCALE_DELTA))
                    hybrid_state = prefill(model, prefix)
                else:
                    cut = T - h
                    target.copy_(original)
                    old_state_at_cut = prefill(model, prefix[:, :cut])
                    target.copy_(original * (1.0 + WEIGHT_SCALE_DELTA))
                    hybrid_state = feed_tokens(model, prefix[:, cut:], old_state_at_cut, cut)

                target.copy_(original * (1.0 + WEIGHT_SCALE_DELTA))
                state_err = cache_distance(hybrid_state, gold_state)
                logits = continue_logits(model, cont, copy.deepcopy(hybrid_state), T)
                ld = logits - gold_logits
                rows.append({
                    "w2_replayed_recent_tokens": int(h),
                    "w1_history_tokens_kept_without_recompute": int(T-h),
                    "state_error_vs_full_w2": state_err,
                    "continuation_logit_error_vs_full_w2": {
                        "rms": float(torch.sqrt(torch.mean(ld*ld))),
                        "max_abs": float(ld.abs().max()),
                        "argmax_same": bool(torch.argmax(logits, dim=-1).item() == torch.argmax(gold_logits, dim=-1).item()),
                    },
                })

            target.copy_(original)

        report = {
            "status": "STATE_REEQUILIBRATION_HORIZON_PROBE",
            "model": MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "weight_scale_delta": WEIGHT_SCALE_DELTA,
            "updated_parameter": "backbone.layers[0].mixer.x_proj.weight",
            "prefix_tokens": T,
            "horizons": horizons,
            "rows": rows,
            "interpretation": (
                "At horizon h, the engine keeps a W1 checkpoint from t-h and reprocesses only the most recent h real tokens under W2. "
                "This measures how much causal replay under the upgraded transition dynamics is needed to approach the W2-native state."
            ),
            "claim_boundary": (
                "Controlled same-architecture weight perturbation, not a learned capability upgrade. Partial causal replay uses actual engine state and token transitions; "
                "it is not declarative record reinjection. Full replay is included only as the W2-native control."
            ),
        }
        write_report(report)
    except BaseException as exc:
        write_report({
            "status": "STATE_REEQUILIBRATION_HORIZON_ERROR",
            "model": MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-28:],
            "claim_boundary": "Runtime/setup failure only; no migration verdict.",
        })
        raise


if __name__ == "__main__":
    run()
