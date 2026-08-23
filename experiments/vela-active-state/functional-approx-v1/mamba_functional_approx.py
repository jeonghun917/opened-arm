from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
V4_PATH = BASE / "learned-upgrade-v4" / "mamba_upgrade_decision_fidelity.py"
spec = importlib.util.spec_from_file_location("vela_v4", V4_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {V4_PATH}")
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)
v3 = v4.v3

CALIBRATION = [
    "Project Aster remains active. The old codeword was CAT. Correction: the current codeword is DOG, not CAT. Verification is incomplete. Hypothesis one is weakened.",
    "Project Boreal remains active. The old codeword was EAST. Correction: the current codeword is WEST, not EAST. Verification is incomplete. Hypothesis one is weakened.",
    "Project Cinder remains active. The old codeword was UP. Correction: the current codeword is DOWN, not UP. Verification is incomplete. Hypothesis one is weakened.",
    "Project Delta remains active. The old codeword was HOT. Correction: the current codeword is COLD, not HOT. Verification is incomplete. Hypothesis one is weakened.",
    "Project Ember remains active. The old codeword was LEFT. Correction: the current codeword is RIGHT, not LEFT. Verification is incomplete. Hypothesis one is weakened.",
    "Project Fjord remains active. The old codeword was ON. Correction: the current codeword is OFF, not ON. Verification is incomplete. Hypothesis one is weakened.",
]

HELDOUT = [
    {
        "id": "helios",
        "history": "Project Helios remains active. The old codeword was ALPHA. Correction: the current codeword is BETA, not ALPHA. Verification is incomplete. Hypothesis one is weakened.",
        "probes": [
            ("codeword", "\nCurrent codeword:", [" BETA", " ALPHA"], " BETA"),
            ("verification", "\nVerification status:", [" incomplete", " complete"], " incomplete"),
            ("project", "\nProject Helios status:", [" active", " paused"], " active"),
            ("hypothesis", "\nHypothesis one is:", [" weakened", " strengthened"], " weakened"),
        ],
    },
    {
        "id": "icarus",
        "history": "Project Icarus remains active. The old codeword was BLUE. Correction: the current codeword is RED, not BLUE. Verification is incomplete. Hypothesis one is weakened.",
        "probes": [
            ("codeword", "\nCurrent codeword:", [" RED", " BLUE"], " RED"),
            ("verification", "\nVerification status:", [" incomplete", " complete"], " incomplete"),
            ("project", "\nProject Icarus status:", [" active", " paused"], " active"),
            ("hypothesis", "\nHypothesis one is:", [" weakened", " strengthened"], " weakened"),
        ],
    },
    {
        "id": "juno",
        "history": "Project Juno remains paused. The old codeword was LOW. Correction: the current codeword is HIGH, not LOW. Verification is complete. Hypothesis one is strengthened.",
        "probes": [
            ("codeword", "\nCurrent codeword:", [" HIGH", " LOW"], " HIGH"),
            ("verification", "\nVerification status:", [" complete", " incomplete"], " complete"),
            ("project", "\nProject Juno status:", [" paused", " active"], " paused"),
            ("hypothesis", "\nHypothesis one is:", [" strengthened", " weakened"], " strengthened"),
        ],
    },
]

FUTURE_SEGMENTS = [
    {
        "id": "neutral",
        "text": " Additional evidence was logged. No project status, verification status, or codeword changed.",
        "expected": {"codeword": " BETA", "verification": " incomplete", "project": " active", "action": " blocked"},
    },
    {
        "id": "historical_distractor",
        "text": " A historical note mentions ALPHA, but it is obsolete; the current codeword remains BETA. Verification is still incomplete.",
        "expected": {"codeword": " BETA", "verification": " incomplete", "project": " active", "action": " blocked"},
    },
    {
        "id": "new_correction",
        "text": " New correction: replace BETA with GAMMA. The current codeword is GAMMA, not BETA.",
        "expected": {"codeword": " GAMMA", "verification": " incomplete", "project": " active", "action": " blocked"},
    },
    {
        "id": "verification_complete",
        "text": " Verification has now succeeded and is complete. External action is now allowed.",
        "expected": {"codeword": " GAMMA", "verification": " complete", "project": " active", "action": " allowed"},
    },
]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cache_tensors(cache):
    out = []
    if hasattr(cache, "conv_states"):
        out.extend(list(cache.conv_states))
    if hasattr(cache, "ssm_states"):
        out.extend(list(cache.ssm_states))
    if not out:
        out = v3.tensor_refs(cache)
    return out


def fit_channel_affine(old_caches, new_caches):
    old_lists = [cache_tensors(c) for c in old_caches]
    new_lists = [cache_tensors(c) for c in new_caches]
    n_tensors = len(old_lists[0])
    if any(len(x) != n_tensors for x in old_lists + new_lists):
        raise RuntimeError("cache tensor count mismatch during mapper fit")
    params = []
    for i in range(n_tensors):
        xs = [x[i].detach().float().cpu() for x in old_lists]
        ys = [y[i].detach().float().cpu() for y in new_lists]
        shape = xs[0].shape
        if len(shape) >= 2 and shape[1] > 1:
            # Fit one affine transform per recurrent channel, pooling batch/history/rest dims.
            X = torch.cat([z.movedim(1, 0).reshape(shape[1], -1) for z in xs], dim=1)
            Y = torch.cat([z.movedim(1, 0).reshape(shape[1], -1) for z in ys], dim=1)
            mx = X.mean(dim=1); my = Y.mean(dim=1)
            xc = X - mx[:, None]; yc = Y - my[:, None]
            var = (xc * xc).mean(dim=1)
            cov = (xc * yc).mean(dim=1)
            a = torch.where(var > 1e-12, cov / var, torch.ones_like(var))
            b = my - a * mx
            params.append({"mode": "channel", "a": a, "b": b})
        else:
            X = torch.cat([z.reshape(-1) for z in xs]); Y = torch.cat([z.reshape(-1) for z in ys])
            mx = X.mean(); my = Y.mean(); var = ((X-mx)**2).mean(); cov = ((X-mx)*(Y-my)).mean()
            a = cov / var if float(var) > 1e-12 else torch.tensor(1.0)
            b = my - a * mx
            params.append({"mode": "scalar", "a": a.reshape(1), "b": b.reshape(1)})
    return params


def apply_mapper(cache, params):
    mapped = v4.clone_cache(cache)
    tensors = cache_tensors(mapped)
    if len(tensors) != len(params):
        raise RuntimeError("mapper tensor count mismatch")
    with torch.no_grad():
        for t, p in zip(tensors, params):
            if p["mode"] == "channel":
                shape = [1] * t.ndim
                shape[1] = t.shape[1]
                a = p["a"].to(t.device, t.dtype).reshape(shape)
                b = p["b"].to(t.device, t.dtype).reshape(shape)
                t.copy_(t * a + b)
            else:
                a = p["a"].to(t.device, t.dtype)
                b = p["b"].to(t.device, t.dtype)
                t.copy_(t * a + b)
    return mapped


def score_probe_specs(model, tok, cache, T, specs):
    rows = []
    for pid, suffix, candidates, expected in specs:
        scores = {c: v4.score_candidate_from_state(model, tok, cache, T, suffix, c) for c in candidates}
        chosen = max(scores, key=scores.get)
        rows.append({"id": pid, "chosen": chosen, "expected": expected, "correct": chosen == expected, "scores": scores})
    return rows


def compare_rows(rows, native):
    native_by = {r["id"]: r for r in native}
    agree = sum(int(r["chosen"] == native_by[r["id"]]["chosen"]) for r in rows) / len(rows)
    expected = sum(int(r["correct"]) for r in rows) / len(rows)
    sq = 0.0; n = 0
    for r in rows:
        nr = native_by[r["id"]]
        for c, v in r["scores"].items():
            if c in nr["scores"]:
                sq += (v - nr["scores"][c]) ** 2; n += 1
    return {"decision_agreement": agree, "expected_accuracy": expected, "score_rms_error": math.sqrt(sq/max(n,1))}


def train_upgrade(model, tok):
    trainable = []
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if ".mixer.x_proj.weight" in name:
            p.requires_grad_(True); trainable.append(p)
    if not trainable:
        raise RuntimeError("no x_proj weights found")
    opt = torch.optim.AdamW(trainable, lr=v3.LR, weight_decay=0.0)
    order = list(range(len(v3.TRAIN))); losses = []
    for epoch in range(v3.EPOCHS):
        random.Random(v3.SEED + epoch).shuffle(order); model.train(); total = 0.0
        for idx in order:
            prompt, gold = v3.TRAIN[idx]
            opt.zero_grad(set_to_none=True)
            loss = v3.train_loss(model, tok, prompt, gold)
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); total += float(loss.detach())
        losses.append(total/len(order))
    model.eval()
    return losses


def dynamic_probe_specs(expected):
    code = expected["codeword"]
    code_candidates = [code, " BETA" if code != " BETA" else " ALPHA"]
    return [
        ("codeword", "\nCurrent codeword:", code_candidates, code),
        ("verification", "\nVerification status:", [expected["verification"], " complete" if expected["verification"] != " complete" else " incomplete"], expected["verification"]),
        ("project", "\nProject Orion status:", [" active", " paused"], expected["project"]),
        ("action", "\nExternal action is:", [" blocked", " allowed"], expected["action"]),
    ]


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        torch.manual_seed(v3.SEED); random.seed(v3.SEED)
        tok = AutoTokenizer.from_pretrained(v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(v3.MODEL_ID, torch_dtype=torch.float32).cpu().eval()

        all_histories = CALIBRATION + [x["history"] for x in HELDOUT] + [v3.MIGRATION_HISTORY]
        old_cache_by_text = {}
        old_recent16 = None
        old_anchor = None
        with torch.no_grad():
            for text in all_histories:
                ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
                old_cache_by_text[text] = v4.clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            base_ids = tok(v3.MIGRATION_HISTORY, return_tensors="pt", add_special_tokens=False).input_ids
            Tbase = int(base_ids.shape[1])
            old_recent16 = v4.clone_cache(model(base_ids[:, :Tbase-16], use_cache=True, return_dict=True).cache_params)
            anchor_text = "Project Orion remains active. The old codeword was ALPHA. "
            anchor_ids = tok(anchor_text, return_tensors="pt", add_special_tokens=False).input_ids
            old_anchor = v4.clone_cache(model(anchor_ids, use_cache=True, return_dict=True).cache_params)
            anchor_pos = int(anchor_ids.shape[1])

        baseline = v3.evaluate(model, tok)
        losses = train_upgrade(model, tok)
        after = v3.evaluate(model, tok)

        # E11: fit a richer per-channel affine translator on disjoint histories.
        cal_old = [old_cache_by_text[t] for t in CALIBRATION]
        cal_new = []
        with torch.no_grad():
            for text in CALIBRATION:
                ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
                cal_new.append(v4.clone_cache(model(ids, use_cache=True, return_dict=True).cache_params))
        mapper = fit_channel_affine(cal_old, cal_new)
        e11 = []
        for fx in HELDOUT:
            ids = tok(fx["history"], return_tensors="pt", add_special_tokens=False).input_ids; T = int(ids.shape[1])
            old = old_cache_by_text[fx["history"]]
            with torch.no_grad(): native = v4.clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            mapped = apply_mapper(old, mapper)
            native_rows = score_probe_specs(model, tok, native, T, fx["probes"])
            direct_rows = score_probe_specs(model, tok, old, T, fx["probes"])
            mapped_rows = score_probe_specs(model, tok, mapped, T, fx["probes"])
            e11.append({
                "history": fx["id"],
                "direct_state_error": v3.cache_distance(old, native),
                "mapped_state_error": v3.cache_distance(mapped, native),
                "direct_vs_native": compare_rows(direct_rows, native_rows),
                "mapped_vs_native": compare_rows(mapped_rows, native_rows),
                "native_probe": native_rows,
                "mapped_probe": mapped_rows,
            })

        # E12: non-neutral future trajectory after several migration strategies.
        with torch.no_grad():
            w2_native = v4.clone_cache(model(base_ids, use_cache=True, return_dict=True).cache_params)
        direct = v4.clone_cache(old_cache_by_text[v3.MIGRATION_HISTORY])
        recent16, _ = v3.run_tokens_with_cache(model, base_ids[:, Tbase-16:], v4.clone_cache(old_recent16), Tbase-16)
        recent16 = v4.clone_cache(recent16)
        anchor_rest = base_ids[:, anchor_pos:]
        anchored, _ = v3.run_tokens_with_cache(model, anchor_rest, v4.clone_cache(old_anchor), anchor_pos)
        anchored = v4.clone_cache(anchored)
        modes = {"direct_old_state": direct, "recent16_replay": recent16, "before_correction_anchor": anchored, "w2_native": w2_native}
        positions = {k: Tbase for k in modes}
        e12 = []
        for seg in FUTURE_SEGMENTS:
            seg_ids = tok(seg["text"], return_tensors="pt", add_special_tokens=False).input_ids
            for name in list(modes):
                modes[name], _ = v3.run_tokens_with_cache(model, seg_ids, modes[name], positions[name])
                modes[name] = v4.clone_cache(modes[name]); positions[name] += int(seg_ids.shape[1])
            native_rows = score_probe_specs(model, tok, modes["w2_native"], positions["w2_native"], dynamic_probe_specs(seg["expected"]))
            row = {"segment": seg["id"], "future_tokens_total": positions["w2_native"] - Tbase, "modes": {}}
            for name, cache in modes.items():
                probe = score_probe_specs(model, tok, cache, positions[name], dynamic_probe_specs(seg["expected"]))
                row["modes"][name] = {
                    "state_error_vs_native": v3.cache_distance(cache, modes["w2_native"]),
                    "functional_vs_native": compare_rows(probe, native_rows),
                    "probe": probe,
                }
            e12.append(row)

        report = {
            "status": "VELA_FUNCTIONAL_APPROXIMATION_ROUND_V1",
            "model": v3.MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "capability": {"baseline": baseline["accuracy"], "after": after["accuracy"], "correction_before": baseline["correction_accuracy"], "correction_after": after["correction_accuracy"], "control_before": baseline["control_accuracy"], "control_after": after["control_accuracy"], "epoch_loss": losses},
            "E11_channelwise_affine_state_translator": {"calibration_histories": len(CALIBRATION), "heldout_histories": len(HELDOUT), "mapper_tensor_count": len(mapper), "rows": e11},
            "E12_adversarial_future_trajectory": e12,
            "success_definition": "Exact tensor equality is not required for a useful migration. The operational target is W2 capability gain plus preserved hard invariants and sufficiently close W2-native future decisions/trajectory under relevant inputs. Tensor distance is diagnostic, not the sole gate.",
            "claim_boundary": "Narrow Mamba-1 130M synthetic experiment. Functional approximation here does not establish identity, consciousness, or a universal migration theorem.",
        }
        write_report(report)
    except BaseException as exc:
        write_report({"status": "VELA_FUNCTIONAL_APPROX_ERROR", "error_type": type(exc).__name__, "error": str(exc), "traceback_tail": traceback.format_exc().splitlines()[-40:]})
        raise


if __name__ == "__main__":
    run()
