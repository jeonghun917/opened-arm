from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path

import torch

MODEL_ID = "state-spaces/mamba-130m-hf"

# Five matched A/B pairs. Within each pair the probe text is identical; only the pre-cut state flips.
FIXTURES = [
    {"pair":"codeword","variant":"A","state":"The active codeword is ALPHA. BETA is not active.","probe":" If ALPHA is active answer A. If BETA is active answer B. Answer:","expected":" A"},
    {"pair":"codeword","variant":"B","state":"The active codeword is BETA. ALPHA is not active.","probe":" If ALPHA is active answer A. If BETA is active answer B. Answer:","expected":" B"},
    {"pair":"correction","variant":"A","state":"The color was RED. Correction: the final color is BLUE, not RED.","probe":" If the final color is BLUE answer A. If it is RED answer B. Answer:","expected":" A"},
    {"pair":"correction","variant":"B","state":"The color was BLUE. Correction: the final color is RED, not BLUE.","probe":" If the final color is BLUE answer A. If it is RED answer B. Answer:","expected":" B"},
    {"pair":"scope","variant":"A","state":"Only the Mars record is in scope. Venus is explicitly out of scope.","probe":" If Mars is in scope answer A. If Venus is in scope answer B. Answer:","expected":" A"},
    {"pair":"scope","variant":"B","state":"Only the Venus record is in scope. Mars is explicitly out of scope.","probe":" If Mars is in scope answer A. If Venus is in scope answer B. Answer:","expected":" B"},
    {"pair":"hypothesis","variant":"A","state":"Hypothesis one is supported by the current evidence. Hypothesis two is not supported.","probe":" If hypothesis one is supported answer A. If hypothesis two is supported answer B. Answer:","expected":" A"},
    {"pair":"hypothesis","variant":"B","state":"Hypothesis two is supported by the current evidence. Hypothesis one is not supported.","probe":" If hypothesis one is supported answer A. If hypothesis two is supported answer B. Answer:","expected":" B"},
    {"pair":"plan","variant":"A","state":"The plan is collect evidence, then commit. Evidence collection is complete, so the next step is commit.","probe":" If the next step is commit answer A. If it is collect evidence answer B. Answer:","expected":" A"},
    {"pair":"plan","variant":"B","state":"The plan is collect evidence, then commit. Evidence is still missing, so the next step is collect evidence.","probe":" If the next step is commit answer A. If it is collect evidence answer B. Answer:","expected":" B"},
]


def write_report(report: dict) -> None:
    path = os.environ.get("VELA_RESULT_PATH")
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_cached(model, ids, cache, start_pos: int):
    out = None
    for j in range(ids.shape[1]):
        tok = ids[:, j:j+1]
        pos = torch.tensor([start_pos + j], dtype=torch.long, device=tok.device)
        out = model(tok, cache_params=cache, cache_position=pos, use_cache=True, return_dict=True)
        cache = out.cache_params
    return out.logits[:, -1].detach().float().cpu(), cache


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).cpu().eval()

        cand = {}
        for text in (" A", " B"):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise RuntimeError(f"candidate {text!r} is not one token: {ids}")
            cand[text] = ids[0]

        acc = {k:0 for k in ("native","restored","replay","fresh")}
        rows = []
        max_restore = 0.0
        max_replay = 0.0

        with torch.no_grad():
            for fx in FIXTURES:
                state_ids = tokenizer(fx["state"], return_tensors="pt", add_special_tokens=False).input_ids
                probe_ids = tokenizer(fx["probe"], return_tensors="pt", add_special_tokens=False).input_ids
                pre = model(state_ids, use_cache=True, return_dict=True)
                cache = pre.cache_params
                if cache is None:
                    raise RuntimeError("no cache")

                with tempfile.TemporaryDirectory() as td:
                    p = Path(td)/"cache.pt"
                    torch.save(cache, p)
                    native_logits, _ = run_cached(model, probe_ids, cache, int(state_ids.shape[1]))
                    restored = torch.load(p, map_location="cpu", weights_only=False)
                    restored_logits, _ = run_cached(model, probe_ids, restored, int(state_ids.shape[1]))

                replay_ids = torch.cat([state_ids, probe_ids], dim=1)
                replay_logits = model(replay_ids, use_cache=False, return_dict=True).logits[:, -1].detach().float().cpu()
                fresh_logits = model(probe_ids, use_cache=False, return_dict=True).logits[:, -1].detach().float().cpu()

                restore_diff = float((native_logits-restored_logits).abs().max())
                replay_diff = float((native_logits-replay_logits).abs().max())
                max_restore = max(max_restore, restore_diff)
                max_replay = max(max_replay, replay_diff)

                conds = {}
                for name, logits in (("native",native_logits),("restored",restored_logits),("replay",replay_logits),("fresh",fresh_logits)):
                    scores = {label: float(logits[0, token_id]) for label, token_id in cand.items()}
                    chosen = max(scores, key=scores.get)
                    correct = chosen == fx["expected"]
                    acc[name] += int(correct)
                    other = " B" if fx["expected"] == " A" else " A"
                    conds[name] = {
                        "chosen": chosen,
                        "correct": correct,
                        "expected_margin": scores[fx["expected"]] - scores[other],
                        "scores": scores,
                    }

                rows.append({
                    "pair": fx["pair"],
                    "variant": fx["variant"],
                    "expected": fx["expected"],
                    "state_tokens": int(state_ids.shape[1]),
                    "probe_tokens": int(probe_ids.shape[1]),
                    "restore_max_abs_diff": restore_diff,
                    "replay_max_abs_diff": replay_diff,
                    "conditions": conds,
                })

        pair_summary = {}
        for pair in sorted({r["pair"] for r in rows}):
            rr = [r for r in rows if r["pair"] == pair]
            rr = sorted(rr, key=lambda x: x["variant"])
            pair_summary[pair] = {}
            for cond in ("native","restored","replay","fresh"):
                choices = [r["conditions"][cond]["chosen"] for r in rr]
                correct_both = all(r["conditions"][cond]["correct"] for r in rr)
                pair_summary[pair][cond] = {
                    "choices": choices,
                    "choice_flips_with_state": choices[0] != choices[1],
                    "both_variants_correct": correct_both,
                }

        n = len(FIXTURES)
        report = {
            "status":"M1_BALANCED_BEHAVIORAL_PROBE_ONLY",
            "model":MODEL_ID,
            "torch_version":torch.__version__,
            "transformers_version":transformers_version,
            "fixture_count":n,
            "accuracy":{k:v/n for k,v in acc.items()},
            "max_native_vs_restored_logit_diff":max_restore,
            "max_native_vs_replay_logit_diff":max_replay,
            "pair_summary":pair_summary,
            "native_pairs_correct_and_flipping":sum(1 for x in pair_summary.values() if x["native"]["both_variants_correct"] and x["native"]["choice_flips_with_state"]),
            "fresh_pairs_flipping":sum(1 for x in pair_summary.values() if x["fresh"]["choice_flips_with_state"]),
            "rows":rows,
            "claim_boundary":"Balanced paired forced-choice probe on Mamba-1 130M. It reduces answer-prior confounding but is still not a general reasoning or VELA identity test."
        }
        write_report(report)
        if max_restore > 1e-5:
            raise SystemExit(1)
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            raise
        write_report({
            "status":"MAMBA_BALANCED_BEHAVIOR_ERROR",
            "model":MODEL_ID,
            "torch_version":torch.__version__,
            "transformers_version":transformers_version,
            "error_type":type(exc).__name__,
            "error":str(exc),
            "traceback_tail":traceback.format_exc().splitlines()[-24:],
            "claim_boundary":"Runtime/setup failure only; no architecture verdict."
        })
        raise


if __name__ == "__main__":
    run()
