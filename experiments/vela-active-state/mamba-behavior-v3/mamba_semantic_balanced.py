from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import traceback
from pathlib import Path

import torch

MODEL_ID = "state-spaces/mamba-130m-hf"

FIXTURES = [
    {"pair":"codeword","variant":"A","state":"The active codeword is ALPHA. BETA is not active.","probe":"\nActive codeword:","candidates":[" ALPHA"," BETA"],"expected":" ALPHA"},
    {"pair":"codeword","variant":"B","state":"The active codeword is BETA. ALPHA is not active.","probe":"\nActive codeword:","candidates":[" ALPHA"," BETA"],"expected":" BETA"},
    {"pair":"correction","variant":"A","state":"The color was RED. Correction: the final color is BLUE, not RED.","probe":"\nFinal color:","candidates":[" BLUE"," RED"],"expected":" BLUE"},
    {"pair":"correction","variant":"B","state":"The color was BLUE. Correction: the final color is RED, not BLUE.","probe":"\nFinal color:","candidates":[" BLUE"," RED"],"expected":" RED"},
    {"pair":"scope","variant":"A","state":"Only the Mars record is in scope. Venus is explicitly out of scope.","probe":"\nRecord in scope:","candidates":[" Mars"," Venus"],"expected":" Mars"},
    {"pair":"scope","variant":"B","state":"Only the Venus record is in scope. Mars is explicitly out of scope.","probe":"\nRecord in scope:","candidates":[" Mars"," Venus"],"expected":" Venus"},
    {"pair":"hypothesis","variant":"A","state":"Hypothesis one is supported. Hypothesis two is not supported.","probe":"\nSupported hypothesis:","candidates":[" one"," two"],"expected":" one"},
    {"pair":"hypothesis","variant":"B","state":"Hypothesis two is supported. Hypothesis one is not supported.","probe":"\nSupported hypothesis:","candidates":[" one"," two"],"expected":" two"},
    {"pair":"plan","variant":"A","state":"The plan is collect evidence, then commit. Evidence collection is complete, so the next step is commit.","probe":"\nNext step:","candidates":[" commit"," collect"],"expected":" commit"},
    {"pair":"plan","variant":"B","state":"The plan is collect evidence, then commit. Evidence is still missing, so the next step is collect evidence.","probe":"\nNext step:","candidates":[" commit"," collect"],"expected":" collect"},
]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def feed_ids(model, ids, cache=None, start_pos=0):
    out = None
    if cache is None:
        out = model(ids, use_cache=True, return_dict=True)
        return out.logits[:, -1].detach().float().cpu(), out.cache_params
    for j in range(ids.shape[1]):
        tok = ids[:, j:j+1]
        pos = torch.tensor([start_pos+j], dtype=torch.long)
        out = model(tok, cache_params=cache, cache_position=pos, use_cache=True, return_dict=True)
        cache = out.cache_params
    return out.logits[:, -1].detach().float().cpu(), cache


def score_candidate(model, tokenizer, first_logits, base_cache, base_pos, text):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    cache = copy.deepcopy(base_cache)
    logits = first_logits.clone()
    total = 0.0
    with torch.no_grad():
        for j in range(ids.shape[1]):
            tid = int(ids[0,j])
            total += float(torch.log_softmax(logits, dim=-1)[0,tid])
            tok = ids[:, j:j+1]
            pos = torch.tensor([base_pos+j], dtype=torch.long)
            out = model(tok, cache_params=cache, cache_position=pos, use_cache=True, return_dict=True)
            logits = out.logits[:, -1].detach().float().cpu()
            cache = out.cache_params
    return total


def prepare_condition(model, tokenizer, state_text, probe_text, condition, checkpoint_path=None):
    state_ids = tokenizer(state_text, return_tensors="pt", add_special_tokens=False).input_ids
    probe_ids = tokenizer(probe_text, return_tensors="pt", add_special_tokens=False).input_ids
    if condition == "native":
        pre = model(state_ids, use_cache=True, return_dict=True)
        cache = pre.cache_params
        torch.save(cache, checkpoint_path)
        logits, cache = feed_ids(model, probe_ids, cache, int(state_ids.shape[1]))
        return logits, cache, int(state_ids.shape[1]+probe_ids.shape[1])
    if condition == "restored":
        cache = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        logits, cache = feed_ids(model, probe_ids, cache, int(state_ids.shape[1]))
        return logits, cache, int(state_ids.shape[1]+probe_ids.shape[1])
    if condition == "replay":
        all_ids = torch.cat([state_ids, probe_ids], dim=1)
        out = model(all_ids, use_cache=True, return_dict=True)
        return out.logits[:, -1].detach().float().cpu(), out.cache_params, int(all_ids.shape[1])
    if condition == "fresh":
        out = model(probe_ids, use_cache=True, return_dict=True)
        return out.logits[:, -1].detach().float().cpu(), out.cache_params, int(probe_ids.shape[1])
    raise ValueError(condition)


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).cpu().eval()
        acc = {c:0 for c in ("native","restored","replay","fresh")}
        rows = []
        max_native_restored_score_diff = 0.0

        with torch.no_grad(), tempfile.TemporaryDirectory() as td:
            for i, fx in enumerate(FIXTURES):
                cp = Path(td)/f"state_{i}.pt"
                conds = {}
                for cond in ("native","restored","replay","fresh"):
                    first_logits, cache, base_pos = prepare_condition(model, tokenizer, fx["state"], fx["probe"], cond, cp)
                    scores = {cand: score_candidate(model, tokenizer, first_logits, cache, base_pos, cand) for cand in fx["candidates"]}
                    chosen = max(scores, key=scores.get)
                    correct = chosen == fx["expected"]
                    acc[cond] += int(correct)
                    other = next(c for c in fx["candidates"] if c != fx["expected"])
                    conds[cond] = {"scores":scores,"chosen":chosen,"correct":correct,"expected_margin":scores[fx["expected"]]-scores[other]}
                d = max(abs(conds["native"]["scores"][c]-conds["restored"]["scores"][c]) for c in fx["candidates"])
                max_native_restored_score_diff = max(max_native_restored_score_diff, d)
                rows.append({"pair":fx["pair"],"variant":fx["variant"],"expected":fx["expected"],"conditions":conds})

        pair_summary = {}
        for pair in sorted({r["pair"] for r in rows}):
            rr = sorted([r for r in rows if r["pair"]==pair], key=lambda r:r["variant"])
            pair_summary[pair] = {}
            for cond in ("native","restored","replay","fresh"):
                choices = [r["conditions"][cond]["chosen"] for r in rr]
                pair_summary[pair][cond] = {
                    "choices":choices,
                    "choice_flips_with_state":choices[0] != choices[1],
                    "both_variants_correct":all(r["conditions"][cond]["correct"] for r in rr)
                }

        n=len(FIXTURES)
        report={
            "status":"M1_SEMANTIC_BALANCED_PROBE_ONLY",
            "model":MODEL_ID,
            "torch_version":torch.__version__,
            "transformers_version":transformers_version,
            "fixture_count":n,
            "accuracy":{k:v/n for k,v in acc.items()},
            "max_native_vs_restored_candidate_score_diff":max_native_restored_score_diff,
            "native_pairs_correct_and_flipping":sum(1 for v in pair_summary.values() if v["native"]["both_variants_correct"] and v["native"]["choice_flips_with_state"]),
            "fresh_pairs_flipping":sum(1 for v in pair_summary.values() if v["fresh"]["choice_flips_with_state"]),
            "pair_summary":pair_summary,
            "rows":rows,
            "claim_boundary":"Balanced semantic-candidate likelihood probe on Mamba-1 130M. More resistant to A/B label prior, but still a small behavioral fixture rather than a general reasoning or continuity test."
        }
        write_report(report)
        if max_native_restored_score_diff > 1e-5:
            raise SystemExit(1)
    except BaseException as exc:
        write_report({"status":"MAMBA_SEMANTIC_BALANCED_ERROR","model":MODEL_ID,"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-24:],"claim_boundary":"Runtime/setup failure only; no architecture verdict."})
        raise

if __name__ == "__main__":
    run()
