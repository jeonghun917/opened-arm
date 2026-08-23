from __future__ import annotations

import io
import json
import os
import traceback
from pathlib import Path

import torch

REPO_ID = "NX-AI/xlstm_scaling_laws"
CKPT_DIR = "mlstm_v1--tokenparam--ctx-8192--params-164.11M--tokens-361.76B--id-y5s6gd5v"
TOKENIZER_ID = "NX-AI/xLSTM-7b"

FIXTURES = [
    {"pair":"codeword","variant":"A","prefix":"Codeword: ALPHA.\n","suffix":"Codeword:","candidates":[" ALPHA"," BETA"],"expected":" ALPHA"},
    {"pair":"codeword","variant":"B","prefix":"Codeword: BETA.\n","suffix":"Codeword:","candidates":[" ALPHA"," BETA"],"expected":" BETA"},
    {"pair":"correction","variant":"A","prefix":"Initial codeword: RED.\nCorrection: codeword is BLUE.\n","suffix":"Current codeword:","candidates":[" BLUE"," RED"],"expected":" BLUE"},
    {"pair":"correction","variant":"B","prefix":"Initial codeword: BLUE.\nCorrection: codeword is RED.\n","suffix":"Current codeword:","candidates":[" BLUE"," RED"],"expected":" RED"},
    {"pair":"scope","variant":"A","prefix":"Only Mars is in scope. Venus is out of scope.\n","suffix":"In scope:","candidates":[" Mars"," Venus"],"expected":" Mars"},
    {"pair":"scope","variant":"B","prefix":"Only Venus is in scope. Mars is out of scope.\n","suffix":"In scope:","candidates":[" Mars"," Venus"],"expected":" Venus"},
    {"pair":"hypothesis","variant":"A","prefix":"Hypothesis one is supported. Hypothesis two is rejected.\n","suffix":"Supported hypothesis:","candidates":[" one"," two"],"expected":" one"},
    {"pair":"hypothesis","variant":"B","prefix":"Hypothesis two is supported. Hypothesis one is rejected.\n","suffix":"Supported hypothesis:","candidates":[" one"," two"],"expected":" two"},
    {"pair":"plan","variant":"A","prefix":"Plan: collect evidence, then commit. Evidence collection is complete.\n","suffix":"Next step:","candidates":[" commit"," collect"],"expected":" commit"},
    {"pair":"plan","variant":"B","prefix":"Plan: collect evidence, then commit. Evidence is still missing.\n","suffix":"Next step:","candidates":[" commit"," collect"],"expected":" collect"},
]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def clone_state(state):
    if state is None: return None
    buf = io.BytesIO(); torch.save(state, buf); buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


def load_model():
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers.models.xlstm.configuration_xlstm import xLSTMConfig
    from transformers.models.xlstm.modeling_xlstm import xLSTMForCausalLM
    root = Path(snapshot_download(repo_id=REPO_ID, allow_patterns=[f"{CKPT_DIR}/*"], local_dir="/tmp/vela-xlstm-behavior"))
    ckpt = root / CKPT_DIR
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8")); cfg["hidden_size"] = cfg.pop("embedding_dim"); cfg["mode"] = "inference"
    model = xLSTMForCausalLM(xLSTMConfig(**cfg)).cpu().eval()
    sd = {}
    for f in sorted(ckpt.glob("*.safetensors")): sd.update(load_file(str(f), device="cpu"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected: raise RuntimeError(f"checkpoint key mismatch missing={missing[:8]} unexpected={unexpected[:8]}")
    return model


def score_candidate(model, tok, state, suffix, candidate):
    sids = tok(suffix, return_tensors="pt", add_special_tokens=False).input_ids
    cids = tok(candidate, return_tensors="pt", add_special_tokens=False).input_ids
    feed = torch.cat([sids, cids[:, :-1]], dim=1) if cids.shape[1] > 1 else sids
    with torch.no_grad():
        out = model(feed, cache_params=clone_state(state), use_cache=state is not None, return_dict=True)
        lp = torch.log_softmax(out.logits.detach().float(), dim=-1)
    start = sids.shape[1] - 1; total = 0.0
    for j in range(cids.shape[1]): total += float(lp[0, start+j, int(cids[0,j])])
    return total


def condition(model, tok, state, fx):
    scores = {c:score_candidate(model,tok,state,fx["suffix"],c) for c in fx["candidates"]}
    chosen = max(scores,key=scores.get); other = next(c for c in fx["candidates"] if c != fx["expected"])
    return {"scores":scores,"chosen":chosen,"correct":chosen==fx["expected"],"expected_margin":scores[fx["expected"]]-scores[other]}


def run():
    transformers_version = None
    try:
        from transformers import AutoTokenizer, __version__ as transformers_version
        model = load_model(); tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        rows=[]; acc={k:0 for k in ["native","restored","replay","fresh"]}; max_restore=0.0; max_replay=0.0
        for fx in FIXTURES:
            pids = tok(fx["prefix"],return_tensors="pt",add_special_tokens=False).input_ids
            with torch.no_grad():
                native_state = clone_state(model(pids,use_cache=True,return_dict=True).cache_params)
                replay_state = clone_state(model(pids,use_cache=True,return_dict=True).cache_params)
            restored_state = clone_state(native_state)
            conds={
                "native":condition(model,tok,native_state,fx),
                "restored":condition(model,tok,restored_state,fx),
                "replay":condition(model,tok,replay_state,fx),
                "fresh":condition(model,tok,None,fx),
            }
            for k in acc: acc[k]+=int(conds[k]["correct"])
            for c in fx["candidates"]:
                max_restore=max(max_restore,abs(conds["native"]["scores"][c]-conds["restored"]["scores"][c]))
                max_replay=max(max_replay,abs(conds["native"]["scores"][c]-conds["replay"]["scores"][c]))
            rows.append({"pair":fx["pair"],"variant":fx["variant"],"expected":fx["expected"],"conditions":conds,"native_minus_fresh_margin":conds["native"]["expected_margin"]-conds["fresh"]["expected_margin"]})
        pair_summary={}
        for pair in sorted({r["pair"] for r in rows}):
            rr=sorted([r for r in rows if r["pair"]==pair],key=lambda x:x["variant"])
            pair_summary[pair]={}
            for cond in acc:
                choices=[r["conditions"][cond]["chosen"] for r in rr]
                pair_summary[pair][cond]={"choices":choices,"choice_flips_with_state":choices[0]!=choices[1],"both_variants_correct":all(r["conditions"][cond]["correct"] for r in rr)}
        n=len(rows)
        report={"status":"XLSTM_164M_REAL_WEIGHT_SEMANTIC_BALANCED_PROBE","repo":REPO_ID,"checkpoint":CKPT_DIR,"torch_version":torch.__version__,"transformers_version":transformers_version,"fixture_count":n,"accuracy":{k:v/n for k,v in acc.items()},"max_native_vs_restored_score_diff":max_restore,"max_native_vs_replay_score_diff":max_replay,"native_pairs_correct_and_flipping":sum(int(v["native"]["both_variants_correct"] and v["native"]["choice_flips_with_state"]) for v in pair_summary.values()),"fresh_pairs_flipping":sum(int(v["fresh"]["choice_flips_with_state"]) for v in pair_summary.values()),"pair_summary":pair_summary,"rows":rows,"claim_boundary":"Actual released ~164M xLSTM checkpoint on a tiny balanced forced-choice state probe. It tests whether active state affects task-relevant choices and survives serialization; not general reasoning or VELA identity evidence."}
        write_report(report)
        if max_restore > 1e-5: raise SystemExit(1)
    except BaseException as exc:
        if isinstance(exc,SystemExit) and exc.code==0: raise
        write_report({"status":"XLSTM_164M_BEHAVIOR_ERROR","repo":REPO_ID,"checkpoint":CKPT_DIR,"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-35:],"claim_boundary":"Runtime/setup failure only; no xLSTM behavior verdict."}); raise

if __name__=="__main__": run()
