from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import traceback
import urllib.request
from pathlib import Path

import torch

SOURCE_COMMIT = "524481d5099b38d9bc8ef1e89209161b86c8011b"
SOURCE_PATH = "RWKV-v7/rwkv_v7_demo_rnn.py"
SOURCE_URL = f"https://raw.githubusercontent.com/BlinkDL/RWKV-LM/{SOURCE_COMMIT}/{SOURCE_PATH}"
VOCAB_URL = f"https://raw.githubusercontent.com/BlinkDL/RWKV-LM/{SOURCE_COMMIT}/RWKV-v7/rwkv_vocab_v20230424.txt"
WEIGHT_REPO = "BlinkDL/rwkv7-g1"
WEIGHT_REVISION = "1e5090cdd819629ff8755e0e04f4db83f0bb9dbb"
WEIGHT_FILE = "rwkv7-g1a-0.1b-20250728-ctx4096.pth"

FIXTURES = [
    {"pair":"codeword","variant":"A","prefix":"Codeword: ALPHA. BETA is inactive.\n","suffix":"Codeword:","candidates":[" ALPHA"," BETA"],"expected":" ALPHA"},
    {"pair":"codeword","variant":"B","prefix":"Codeword: BETA. ALPHA is inactive.\n","suffix":"Codeword:","candidates":[" ALPHA"," BETA"],"expected":" BETA"},
    {"pair":"correction","variant":"A","prefix":"Initial color: RED.\nCorrection: final color is BLUE.\n","suffix":"Final color:","candidates":[" BLUE"," RED"],"expected":" BLUE"},
    {"pair":"correction","variant":"B","prefix":"Initial color: BLUE.\nCorrection: final color is RED.\n","suffix":"Final color:","candidates":[" BLUE"," RED"],"expected":" RED"},
    {"pair":"scope","variant":"A","prefix":"Mars is in scope. Venus is out of scope.\n","suffix":"Record in scope:","candidates":[" Mars"," Venus"],"expected":" Mars"},
    {"pair":"scope","variant":"B","prefix":"Venus is in scope. Mars is out of scope.\n","suffix":"Record in scope:","candidates":[" Mars"," Venus"],"expected":" Venus"},
    {"pair":"hypothesis","variant":"A","prefix":"Hypothesis one is supported. Hypothesis two is rejected.\n","suffix":"Supported hypothesis:","candidates":[" one"," two"],"expected":" one"},
    {"pair":"hypothesis","variant":"B","prefix":"Hypothesis two is supported. Hypothesis one is rejected.\n","suffix":"Supported hypothesis:","candidates":[" one"," two"],"expected":" two"},
    {"pair":"plan","variant":"A","prefix":"Plan: collect, verify, commit.\nCompleted: collect, verify.\n","suffix":"Next step:","candidates":[" commit"," collect"],"expected":" commit"},
    {"pair":"plan","variant":"B","prefix":"Plan: collect, verify, commit.\nCompleted: none.\n","suffix":"Next step:","candidates":[" commit"," collect"],"expected":" collect"},
]


def write_report(report: dict) -> None:
    path = os.environ.get("VELA_RESULT_PATH")
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RWKVTokenizer:
    def __init__(self, vocab_path: str):
        self.idx2token = {}
        ordered = []
        for line in open(vocab_path, "r", encoding="utf-8"):
            idx = int(line[:line.index(" ")])
            token = eval(line[line.index(" "):line.rindex(" ")])
            token = token.encode("utf-8") if isinstance(token, str) else token
            ordered.append(token)
            self.idx2token[idx] = token
        self.token2idx = {v:k for k,v in self.idx2token.items()}
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]
        for token in reversed(ordered):
            if len(token) >= 2:
                a,b = int(token[0]), int(token[1])
                self.table[a][b].append(token)
                self.wlen[a] = max(self.wlen[a], len(token))
                self.good[a].add(b)

    def encode_bytes(self, src: bytes):
        out=[]; i=0
        while i < len(src):
            s=src[i:i+1]
            if i < len(src)-1:
                a,b=int(src[i]),int(src[i+1])
                if b in self.good[a]:
                    ss=src[i:i+self.wlen[a]]
                    try: s=next(filter(ss.startswith,self.table[a][b]))
                    except StopIteration: pass
            out.append(self.token2idx[s]); i += len(s)
        return out

    def encode(self, text: str):
        return self.encode_bytes(text.encode("utf-8"))


def load_official_reference(weight_path: str):
    source = urllib.request.urlopen(SOURCE_URL, timeout=60).read().decode("utf-8")
    source = source.split("# RWKV Tokenizer (slow version)")[0]
    source = source.replace("MyModule = torch.jit.ScriptModule", "MyModule = nn.Module")
    source = source.replace("MyFunction = torch.jit.script_method", "MyFunction = (lambda f: f)")
    source = source.replace("MyStatic = torch.jit.script", "MyStatic = (lambda f: f)")
    source = source.replace("DTYPE = torch.half # better", "DTYPE = torch.float32")
    source = source.replace("map_location='cuda'", "map_location='cpu'")
    source = re.sub(r"try:\n\s*time_mixing = torch\.compile\(time_mixing__,.*?\nexcept:\n\s*time_mixing = torch\.jit\.script\(time_mixing__\)", "time_mixing = time_mixing__", source, flags=re.S)
    source = re.sub(r"try:\n\s*channel_mixing = torch\.compile\(channel_mixing__,.*?\nexcept:\n\s*channel_mixing = torch\.jit\.script\(channel_mixing__\)", "channel_mixing = channel_mixing__", source, flags=re.S)
    ns={"__name__":"rwkv7_pinned_reference"}
    exec(compile(source,SOURCE_PATH,"exec"),ns,ns)
    args=ns["args"]
    args.MODEL_NAME=weight_path.removesuffix(".pth"); args.n_layer=12; args.n_embd=768; args.vocab_size=65536; args.head_size=64
    ns["DTYPE"]=torch.float32
    model=ns["RWKV_RNN"](args).cpu().eval()
    return model,args


def zero_state(args,model):
    state=[None]*(args.n_layer*3)
    for i in range(args.n_layer):
        state[i*3+0]=torch.zeros(args.n_embd,dtype=torch.float32)
        state[i*3+1]=torch.zeros((model.n_head,args.head_size,args.head_size),dtype=torch.float32)
        state[i*3+2]=torch.zeros(args.n_embd,dtype=torch.float32)
    return state


def run_tokens(model,tokens,state):
    out=None
    for token in tokens:
        out,state=model.forward(int(token),state)
    return out.detach().float().cpu(),state


def score_candidate(model,tok,state,suffix,candidate):
    suffix_ids=tok.encode(suffix); cand_ids=tok.encode(candidate)
    if not suffix_ids or not cand_ids: raise RuntimeError("empty tokenization")
    st=copy.deepcopy(state)
    out,st=run_tokens(model,suffix_ids,st)
    total=0.0
    for i,tid in enumerate(cand_ids):
        lp=torch.log_softmax(out.float(),dim=-1)
        total += float(lp[int(tid)])
        if i < len(cand_ids)-1:
            out,st=run_tokens(model,[tid],st)
    return total


def score_condition(model,tok,state,fx):
    scores={c:score_candidate(model,tok,state,fx["suffix"],c) for c in fx["candidates"]}
    chosen=max(scores,key=scores.get); other=next(c for c in fx["candidates"] if c!=fx["expected"])
    return {"scores":scores,"chosen":chosen,"correct":chosen==fx["expected"],"expected_margin":scores[fx["expected"]]-scores[other]}


def run():
    try:
        from huggingface_hub import hf_hub_download
        weight_path=hf_hub_download(repo_id=WEIGHT_REPO,filename=WEIGHT_FILE,revision=WEIGHT_REVISION)
        weight_sha=sha256_file(weight_path)
        model,args=load_official_reference(weight_path)
        with tempfile.TemporaryDirectory() as td:
            vocab_path=Path(td)/"rwkv_vocab_v20230424.txt"
            urllib.request.urlretrieve(VOCAB_URL,vocab_path)
            tok=RWKVTokenizer(str(vocab_path))
            rows=[]; max_restore=0.0; max_replay=0.0; acc={c:0 for c in ("native","restored","replay","fresh")}
            for idx,fx in enumerate(FIXTURES):
                prefix_ids=tok.encode(fx["prefix"])
                with torch.no_grad():
                    _,base=run_tokens(model,prefix_ids,zero_state(args,model))
                    _,replay=run_tokens(model,prefix_ids,zero_state(args,model))
                cp=Path(td)/f"state_{idx}.pt"; torch.save(base,cp); restored=torch.load(cp,map_location="cpu",weights_only=False)
                conds={
                    "native":score_condition(model,tok,base,fx),
                    "restored":score_condition(model,tok,restored,fx),
                    "replay":score_condition(model,tok,replay,fx),
                    "fresh":score_condition(model,tok,zero_state(args,model),fx),
                }
                for cond in conds: acc[cond]+=int(conds[cond]["correct"])
                for c in fx["candidates"]:
                    max_restore=max(max_restore,abs(conds["native"]["scores"][c]-conds["restored"]["scores"][c]))
                    max_replay=max(max_replay,abs(conds["native"]["scores"][c]-conds["replay"]["scores"][c]))
                rows.append({"pair":fx["pair"],"variant":fx["variant"],"expected":fx["expected"],"prefix_tokens":len(prefix_ids),"conditions":conds})

        pair_summary={}
        for pair in sorted({r["pair"] for r in rows}):
            rr=sorted([r for r in rows if r["pair"]==pair],key=lambda r:r["variant"]); pair_summary[pair]={}
            for cond in ("native","restored","replay","fresh"):
                choices=[r["conditions"][cond]["chosen"] for r in rr]
                pair_summary[pair][cond]={"choices":choices,"choice_flips_with_state":choices[0]!=choices[1],"both_variants_correct":all(r["conditions"][cond]["correct"] for r in rr)}
        n=len(rows)
        report={
            "status":"RWKV7_REAL_WEIGHT_SEMANTIC_BALANCED_PROBE",
            "source_commit":SOURCE_COMMIT,"weight_revision":WEIGHT_REVISION,"weight_file":WEIGHT_FILE,"weight_sha256":weight_sha,
            "device":"cpu","dtype":"float32","fixture_count":n,"accuracy":{k:v/n for k,v in acc.items()},
            "max_native_vs_restored_score_diff":max_restore,"max_native_vs_replay_score_diff":max_replay,
            "native_pairs_correct_and_flipping":sum(1 for v in pair_summary.values() if v["native"]["both_variants_correct"] and v["native"]["choice_flips_with_state"]),
            "fresh_pairs_flipping":sum(1 for v in pair_summary.values() if v["fresh"]["choice_flips_with_state"]),
            "pair_summary":pair_summary,"rows":rows,
            "claim_boundary":"Actual released RWKV-7 0.1B weights + pinned official RNN equations and tokenizer. Small semantic state smoke only; not general reasoning, identity, or engine-selection evidence."
        }
        write_report(report)
        if max_restore>1e-5 or max_replay>1e-5: raise SystemExit(1)
    except BaseException as exc:
        write_report({"status":"RWKV7_REAL_WEIGHT_BEHAVIOR_ERROR","source_commit":SOURCE_COMMIT,"weight_revision":WEIGHT_REVISION,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-28:],"claim_boundary":"Runtime/reference-adaptation failure only; no RWKV-7 architecture verdict."})
        raise

if __name__=="__main__": run()
