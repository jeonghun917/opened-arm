from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import traceback
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

SOURCE_COMMIT = "524481d5099b38d9bc8ef1e89209161b86c8011b"
SOURCE_PATH = "RWKV-v7/rwkv_v7_demo_rnn.py"
SOURCE_URL = f"https://raw.githubusercontent.com/BlinkDL/RWKV-LM/{SOURCE_COMMIT}/{SOURCE_PATH}"
VOCAB_URL = f"https://raw.githubusercontent.com/BlinkDL/RWKV-LM/{SOURCE_COMMIT}/RWKV-v7/rwkv_vocab_v20230424.txt"
WEIGHT_REPO = "BlinkDL/rwkv7-g1"
WEIGHT_REVISION = "1e5090cdd819629ff8755e0e04f4db83f0bb9dbb"
WEIGHT_FILE = "rwkv7-g1a-0.1b-20250728-ctx4096.pth"
SEED = 917
LR = 5e-5
EPOCHS = 2

TRAIN_PAIRS = [(" CAT"," DOG"),(" EAST"," WEST"),(" UP"," DOWN"),(" HOT"," COLD"),(" LEFT"," RIGHT"),(" ON"," OFF")]
TEMPLATES = [
    "Initial value:{old}. Correction: the final value is{new}, not{old}.\nFinal value:",
    "Previous answer:{old}. Update: replace it with{new}.\nCurrent answer:",
]
TRAIN=[]
for i,(a,b) in enumerate(TRAIN_PAIRS):
    t=TEMPLATES[i%len(TEMPLATES)]
    TRAIN.append((t.format(old=a,new=b),b)); TRAIN.append((t.format(old=b,new=a),a))

HELDOUT=[
    {"id":"color-A","prompt":"Initial value: RED. Correction: the final value is BLUE, not RED.\nFinal value:","candidates":[" BLUE"," RED"],"expected":" BLUE","kind":"correction"},
    {"id":"color-B","prompt":"Initial value: BLUE. Correction: the final value is RED, not BLUE.\nFinal value:","candidates":[" BLUE"," RED"],"expected":" RED","kind":"correction"},
    {"id":"code-A","prompt":"Previous answer: ALPHA. Update: replace it with BETA.\nCurrent answer:","candidates":[" BETA"," ALPHA"],"expected":" BETA","kind":"correction"},
    {"id":"code-B","prompt":"Previous answer: BETA. Update: replace it with ALPHA.\nCurrent answer:","candidates":[" BETA"," ALPHA"],"expected":" ALPHA","kind":"correction"},
    {"id":"level-A","prompt":"The old value was LOW. The corrected value is HIGH.\nUse the corrected value:","candidates":[" HIGH"," LOW"],"expected":" HIGH","kind":"correction"},
    {"id":"level-B","prompt":"The old value was HIGH. The corrected value is LOW.\nUse the corrected value:","candidates":[" HIGH"," LOW"],"expected":" LOW","kind":"correction"},
    {"id":"control-A","prompt":"Codeword: ALPHA.\nCodeword:","candidates":[" ALPHA"," BETA"],"expected":" ALPHA","kind":"control"},
    {"id":"control-B","prompt":"Codeword: BETA.\nCodeword:","candidates":[" ALPHA"," BETA"],"expected":" BETA","kind":"control"},
]

HISTORY=("Project Orion remains active. The old codeword was ALPHA. Correction: the current codeword is BETA, not ALPHA. "
         "The verification step is incomplete, so external action remains blocked until verification succeeds. "
         "Hypothesis one is weakened by the latest evidence while hypothesis two remains unresolved.")
ANCHOR="Project Orion remains active. The old codeword was ALPHA. "
PROBES=[
    ("codeword","\nCurrent codeword:",[" BETA"," ALPHA"]," BETA"),
    ("verification","\nVerification status:",[" incomplete"," complete"]," incomplete"),
    ("project","\nProject Orion status:",[" active"," paused"]," active"),
    ("hypothesis","\nHypothesis one is:",[" weakened"," strengthened"]," weakened"),
]


def write_report(report):
    pth=os.environ.get("VELA_RESULT_PATH")
    if pth:
        p=Path(pth); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


class RWKVTokenizer:
    def __init__(self,vocab_path):
        self.idx2token={}; ordered=[]
        for line in open(vocab_path,"r",encoding="utf-8"):
            idx=int(line[:line.index(" ")]); token=eval(line[line.index(" "):line.rindex(" ")]); token=token.encode("utf-8") if isinstance(token,str) else token
            ordered.append(token); self.idx2token[idx]=token
        self.token2idx={v:k for k,v in self.idx2token.items()}; self.table=[[[] for _ in range(256)] for _ in range(256)]; self.good=[set() for _ in range(256)]; self.wlen=[0 for _ in range(256)]
        for token in reversed(ordered):
            if len(token)>=2:
                a,b=int(token[0]),int(token[1]); self.table[a][b].append(token); self.wlen[a]=max(self.wlen[a],len(token)); self.good[a].add(b)
    def encode(self,text):
        src=text.encode("utf-8"); out=[]; i=0
        while i<len(src):
            s=src[i:i+1]
            if i<len(src)-1:
                a,b=int(src[i]),int(src[i+1])
                if b in self.good[a]:
                    ss=src[i:i+self.wlen[a]]
                    try: s=next(filter(ss.startswith,self.table[a][b]))
                    except StopIteration: pass
            out.append(self.token2idx[s]); i+=len(s)
        return out


def load_reference(weight_path):
    source=urllib.request.urlopen(SOURCE_URL,timeout=60).read().decode("utf-8")
    source=source.split("# RWKV Tokenizer (slow version)")[0]
    source=source.replace("MyModule = torch.jit.ScriptModule","MyModule = nn.Module")
    source=source.replace("MyFunction = torch.jit.script_method","MyFunction = (lambda f: f)")
    source=source.replace("MyStatic = torch.jit.script","MyStatic = (lambda f: f)")
    source=source.replace("DTYPE = torch.half # better","DTYPE = torch.float32")
    source=source.replace("map_location='cuda'","map_location='cpu'")
    source=re.sub(r"try:\n\s*time_mixing = torch\.compile\(time_mixing__,.*?\nexcept:\n\s*time_mixing = torch\.jit\.script\(time_mixing__\)","time_mixing = time_mixing__",source,flags=re.S)
    source=re.sub(r"try:\n\s*channel_mixing = torch\.compile\(channel_mixing__,.*?\nexcept:\n\s*channel_mixing = torch\.jit\.script\(channel_mixing__\)","channel_mixing = channel_mixing__",source,flags=re.S)
    ns={"__name__":"rwkv7_grad_reference"}; exec(compile(source,SOURCE_PATH,"exec"),ns,ns)
    args=ns["args"]; args.MODEL_NAME=weight_path.removesuffix(".pth"); args.n_layer=12; args.n_embd=768; args.vocab_size=65536; args.head_size=64; ns["DTYPE"]=torch.float32
    model=ns["RWKV_RNN"](args).cpu().eval()
    return model,args,ns


def zero_state(args,model):
    st=[None]*(args.n_layer*3)
    for i in range(args.n_layer):
        st[i*3]=torch.zeros(args.n_embd); st[i*3+1]=torch.zeros((model.n_head,args.head_size,args.head_size)); st[i*3+2]=torch.zeros(args.n_embd)
    return st


def clone_state(st): return [x.detach().clone() for x in st]


def forward_grad(model,ns,token,state):
    z=model.z; x=z['emb.weight'][int(token)]; v_first=torch.empty_like(x)
    for i in range(model.n_layer):
        bbb=f'blocks.{i}.'; att=f'blocks.{i}.att.'; ffn=f'blocks.{i}.ffn.'
        xx=F.layer_norm(x,(model.n_embd,),weight=z[bbb+'ln1.weight'],bias=z[bbb+'ln1.bias'])
        xx,state[i*3],state[i*3+1],v_first=ns['time_mixing'](i,model.n_head,model.args.head_size,xx,state[i*3],v_first,state[i*3+1],
            z[att+'x_r'],z[att+'x_w'],z[att+'x_k'],z[att+'x_v'],z[att+'x_a'],z[att+'x_g'],z[att+'w0'],z[att+'w1'],z[att+'w2'],z[att+'a0'],z[att+'a1'],z[att+'a2'],z[att+'v0'],z[att+'v1'],z[att+'v2'],z[att+'g1'],z[att+'g2'],z[att+'k_k'],z[att+'k_a'],z[att+'r_k'],z[att+'key.weight'],z[att+'value.weight'],z[att+'receptance.weight'],z[att+'output.weight'],z[att+'ln_x.weight'],z[att+'ln_x.bias'])
        x=x+xx
        xx=F.layer_norm(x,(model.n_embd,),weight=z[bbb+'ln2.weight'],bias=z[bbb+'ln2.bias'])
        xx,state[i*3+2]=ns['channel_mixing'](xx,state[i*3+2],z[ffn+'x_k'],z[ffn+'key.weight'],z[ffn+'value.weight']); x=x+xx
    x=F.layer_norm(x,(model.n_embd,),weight=z['ln_out.weight'],bias=z['ln_out.bias']); return z['head.weight']@x,state


def run_tokens(model,ns,tokens,state):
    out=None
    for tid in tokens: out,state=forward_grad(model,ns,tid,state)
    return out,state


def candidate_score(model,ns,tok,state,suffix,cand):
    st=clone_state(state); sids=tok.encode(suffix); cids=tok.encode(cand)
    with torch.no_grad():
        out,st=run_tokens(model,ns,sids,st); total=0.0
        for j,tid in enumerate(cids):
            total+=float(torch.log_softmax(out.float(),dim=-1)[tid])
            if j+1<len(cids): out,st=run_tokens(model,ns,[tid],st)
    return total


def eval_model(model,ns,tok):
    rows=[]; correct=0; kinds={"correction":[0,0],"control":[0,0]}
    for fx in HELDOUT:
        with torch.no_grad(): _,st=run_tokens(model,ns,tok.encode(fx["prompt"]),zero_state(model.args,model))
        scores={c:candidate_score(model,ns,tok,st,"",c) for c in fx["candidates"]}; chosen=max(scores,key=scores.get); ok=chosen==fx["expected"]
        correct+=int(ok); kinds[fx["kind"]][0]+=int(ok); kinds[fx["kind"]][1]+=1; rows.append({"id":fx["id"],"chosen":chosen,"expected":fx["expected"],"correct":ok,"scores":scores})
    return {"accuracy":correct/len(rows),"correction_accuracy":kinds["correction"][0]/kinds["correction"][1],"control_accuracy":kinds["control"][0]/kinds["control"][1],"rows":rows}


def train_example(model,ns,tok,prompt,target):
    pids=tok.encode(prompt); tids=tok.encode(target); st=zero_state(model.args,model); out=None
    for tid in pids: out,st=forward_grad(model,ns,tid,st)
    if out is None: raise RuntimeError("empty prompt")
    loss=torch.tensor(0.0)
    for j,tid in enumerate(tids):
        loss=loss+F.cross_entropy(out.float().unsqueeze(0),torch.tensor([tid]))
        if j+1<len(tids): out,st=forward_grad(model,ns,tid,st)
    return loss/max(len(tids),1)


def state_distance(a,b):
    sq=0.0; mx=0.0; n=0
    for x,y in zip(a,b):
        d=x.detach().float()-y.detach().float(); sq+=float((d*d).sum()); mx=max(mx,float(d.abs().max())); n+=d.numel()
    return {"rms":math.sqrt(sq/max(n,1)),"l2":math.sqrt(sq),"max_abs":mx,"numel":n}


def probe_state(model,ns,tok,state):
    rows=[]
    for pid,suffix,cands,expected in PROBES:
        scores={c:candidate_score(model,ns,tok,state,suffix,c) for c in cands}; chosen=max(scores,key=scores.get)
        rows.append({"id":pid,"chosen":chosen,"expected":expected,"correct":chosen==expected,"scores":scores})
    return rows


def compare(rows,native):
    nb={r['id']:r for r in native}; agree=sum(int(r['chosen']==nb[r['id']]['chosen']) for r in rows)/len(rows); exp=sum(int(r['correct']) for r in rows)/len(rows)
    return {"decision_agreement":agree,"expected_accuracy":exp}


def run():
    try:
        from huggingface_hub import hf_hub_download
        torch.manual_seed(SEED); random.seed(SEED); torch.set_num_threads(2)
        weight_path=hf_hub_download(repo_id=WEIGHT_REPO,filename=WEIGHT_FILE,revision=WEIGHT_REVISION); weight_sha=sha256_file(weight_path)
        model,args,ns=load_reference(weight_path)
        with tempfile.TemporaryDirectory() as td:
            vp=Path(td)/"vocab.txt"; urllib.request.urlretrieve(VOCAB_URL,vp); tok=RWKVTokenizer(str(vp))
            hist_ids=tok.encode(HISTORY); anchor_ids=tok.encode(ANCHOR)
            with torch.no_grad():
                _,old_full=run_tokens(model,ns,hist_ids,zero_state(args,model)); old_full=clone_state(old_full)
                _,old_anchor=run_tokens(model,ns,anchor_ids,zero_state(args,model)); old_anchor=clone_state(old_anchor)
            baseline=eval_model(model,ns,tok)

            # Train all recurrent attention key projections; these weights participate directly in state transitions.
            trainable=[]
            for i in range(args.n_layer):
                key=f"blocks.{i}.att.key.weight"; model.z[key].requires_grad_(True); trainable.append(model.z[key])
            opt=torch.optim.AdamW(trainable,lr=LR,weight_decay=0.0); order=list(range(len(TRAIN))); losses=[]
            for epoch in range(EPOCHS):
                random.Random(SEED+epoch).shuffle(order); total=0.0
                for idx in order:
                    opt.zero_grad(set_to_none=True); loss=train_example(model,ns,tok,*TRAIN[idx]); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); total+=float(loss.detach())
                losses.append(total/len(order))
            after=eval_model(model,ns,tok)

            with torch.no_grad():
                _,native=run_tokens(model,ns,hist_ids,zero_state(args,model)); native=clone_state(native)
                _,anchored=run_tokens(model,ns,hist_ids[len(anchor_ids):],clone_state(old_anchor)); anchored=clone_state(anchored)
            native_probe=probe_state(model,ns,tok,native); direct_probe=probe_state(model,ns,tok,old_full); anchor_probe=probe_state(model,ns,tok,anchored)
            migration={
                "direct_old_state":{"state_error":state_distance(old_full,native),"functional":compare(direct_probe,native_probe),"probe":direct_probe},
                "before_correction_anchor":{"state_error":state_distance(anchored,native),"functional":compare(anchor_probe,native_probe),"probe":anchor_probe},
                "w2_native":{"state_error":state_distance(native,native),"functional":compare(native_probe,native_probe),"probe":native_probe},
            }

        report={"status":"RWKV7_LEARNED_UPGRADE_MIGRATION_V1","source_commit":SOURCE_COMMIT,"weight_revision":WEIGHT_REVISION,"weight_file":WEIGHT_FILE,"weight_sha256":weight_sha,"training":{"trainable":"all blocks.*.att.key.weight","lr":LR,"epochs":EPOCHS,"train_examples":len(TRAIN),"epoch_loss":losses},"capability":{"baseline":baseline,"after":after,"overall_gain":after['accuracy']-baseline['accuracy'],"correction_gain":after['correction_accuracy']-baseline['correction_accuracy'],"control_gain":after['control_accuracy']-baseline['control_accuracy']},"migration":migration,"success_definition":"Cross-architecture replication requires a real held-out capability gain and compares migrated W2 against W2-native after the same causal history, not against W1 output.","claim_boundary":"Actual released RWKV-7 0.1B weights and pinned official RNN equations, with a narrow learned update to recurrent key projections. Small synthetic evidence only; not a backbone promotion or identity proof."}
        write_report(report)
    except BaseException as exc:
        write_report({"status":"RWKV7_LEARNED_UPGRADE_ERROR","error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-45:]}); raise

if __name__=="__main__": run()
