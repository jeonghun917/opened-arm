from __future__ import annotations
import hashlib,json

def snap(state): return [x.detach().contiguous().cpu().clone() for x in state]
def clone_snap(state): return [x.clone() for x in state]
def restore(model,snapshot):
 zero=model.generate_zero_state()
 if len(zero)!=len(snapshot): raise RuntimeError('state tensor count drift')
 return [x.to(device=z.device,dtype=z.dtype).contiguous() for x,z in zip(snapshot,zero)]
def state_digest(snapshot):
 import torch
 h=hashlib.sha256()
 for i,tensor in enumerate(snapshot):
  t=tensor.detach().contiguous().cpu(); h.update(i.to_bytes(4,'big')); h.update(str(t.dtype).encode()); h.update(json.dumps(list(t.shape),separators=(',',':')).encode()); h.update(t.view(torch.uint8).numpy().tobytes(order='C'))
 return h.hexdigest()
def feed_text(model,pipe,snapshot,text):
 tokens=[int(x) for x in pipe.encode(text)]
 if not tokens: raise RuntimeError('zero-token input')
 logits,next_state=model.forward(tokens,restore(model,snapshot)); return logits,snap(next_state),len(tokens)
def _answer_region(text):
 last_open=text.rfind('<think>'); last_close=text.rfind('</think>')
 if last_open>=0 and last_close<last_open:return None
 if last_close>=0:return text[last_close+len('</think>'):]
 return text
def extract_json(text):
 region=_answer_region(text)
 if region is None:return None
 for start,ch0 in enumerate(region):
  if ch0!='{':continue
  depth=0;ins=False;esc=False
  for i,ch in enumerate(region[start:],start):
   if ins:
    if esc:esc=False
    elif ch=='\\':esc=True
    elif ch=='"':ins=False
    continue
   if ch=='"':ins=True
   elif ch=='{':depth+=1
   elif ch=='}':
    depth-=1
    if depth==0:
     try:obj=json.loads(region[start:i+1])
     except Exception:break
     if isinstance(obj,dict):return obj
     break
 return None
def greedy_json_probe(model,pipe,snapshot,prompt,max_new_tokens):
 import torch
 tokens=[int(x) for x in pipe.encode(prompt)]
 if not tokens: raise RuntimeError('zero-token probe')
 logits,state=model.forward(tokens,restore(model,snapshot));generated=[];stop_reason='MAX_NEW_TOKENS'
 for _ in range(max_new_tokens):
  v=logits if getattr(logits,'ndim',1)==1 else logits[-1];tok=int(torch.argmax(v.float()).item())
  if tok==0:
   stop_reason='EOS';break
  generated.append(tok);text=pipe.decode(generated);obj=extract_json(text)
  if obj is not None:return obj,text,len(tokens)+len(generated),'FINAL_JSON'
  logits,state=model.forward([tok],state)
 return None,pipe.decode(generated),len(tokens)+len(generated),stop_reason
