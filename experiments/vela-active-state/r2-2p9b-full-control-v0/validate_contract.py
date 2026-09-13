from __future__ import annotations
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
INDEX_BLOB="95b2371ac7bb9416ad5b9a7d33f2df4de40d4d4c"
EVALUATOR_BLOB="55d28e6564396408f293f67ebabfc258e41ae9ef"
EXTERNAL_BLOB="fb056e547feaedfead6b2bb67f87e31449d21466"
EVALUATOR_SHA256="2bcca90657a39eb67b14b1015f617b15fdd9e588a5620144fffa591cb1e1b92f"
EXTERNAL_SHA256="e901b4e5b343d567816fd72bfd8c6b11e314ddbb442578da9999f55e8a26db69"
BLOBS=["9b91ed85c5a3f0e1495fbcda627b660857743235","22e590dd8318ac64cc12bd3d41fc765e6ce7db46","426055ca45b40476e0baa9917c72a7d1aba0332f","fb232b0b6683276d8d323cc306071d7e5cf83f55","8c00d81963e25f998348aac88ebdf92d45077e95","8091870b4f8ac512ac835d9db48f16a80033ad15"]
PROMPTS=["0dac2332a7a9b2f0fab6ff94319ffdceb2b334af997813c659e2da490da21567","702c4774784497d67d4a6e010421dc864411f24ea27fa8ea94ff607877d97fa4","9d29d8eddeb8fed9da83947f347bf9ef4fadf84fc29fd71192d423a834b5c4e6","c842f9cf7141cbeef2096396f2bd45e3b6f3b51b50d7cf62a680238840a53a79","31b1fb7f2d62a1cc64e9817c9b0270dcb125494a6a6b1845efaeef1c57fc8348","3c0e250cd7bda9e5f2aad3794055c04856bdf35d962ffc922a51b6d5e4dc6118"]
OPTIONS=["c3e75855b3e8eada228fc28589cb86a15112d630ecf0a5defe67477b216c7b66","68b3dbafa9ed522b3c7fc729d7c3e4ae50aa21c601cf7aad7e16307fb6aa5b45","497da7a57081c76e25253d6e722ec34b026987ac4080c1399ac2a3ec7de68e1c","6532cdacf9832e2887a635264242d0a8f4e0f596e81b09a354539cdfa311e660","0427e4d28e79340547895ed9aa20007e9d3e3a9727d52f2b69b001268becf9f0","5f5d9ab05e2e62b06f9b57c4d9ec645243a0bb8cbcc5df81b0e5e3b0678a1c48"]
def sha256(b): return hashlib.sha256(b).hexdigest()
def gitblob(b): return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def read(name,blob,sha=None):
 b=(ROOT/name).read_bytes(); assert gitblob(b)==blob,(name,gitblob(b),blob)
 if sha: assert sha256(b)==sha,(name,sha256(b),sha)
 return json.loads(b)
idx=read("MODEL_STIMULI_INDEX.json",INDEX_BLOB); assert idx["format"]=="vela-r2-p-r2-01-frozen-model-facing-stimuli-index-v2-git-blob-bound"; assert [x["git_blob_sha1"] for x in idx["episodes"]]==BLOBS
for i,m in enumerate(idx["episodes"]):
 b=(ROOT/m["file"]).read_bytes(); assert gitblob(b)==BLOBS[i]; r=json.loads(b); assert r["episode"]==i+1 and r["prompt_sha256"]==PROMPTS[i] and r["options_digest"]==OPTIONS[i]; assert sha256(r["prompt"].encode())==PROMPTS[i]; assert "expected_label" not in r and "expected_option_id" not in r
ev=read("EVALUATOR.json",EVALUATOR_BLOB,EVALUATOR_SHA256); ext=read("EXTERNAL_TRACE.json",EXTERNAL_BLOB,EXTERNAL_SHA256); assert [x["expected_label"] for x in ev["rows"]]==["B","A","C","B","A","C"]; assert ext["proposal_units_per_condition"]==3112 and ext["decision_prompt_bytes_total"]==11300 and ext["equality_observed_across_conditions"]==["B","C","G","M","X"]
proto=json.loads((ROOT/"PROTOCOL.json").read_text()); assert proto["status"]=="READY_FOR_SCIENCE"; assert proto["routes"]["M"]=={"1":"SEED_AB","2":"A_KEEP","3":"B_KEEP","4":"A_RESET","5":"B_KEEP","6":"A_KEEP"}; assert proto["routes"]["X"]=={"1":"SEED_AB","2":"A_KEEP","3":"B_KEEP","4":"B_RESET","5":"B_KEEP","6":"A_KEEP"}; assert proto["inputs"]["stimuli_index_git_blob_sha1"]==INDEX_BLOB and proto["inputs"]["stimulus_git_blob_sha1"]==BLOBS; assert proto["runtime"]["torch"]=="2.7.1+cu126" and proto["runtime"]["preflight_run"]==34730761803 and proto["runtime"]["preflight_artifact"]==10309273862
for name in ["science_common.py","condition_runner.py","aggregate.py"]: ast.parse((ROOT/name).read_text())
common=(ROOT/"science_common.py").read_text(); agg=(ROOT/"aggregate.py").read_text(); assert 'CONDITIONS=["B","C","G","M","X"]' in common and 'RWKV_LOGPROB_TYPED_CHOICE_V1' in common and 'EXPECTED_TORCH="2.7.1+cu126"' in common; assert 'return "SUPPORTED"' in agg and 'return "MIXED"' in agg and 'return "NOT_SUPPORTED"' in agg and 'PROTOCOL_BLOCKED' in agg
print(json.dumps({"status":"PASS","stimuli_index_git_blob_sha1":INDEX_BLOB,"stimulus_git_blob_sha1":BLOBS,"evaluator_git_blob_sha1":EVALUATOR_BLOB,"external_trace_git_blob_sha1":EXTERNAL_BLOB,"conditions":5,"episodes":6},sort_keys=True))
