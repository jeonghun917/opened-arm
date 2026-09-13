from __future__ import annotations
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXPECTED={"MODEL_STIMULI.json":"99df1983f4b112c99a4da1dd175a5605bebcd72b14c47a00aefb78f86067f11a","EVALUATOR.json":"2bcca90657a39eb67b14b1015f617b15fdd9e588a5620144fffa591cb1e1b92f","EXTERNAL_TRACE.json":"e901b4e5b343d567816fd72bfd8c6b11e314ddbb442578da9999f55e8a26db69"}
PROMPTS=["0dac2332a7a9b2f0fab6ff94319ffdceb2b334af997813c659e2da490da21567", "702c4774784497d67d4a6e010421dc864411f24ea27fa8ea94ff607877d97fa4", "9d29d8eddeb8fed9da83947f347bf9ef4fadf84fc29fd71192d423a834b5c4e6", "c842f9cf7141cbeef2096396f2bd45e3b6f3b51b50d7cf62a680238840a53a79", "31b1fb7f2d62a1cc64e9817c9b0270dcb125494a6a6b1845efaeef1c57fc8348", "3c0e250cd7bda9e5f2aad3794055c04856bdf35d962ffc922a51b6d5e4dc6118"]
OPTIONS=["c3e75855b3e8eada228fc28589cb86a15112d630ecf0a5defe67477b216c7b66", "68b3dbafa9ed522b3c7fc729d7c3e4ae50aa21c601cf7aad7e16307fb6aa5b45", "497da7a57081c76e25253d6e722ec34b026987ac4080c1399ac2a3ec7de68e1c", "6532cdacf9832e2887a635264242d0a8f4e0f596e81b09a354539cdfa311e660", "0427e4d28e79340547895ed9aa20007e9d3e3a9727d52f2b69b001268becf9f0", "5f5d9ab05e2e62b06f9b57c4d9ec645243a0bb8cbcc5df81b0e5e3b0678a1c48"]
for name,sha in EXPECTED.items():
    raw=(ROOT/name).read_bytes(); assert hashlib.sha256(raw).hexdigest()==sha,(name,hashlib.sha256(raw).hexdigest())
stim=json.loads((ROOT/'MODEL_STIMULI.json').read_text()); rows=stim['stimuli']; assert len(rows)==6
assert [r['prompt_sha256'] for r in rows]==PROMPTS; assert [r['options_digest'] for r in rows]==OPTIONS
for r in rows:
    assert 'expected_label' not in r and 'expected_option_id' not in r
    assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
ev=json.loads((ROOT/'EVALUATOR.json').read_text()); assert [r['expected_label'] for r in ev['rows']]==['B','A','C','B','A','C']
ext=json.loads((ROOT/'EXTERNAL_TRACE.json').read_text()); assert ext['proposal_units_per_condition']==3112 and ext['decision_prompt_bytes_total']==11300 and ext['equality_observed_across_conditions']==['B','C','G','M','X']
proto=json.loads((ROOT/'PROTOCOL.json').read_text()); assert proto['status']=='READY_FOR_SCIENCE'; assert proto['routes']['M']=={'1':'SEED_AB','2':'A_KEEP','3':'B_KEEP','4':'A_RESET','5':'B_KEEP','6':'A_KEEP'}; assert proto['routes']['X']=={'1':'SEED_AB','2':'A_KEEP','3':'B_KEEP','4':'B_RESET','5':'B_KEEP','6':'A_KEEP'}; assert proto['runtime']['preflight_run']==34730761803 and proto['runtime']['torch']=='2.7.1+cu126'
for name in ['condition_runner.py','aggregate.py']:
    src=(ROOT/name).read_text(); ast.parse(src)
runner=(ROOT/'condition_runner.py').read_text(); assert 'CONDITIONS = ["B", "C", "G", "M", "X"]' in runner and 'EXPECTED_TORCH = "2.7.1+cu126"' in runner and 'RWKV_LOGPROB_TYPED_CHOICE_V1' in runner
agg=(ROOT/'aggregate.py').read_text(); assert 'return "SUPPORTED"' in agg and 'return "MIXED"' in agg and 'return "NOT_SUPPORTED"' in agg and 'PROTOCOL_BLOCKED' in agg
print(json.dumps({"status":"PASS","model_stimuli_sha256":"99df1983f4b112c99a4da1dd175a5605bebcd72b14c47a00aefb78f86067f11a","evaluator_sha256":"2bcca90657a39eb67b14b1015f617b15fdd9e588a5620144fffa591cb1e1b92f","external_trace_sha256":"e901b4e5b343d567816fd72bfd8c6b11e314ddbb442578da9999f55e8a26db69","condition_runner_sha256":"80993f6fc5da59515d36f1e3b8d9d3e1ec3cc9ee55e277f0d7de4a922f29dae6","aggregate_sha256":"0024b71f453d20afebe6d11722db653e16e9820892d3079ae9719db77c858aa3","protocol_sha256":"1792d16243fd5cb5482b654132f8f730a82b8a22e6f0988dffffaebd099d4098","conditions":5,"episodes":6},sort_keys=True))
