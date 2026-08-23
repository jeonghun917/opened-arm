from __future__ import annotations

import importlib.util
from pathlib import Path

TARGET = Path(__file__).with_name("rwkv7_causal_anchor_v1.py")
spec = importlib.util.spec_from_file_location("vela_rwkv7_causal_anchor_v1", TARGET)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {TARGET}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixed_eval_model(model, ns, tok):
    rows=[]; correct=0; kinds={"correction":[0,0],"control":[0,0]}
    for fx in mod.rw.HELDOUT:
        scores={c:mod.rw.candidate_score(model,ns,tok,mod.rw.zero_state(model.args,model),fx["prompt"],c) for c in fx["candidates"]}
        chosen=max(scores,key=scores.get); ok=chosen==fx["expected"]
        correct+=int(ok); kinds[fx["kind"]][0]+=int(ok); kinds[fx["kind"]][1]+=1
        rows.append({"id":fx["id"],"chosen":chosen,"expected":fx["expected"],"correct":ok,"scores":scores})
    return {"accuracy":correct/len(rows),"correction_accuracy":kinds["correction"][0]/kinds["correction"][1],"control_accuracy":kinds["control"][0]/kinds["control"][1],"rows":rows}

mod.rw.eval_model = fixed_eval_model

if __name__ == "__main__":
    mod.run()
