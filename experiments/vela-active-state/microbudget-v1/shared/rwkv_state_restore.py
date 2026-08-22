from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import torch

MODEL_ID = "RWKV/rwkv-4-169m-pile"

def clone_state_to_cpu(state):
    return [x.detach().cpu().clone() for x in state]

def independent_state(state, device):
    # Important on CPU: Tensor.to('cpu') may return the same tensor object.
    # RWKV forward can mutate the supplied recurrent state, so every branch
    # must receive an independent clone from the same pre-continuation state.
    return [x.to(device).clone() for x in state]

def run():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model = model.to(device).eval()
    text = (
        "VELA keeps two hypotheses open while new evidence arrives. "
        "The first hypothesis explains the timing. The second explains the scope. "
        "A correction changes only the first hypothesis, while the second remains unresolved."
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    split = max(2, ids.shape[1] // 2)
    first, second = ids[:, :split], ids[:, split:]

    with torch.no_grad():
        out1 = model(first, use_cache=True, return_dict=True)
        base_state_cpu = clone_state_to_cpu(out1.state)

        # Serialize the exact pre-continuation state before either branch runs.
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "rwkv_state.pt"
            torch.save(base_state_cpu, cp)
            reloaded = torch.load(cp, map_location="cpu")

            native_state = independent_state(base_state_cpu, device)
            restored_state = independent_state(reloaded, device)

            native = model(
                second,
                state=native_state,
                use_cache=True,
                return_dict=True,
            ).logits.detach().float().cpu()

            restored = model(
                second,
                state=restored_state,
                use_cache=True,
                return_dict=True,
            ).logits.detach().float().cpu()

        fresh = model(second, use_cache=True, return_dict=True).logits.detach().float().cpu()

    restore_diff = float((native - restored).abs().max())
    fresh_diff = float((native - fresh).abs().max())
    report = {
        "model": MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "state_tensor_count": len(base_state_cpu),
        "restore_max_abs_diff": restore_diff,
        "fresh_max_abs_diff": fresh_diff,
        "restore_equivalent": restore_diff <= 1e-5,
        "fresh_is_different": fresh_diff > 1e-5,
        "fix_note": "Each continuation receives an independent clone of the same pre-continuation recurrent state; this avoids CPU .to('cpu') aliasing/in-place mutation confounds.",
        "claim_boundary": (
            "This tests real RWKV recurrent-state checkpoint/restore equivalence. "
            "It does not establish VELA identity or overall cognitive superiority."
        ),
    }
    result_path = os.environ.get("VELA_RESULT_PATH")
    if result_path:
        p = Path(result_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not (report["restore_equivalent"] and report["fresh_is_different"]):
        raise SystemExit(1)
    return report

if __name__ == "__main__":
    run()
