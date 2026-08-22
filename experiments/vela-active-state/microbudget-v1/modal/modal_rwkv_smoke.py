import modal

app = modal.App("vela-rwkv-state-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.40,<5")
)

@app.function(
    image=image,
    gpu="T4",
    timeout=1200,
)
def smoke():
    import json, tempfile
    from pathlib import Path
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "RWKV/rwkv-4-169m-pile"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device).eval()

    ids = tokenizer(
        "VELA keeps competing hypotheses visible and updates only the affected one.",
        return_tensors="pt",
    ).input_ids.to(device)
    split = max(2, ids.shape[1] // 2)

    with torch.no_grad():
        a = model(ids[:, :split], use_cache=True, return_dict=True)
        state = [x.detach().cpu().clone() for x in a.state]

        native = model(
            ids[:, split:], state=[x.to(device) for x in state],
            use_cache=True, return_dict=True
        ).logits.detach().float().cpu()

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.pt"
            torch.save(state, p)
            reload_state = torch.load(p, map_location="cpu")
            restored = model(
                ids[:, split:], state=[x.to(device) for x in reload_state],
                use_cache=True, return_dict=True
            ).logits.detach().float().cpu()

        fresh = model(
            ids[:, split:], use_cache=True, return_dict=True
        ).logits.detach().float().cpu()

    report = {
        "provider": "modal",
        "model": model_id,
        "restore_max_abs_diff": float((native-restored).abs().max()),
        "fresh_max_abs_diff": float((native-fresh).abs().max()),
    }
    print(json.dumps(report, indent=2))
    return report

@app.local_entrypoint()
def main():
    print(smoke.remote())
