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
WEIGHT_REPO = "BlinkDL/rwkv7-g1"
WEIGHT_REVISION = "1e5090cdd819629ff8755e0e04f4db83f0bb9dbb"
WEIGHT_FILE = "rwkv7-g1a-0.1b-20250728-ctx4096.pth"


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


def load_official_reference(weight_path: str):
    source = urllib.request.urlopen(SOURCE_URL, timeout=60).read().decode("utf-8")
    # Only execute model/state-transition definitions. No tokenizer, generation loop, or evaluation code.
    source = source.split("# RWKV Tokenizer (slow version)")[0]
    # Make the pinned official reference eager + CPU without changing its equations.
    source = source.replace("MyModule = torch.jit.ScriptModule", "MyModule = nn.Module")
    source = source.replace("MyFunction = torch.jit.script_method", "MyFunction = (lambda f: f)")
    source = source.replace("MyStatic = torch.jit.script", "MyStatic = (lambda f: f)")
    source = source.replace("DTYPE = torch.half # better", "DTYPE = torch.float32")
    source = source.replace("map_location='cuda'", "map_location='cpu'")
    source = re.sub(
        r"try:\n\s*time_mixing = torch\.compile\(time_mixing__,.*?\nexcept:\n\s*time_mixing = torch\.jit\.script\(time_mixing__\)",
        "time_mixing = time_mixing__",
        source,
        flags=re.S,
    )
    source = re.sub(
        r"try:\n\s*channel_mixing = torch\.compile\(channel_mixing__,.*?\nexcept:\n\s*channel_mixing = torch\.jit\.script\(channel_mixing__\)",
        "channel_mixing = channel_mixing__",
        source,
        flags=re.S,
    )
    ns = {"__name__": "rwkv7_pinned_reference"}
    exec(compile(source, SOURCE_PATH, "exec"), ns, ns)
    args = ns["args"]
    args.MODEL_NAME = weight_path.removesuffix(".pth")
    args.n_layer = 12
    args.n_embd = 768
    args.vocab_size = 65536
    args.head_size = 64
    ns["DTYPE"] = torch.float32
    model = ns["RWKV_RNN"](args).cpu().eval()
    return ns, model, args


def zero_state(args, model):
    state = [None for _ in range(args.n_layer * 3)]
    n_head = model.n_head
    for i in range(args.n_layer):
        state[i*3+0] = torch.zeros(args.n_embd, dtype=torch.float32)
        state[i*3+1] = torch.zeros((n_head, args.head_size, args.head_size), dtype=torch.float32)
        state[i*3+2] = torch.zeros(args.n_embd, dtype=torch.float32)
    return state


def run_tokens(model, tokens, state):
    out = None
    for token in tokens:
        out, state = model.forward(int(token), state)
    return out.detach().float().cpu(), state


def run():
    try:
        from huggingface_hub import hf_hub_download
        weight_path = hf_hub_download(repo_id=WEIGHT_REPO, filename=WEIGHT_FILE, revision=WEIGHT_REVISION)
        weight_sha256 = sha256_file(weight_path)
        ns, model, args = load_official_reference(weight_path)

        prefix = [0, 1234, 4321, 777, 2048]
        continuation = [31415]

        with torch.no_grad(), tempfile.TemporaryDirectory() as td:
            init = zero_state(args, model)
            _, state_after_prefix = run_tokens(model, prefix, init)
            cp = Path(td) / "rwkv7_state.pt"
            torch.save(state_after_prefix, cp)

            native_logits, _ = run_tokens(model, continuation, copy.deepcopy(state_after_prefix))
            restored_state = torch.load(cp, map_location="cpu", weights_only=False)
            restored_logits, _ = run_tokens(model, continuation, restored_state)

            replay_logits, _ = run_tokens(model, prefix + continuation, zero_state(args, model))
            fresh_logits, _ = run_tokens(model, continuation, zero_state(args, model))

        restore_diff = float((native_logits-restored_logits).abs().max())
        replay_diff = float((native_logits-replay_logits).abs().max())
        fresh_diff = float((native_logits-fresh_logits).abs().max())
        state_tensors = len(state_after_prefix)
        state_numel = sum(int(t.numel()) for t in state_after_prefix)

        report = {
            "status":"RWKV7_REAL_WEIGHT_M0_PROBE",
            "source_repo":"BlinkDL/RWKV-LM",
            "source_commit":SOURCE_COMMIT,
            "source_path":SOURCE_PATH,
            "weight_repo":WEIGHT_REPO,
            "weight_revision":WEIGHT_REVISION,
            "weight_file":WEIGHT_FILE,
            "weight_sha256":weight_sha256,
            "device":"cpu",
            "dtype":"float32",
            "state_tensor_count":state_tensors,
            "state_numel":state_numel,
            "restore_max_abs_diff":restore_diff,
            "full_replay_max_abs_diff":replay_diff,
            "fresh_max_abs_diff":fresh_diff,
            "restore_equivalent":restore_diff <= 1e-5,
            "full_replay_equivalent":replay_diff <= 1e-5,
            "fresh_is_different":fresh_diff > 1e-5,
            "claim_boundary":"Actual released RWKV-7 0.1B weights with the pinned official RNN transition equations adapted only to eager CPU execution. M0 state-checkpoint evidence only; not reasoning-quality or identity evidence."
        }
        write_report(report)
        if not (report["restore_equivalent"] and report["full_replay_equivalent"] and report["fresh_is_different"]):
            raise SystemExit(1)
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            raise
        write_report({
            "status":"RWKV7_REAL_WEIGHT_M0_ERROR",
            "source_commit":SOURCE_COMMIT,
            "weight_revision":WEIGHT_REVISION,
            "error_type":type(exc).__name__,
            "error":str(exc),
            "traceback_tail":traceback.format_exc().splitlines()[-28:],
            "claim_boundary":"Runtime/reference-adaptation failure only; no RWKV-7 architecture verdict."
        })
        raise


if __name__ == "__main__":
    run()
