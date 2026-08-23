from __future__ import annotations

import io
import json
import math
import os
import traceback
from pathlib import Path

import torch

REPO_ID = "NX-AI/xlstm_scaling_laws"
CKPT_DIR = "mlstm_v1--tokenparam--ctx-8192--params-164.11M--tokens-361.76B--id-y5s6gd5v"
TOKENIZER_ID = "NX-AI/xLSTM-7b"


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def clone_state(state):
    buf = io.BytesIO(); torch.save(state, buf); buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


def tensor_refs(obj):
    refs = []; seen = set()
    def walk(x):
        if torch.is_tensor(x):
            if id(x) not in seen: seen.add(id(x)); refs.append(x)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
        elif hasattr(x, "__dict__"):
            for v in vars(x).values(): walk(v)
    walk(obj); return refs


def run():
    transformers_version = None
    try:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, __version__ as transformers_version
        from transformers.models.xlstm.configuration_xlstm import xLSTMConfig
        from transformers.models.xlstm.modeling_xlstm import xLSTMForCausalLM

        root = Path(snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{CKPT_DIR}/*"],
            local_dir="/tmp/vela-xlstm-scaling",
        ))
        ckpt = root / CKPT_DIR
        cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
        cfg["hidden_size"] = cfg.pop("embedding_dim")
        cfg["mode"] = "inference"
        config = xLSTMConfig(**cfg)
        model = xLSTMForCausalLM(config).cpu().eval()

        state_dict = {}
        files = sorted(ckpt.glob("*.safetensors"))
        if not files:
            raise RuntimeError(f"no safetensors found in {ckpt}")
        for f in files:
            state_dict.update(load_file(str(f), device="cpu"))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint key mismatch missing={missing[:8]} unexpected={unexpected[:8]}")

        tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        text = "Project Orion remains active. Verification is incomplete, and the current codeword is BETA."
        prefix = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        cont = tok(" Therefore", return_tensors="pt", add_special_tokens=False).input_ids[:, :1]

        with torch.no_grad():
            pre = model(prefix, use_cache=True, return_dict=True)
            state = pre.cache_params
            if state is None:
                raise RuntimeError("xLSTM returned no cache_params")
            state_saved = clone_state(state)
            native = model(cont, cache_params=clone_state(state_saved), use_cache=True, return_dict=True).logits[:, -1].detach().float()
            restored = model(cont, cache_params=clone_state(state_saved), use_cache=True, return_dict=True).logits[:, -1].detach().float()
            replay = model(torch.cat([prefix, cont], dim=1), use_cache=False, return_dict=True).logits[:, -1].detach().float()
            fresh = model(cont, use_cache=False, return_dict=True).logits[:, -1].detach().float()

        restore_diff = float((native - restored).abs().max())
        replay_diff = float((native - replay).abs().max())
        fresh_diff = float((native - fresh).abs().max())
        refs = tensor_refs(state_saved)
        report = {
            "status":"XLSTM_164M_REAL_WEIGHT_M0_PROBE",
            "repo":REPO_ID,
            "checkpoint":CKPT_DIR,
            "tokenizer":TOKENIZER_ID,
            "torch_version":torch.__version__,
            "transformers_version":transformers_version,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "checkpoint_files":[{"name":f.name,"bytes":f.stat().st_size} for f in files],
            "state_tensor_count":len(refs),
            "state_numel":sum(x.numel() for x in refs),
            "restore_max_abs_diff":restore_diff,
            "full_replay_max_abs_diff":replay_diff,
            "fresh_max_abs_diff":fresh_diff,
            "restore_equivalent":restore_diff <= 1e-5,
            "full_replay_equivalent":replay_diff <= 1e-4,
            "fresh_is_different":fresh_diff > 1e-5,
            "claim_boundary":"Actual released ~164M xLSTM scaling-law checkpoint. M0 active-state checkpoint evidence only; not reasoning-quality, identity, or engine-selection evidence."
        }
        write_report(report)
        if not (report["restore_equivalent"] and report["fresh_is_different"]):
            raise SystemExit(1)
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0: raise
        write_report({"status":"XLSTM_164M_REAL_WEIGHT_ERROR","repo":REPO_ID,"checkpoint":CKPT_DIR,"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-35:],"claim_boundary":"Runtime/setup failure only; do not interpret as an xLSTM architecture failure."})
        raise

if __name__ == "__main__": run()
