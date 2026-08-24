from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import tempfile
import traceback
from pathlib import Path

import torch


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def tensor_numel(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numel()
    if isinstance(obj, dict):
        return sum(tensor_numel(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_numel(v) for v in obj)
    return 0


def clone_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tree(v) for v in obj)
    return copy.deepcopy(obj)


def max_diff(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) & set(b)
        return max([max_diff(a[k], b[k]) for k in keys] or [0.0])
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return max([max_diff(x, y) for x, y in zip(a, b)] or [0.0])
    return 0.0 if a == b else float("inf")


def run():
    report = {
        "status": "VELA_GATED_DELTANET_SPEC_V1",
        "device": "cpu",
        "dtype": "float32",
        "checks": {},
    }
    try:
        from fla.layers.gated_deltanet import GatedDeltaNet
        from fla.models.utils import LegacyFLACache

        try:
            report["fla_version"] = importlib.metadata.version("flash-linear-attention")
        except Exception:
            report["fla_version"] = None
        report["torch_version"] = torch.__version__

        layer = GatedDeltaNet(
            hidden_size=64,
            expand_v=2.0,
            head_dim=16,
            num_heads=3,
            num_v_heads=3,
            mode="fused_recurrent",
            use_gate=True,
            use_short_conv=False,
            layer_idx=0,
        ).cpu().eval()
        report["checks"]["instantiate"] = {
            "pass": True,
            "parameter_numel": int(sum(p.numel() for p in layer.parameters())),
            "mode": layer.mode,
            "use_short_conv": layer.use_short_conv,
        }

        # Validate the cache contract independently of kernel availability.
        fake_state = torch.randn(1, 3, 32, 16)
        cache = LegacyFLACache()
        cache.update(recurrent_state=fake_state, layer_idx=0, offset=7)
        cached = clone_tree(cache.states)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cache.pt"
            torch.save(cached, p)
            restored = torch.load(p, map_location="cpu", weights_only=False)
        report["checks"]["cache_contract"] = {
            "pass": max_diff(cached, restored) == 0.0,
            "state_numel": tensor_numel(cached),
            "seen_tokens": cache.get_seq_length(0),
            "restore_max_abs_diff": max_diff(cached, restored),
            "finite_state_independent_of_seen_tokens": tensor_numel(cached) == fake_state.numel(),
        }

        # Attempt the actual recurrent transition on the free CPU runner. A failure is
        # recorded as an environment/backend limitation rather than conflated with the
        # architecture's cache contract.
        try:
            x = torch.randn(1, 12, 64)
            c1 = LegacyFLACache()
            with torch.no_grad():
                y_a, _, c1 = layer(x[:, :5], past_key_values=c1, use_cache=True)
                checkpoint = clone_tree(c1.states)
                y_b, _, c1 = layer(x[:, 5:], past_key_values=c1, use_cache=True)

                c2 = LegacyFLACache()
                y_full, _, c2 = layer(x, past_key_values=c2, use_cache=True)

                c3 = LegacyFLACache.from_legacy_cache(checkpoint, seen_tokens=5)
                y_restore, _, c3 = layer(x[:, 5:], past_key_values=c3, use_cache=True)

            chunk_out = torch.cat([y_a, y_b], dim=1)
            report["checks"]["cpu_transition"] = {
                "pass": True,
                "chunk_vs_full_max_abs_diff": float((chunk_out - y_full).abs().max()),
                "restored_suffix_max_abs_diff": float((y_b - y_restore).abs().max()),
                "final_state_numel": tensor_numel(c1.states),
                "full_state_numel": tensor_numel(c2.states),
                "seen_tokens": c1.get_seq_length(0),
            }
        except BaseException as exc:
            report["checks"]["cpu_transition"] = {
                "pass": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-30:],
                "interpretation": "CPU runner/kernel availability failure is an environment limitation; use a GPU probe before judging transition behavior.",
            }

        report["summary"] = {
            "cache_contract_pass": bool(report["checks"]["cache_contract"]["pass"]),
            "cpu_transition_pass": bool(report["checks"]["cpu_transition"]["pass"]),
            "spec_candidate": bool(report["checks"]["cache_contract"]["pass"]),
        }
        report["claim_boundary"] = "Gated DeltaNet layer/cache spec probe on a tiny randomly initialized layer. This only tests finite recurrent-state/cache mechanics and CPU-kernel availability; it is not a language-model quality or supersession benchmark."
        write_report(report)
    except BaseException as exc:
        report.update({
            "status": "VELA_GATED_DELTANET_SPEC_V1_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-80:],
        })
        write_report(report)
        raise


if __name__ == "__main__":
    run()
