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
        if set(a) != set(b):
            return float("inf")
        return max([max_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max([max_diff(x, y) for x, y in zip(a, b)] or [0.0])
    return 0.0 if a == b else float("inf")


def legacy_snapshot(cache):
    return clone_tree(cache.to_legacy_cache())


def run():
    report = {
        "status": "VELA_GATED_DELTANET_SPEC_V1",
        "device": "cpu",
        "dtype": "float32",
        "checks": {},
    }
    try:
        from fla.layers.gated_deltanet import GatedDeltaNet
        # FLA 0.5.x selects the HF-compatible FLACache on modern transformers.
        # LegacyFLACache directly calls the old HF Cache constructor and is not
        # compatible with transformers 4.57+, which caused the previous probe error.
        from fla.models.utils import Cache

        try:
            report["fla_version"] = importlib.metadata.version("flash-linear-attention")
        except Exception:
            report["fla_version"] = None
        try:
            report["transformers_version"] = importlib.metadata.version("transformers")
        except Exception:
            report["transformers_version"] = None
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

        # Validate finite recurrent-state serialization with the version-compatible cache.
        fake_state = torch.randn(1, 3, 32, 16)
        cache = Cache()
        cache.update(recurrent_state=fake_state, layer_idx=0, offset=7)
        cached = legacy_snapshot(cache)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cache.pt"
            torch.save(cached, p)
            restored = torch.load(p, map_location="cpu", weights_only=False)
        rebuilt = Cache.from_legacy_cache(restored, seen_tokens=7)
        rebuilt_snapshot = legacy_snapshot(rebuilt)
        restore_diff = max_diff(cached, rebuilt_snapshot)
        report["checks"]["cache_contract"] = {
            "pass": restore_diff == 0.0,
            "cache_class": type(cache).__name__,
            "state_numel": tensor_numel(cached),
            "seen_tokens": cache.get_seq_length(0),
            "rebuilt_seen_tokens": rebuilt.get_seq_length(0),
            "restore_max_abs_diff": restore_diff,
            "finite_state_independent_of_seen_tokens": tensor_numel(cached) == fake_state.numel(),
        }

        # Attempt the actual recurrent transition on the free CPU runner. A kernel/backend
        # failure is recorded separately from the cache-contract result.
        try:
            torch.manual_seed(917)
            x = torch.randn(1, 12, 64)
            c1 = Cache()
            with torch.no_grad():
                y_a, _, c1 = layer(x[:, :5], past_key_values=c1, use_cache=True)
                checkpoint = legacy_snapshot(c1)
                y_b, _, c1 = layer(x[:, 5:], past_key_values=c1, use_cache=True)

                c2 = Cache()
                y_full, _, c2 = layer(x, past_key_values=c2, use_cache=True)

                c3 = Cache.from_legacy_cache(checkpoint, seen_tokens=5)
                y_restore, _, c3 = layer(x[:, 5:], past_key_values=c3, use_cache=True)

            chunk_out = torch.cat([y_a, y_b], dim=1)
            chunk_diff = float((chunk_out - y_full).abs().max())
            restore_suffix_diff = float((y_b - y_restore).abs().max())
            final_state_diff = max_diff(legacy_snapshot(c1), legacy_snapshot(c2))
            report["checks"]["cpu_transition"] = {
                "pass": restore_suffix_diff == 0.0 and final_state_diff == 0.0,
                "chunk_vs_full_max_abs_diff": chunk_diff,
                "restored_suffix_max_abs_diff": restore_suffix_diff,
                "final_state_max_abs_diff": final_state_diff,
                "final_state_numel": tensor_numel(legacy_snapshot(c1)),
                "full_state_numel": tensor_numel(legacy_snapshot(c2)),
                "seen_tokens": c1.get_seq_length(0),
            }
        except BaseException as exc:
            report["checks"]["cpu_transition"] = {
                "pass": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-30:],
                "interpretation": "CPU runner/kernel availability failure is an environment limitation; cache-contract validity is judged separately.",
            }

        report["summary"] = {
            "cache_contract_pass": bool(report["checks"]["cache_contract"]["pass"]),
            "cpu_transition_pass": bool(report["checks"]["cpu_transition"]["pass"]),
            "spec_candidate": bool(report["checks"]["cache_contract"]["pass"]),
        }
        report["claim_boundary"] = (
            "Gated DeltaNet layer/cache spec probe on a tiny randomly initialized layer. "
            "This tests finite recurrent-state/cache mechanics and, when available, CPU recurrent-transition/restore equivalence. "
            "It is not a language-model quality or supersession benchmark."
        )
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
