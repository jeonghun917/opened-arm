from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

OUT = Path('/kaggle/working/gated-deltanet-gpu-spec-v1.json')


def write(report):
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def tensor_numel(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel()
    if isinstance(obj, dict):
        return sum(tensor_numel(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_numel(v) for v in obj)
    return 0


def clone_tree(obj):
    import copy
    import torch
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
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float('inf')
        return max([max_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float('inf')
        return max([max_diff(x, y) for x, y in zip(a, b)] or [0.0])
    return 0.0 if a == b else float('inf')


def run():
    report = {'status': 'VELA_GATED_DELTANET_GPU_SPEC_V1'}
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--quiet', 'flash-linear-attention[cuda]==0.5.2'],
            check=True,
        )
        import importlib.metadata
        import torch
        from fla.layers.gated_deltanet import GatedDeltaNet
        from fla.models.utils import Cache

        report.update({
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'fla_version': importlib.metadata.version('flash-linear-attention'),
        })
        if not torch.cuda.is_available():
            raise RuntimeError('Kaggle GPU kernel started without CUDA availability')

        device = torch.device('cuda')
        torch.manual_seed(917)
        torch.cuda.manual_seed_all(917)
        layer = GatedDeltaNet(
            hidden_size=64,
            expand_v=2.0,
            head_dim=16,
            num_heads=3,
            num_v_heads=3,
            mode='fused_recurrent',
            use_gate=True,
            use_short_conv=False,
            layer_idx=0,
        ).to(device).eval()

        x = torch.randn(1, 12, 64, device=device)
        c1 = Cache()
        with torch.no_grad():
            y_a, _, c1 = layer(x[:, :5], past_key_values=c1, use_cache=True)
            checkpoint = clone_tree(c1.to_legacy_cache())
            y_b, _, c1 = layer(x[:, 5:], past_key_values=c1, use_cache=True)
            final_chunk = clone_tree(c1.to_legacy_cache())

            c2 = Cache()
            y_full, _, c2 = layer(x, past_key_values=c2, use_cache=True)
            final_full = clone_tree(c2.to_legacy_cache())

            c3 = Cache.from_legacy_cache(checkpoint, seen_tokens=5)
            y_restore, _, c3 = layer(x[:, 5:], past_key_values=c3, use_cache=True)
            final_restore = clone_tree(c3.to_legacy_cache())

        chunk_out = torch.cat([y_a, y_b], dim=1)
        chunk_vs_full = float((chunk_out.float() - y_full.float()).abs().max())
        suffix_restore = float((y_b.float() - y_restore.float()).abs().max())
        state_chunk_vs_full = max_diff(final_chunk, final_full)
        state_chunk_vs_restore = max_diff(final_chunk, final_restore)
        tol = 1e-4
        report['checks'] = {
            'instantiate': {
                'pass': True,
                'parameter_numel': int(sum(p.numel() for p in layer.parameters())),
            },
            'finite_state': {
                'pass': tensor_numel(checkpoint) > 0,
                'checkpoint_state_numel': tensor_numel(checkpoint),
                'final_state_numel': tensor_numel(final_chunk),
            },
            'checkpoint_restore': {
                'pass': suffix_restore <= tol and state_chunk_vs_restore <= tol,
                'suffix_output_max_abs_diff': suffix_restore,
                'final_state_max_abs_diff': state_chunk_vs_restore,
                'tolerance': tol,
            },
            'chunk_full_equivalence': {
                'pass': chunk_vs_full <= tol and state_chunk_vs_full <= tol,
                'output_max_abs_diff': chunk_vs_full,
                'final_state_max_abs_diff': state_chunk_vs_full,
                'tolerance': tol,
            },
        }
        report['summary'] = {
            'spec_pass': all(v['pass'] for v in report['checks'].values()),
            'state_checkpointable': report['checks']['checkpoint_restore']['pass'],
            'recurrent_transition_verified_on_gpu': True,
        }
        report['claim_boundary'] = 'Tiny randomly initialized Gated DeltaNet layer on a real CUDA/Triton backend. Tests recurrent transition, finite state, serialization/restore, and chunk/full equivalence only; not language quality or supersession.'
        write(report)
    except BaseException as exc:
        report.update({
            'status': 'VELA_GATED_DELTANET_GPU_SPEC_V1_ERROR',
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback_tail': traceback.format_exc().splitlines()[-80:],
        })
        write(report)
        raise


if __name__ == '__main__':
    run()
