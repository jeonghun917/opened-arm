from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path

import torch

XLSTM_COMMIT = "f539ba80770ba2b9acd5bf4c1e0f0d4827494184"


def write_report(report: dict) -> None:
    path = os.environ.get("VELA_RESULT_PATH")
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def run():
    try:
        from xlstm.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig

        torch.manual_seed(917)
        cfg = xLSTMLargeConfig(
            embedding_dim=32,
            num_heads=4,
            num_blocks=2,
            vocab_size=64,
            return_last_states=True,
            mode="inference",
            chunkwise_kernel="chunkwise--native_autograd",
            sequence_kernel="native_sequence__native",
            step_kernel="native",
            chunk_size=4,
            inference_state_dtype="float32",
            autocast_kernel_dtype="float32",
        )
        model = xLSTMLarge(cfg).cpu().eval()

        prefix = torch.tensor([[1, 5, 7, 9, 13, 17]], dtype=torch.long)
        continuation = torch.tensor([[19]], dtype=torch.long)

        with torch.no_grad():
            _, state = model(prefix, state=None)
            if not isinstance(state, dict) or not state:
                raise RuntimeError(f"unexpected xLSTM state object: {type(state).__name__}")

            state_tensor_count = sum(len(layer_state) for layer_state in state.values() if layer_state is not None)
            state_shapes = {
                str(layer): [list(t.shape) for t in layer_state]
                for layer, layer_state in state.items()
                if layer_state is not None
            }

            with tempfile.TemporaryDirectory() as td:
                cp = Path(td) / "xlstm_state.pt"
                torch.save(state, cp)
                native_logits, _ = model(continuation, state=state)
                restored_state = torch.load(cp, map_location="cpu", weights_only=False)
                restored_logits, _ = model(continuation, state=restored_state)

            replay_logits, _ = model(torch.cat([prefix, continuation], dim=1), state=None)
            fresh_logits, _ = model(continuation, state=None)

        native_last = native_logits[:, -1].detach().float()
        restored_last = restored_logits[:, -1].detach().float()
        replay_last = replay_logits[:, -1].detach().float()
        fresh_last = fresh_logits[:, -1].detach().float()

        restore_diff = float((native_last - restored_last).abs().max())
        replay_diff = float((native_last - replay_last).abs().max())
        fresh_diff = float((native_last - fresh_last).abs().max())

        report = {
            "status": "XLSTM_ARCH_STATE_PROBE",
            "source_commit": XLSTM_COMMIT,
            "device": "cpu",
            "weights": "deterministic_random_fixture_only",
            "state_tensor_count": state_tensor_count,
            "state_shapes": state_shapes,
            "restore_max_abs_diff": restore_diff,
            "full_replay_max_abs_diff": replay_diff,
            "fresh_max_abs_diff": fresh_diff,
            "restore_equivalent": restore_diff <= 1e-5,
            "full_replay_equivalent": replay_diff <= 1e-5,
            "fresh_is_different": fresh_diff > 1e-5,
            "claim_boundary": (
                "Architecture/state-interface probe with deterministic random weights only. "
                "It does not validate the released xLSTM 7B checkpoint or establish VELA cognitive continuity."
            ),
        }
        write_report(report)
        if not (report["restore_equivalent"] and report["full_replay_equivalent"] and report["fresh_is_different"]):
            raise SystemExit(1)
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            raise
        report = {
            "status": "XLSTM_PROBE_ERROR",
            "source_commit": XLSTM_COMMIT,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-20:],
            "claim_boundary": "Runtime/setup failure only; do not interpret as an xLSTM architecture failure.",
        }
        write_report(report)
        raise


if __name__ == "__main__":
    run()
