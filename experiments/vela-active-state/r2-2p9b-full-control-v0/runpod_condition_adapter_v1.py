#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
from pathlib import Path

CONDITION_RUNNER_BLOB = "70b696bcfdb3ba7fc7f4010a1143aac1063c452b"
P100_GUARD = 'if cap!=(6,0) or "sm_60" not in arches: raise RuntimeError(f"frozen backend requires P100 sm_60; device={torch.cuda.get_device_name(0)} cap={cap} arches={arches}")'
A40_GUARD = 'if cap!=(8,6) or "sm_86" not in arches or "A40" not in torch.cuda.get_device_name(0): raise RuntimeError(f"RunPod backend requires A40 sm_86; device={torch.cuda.get_device_name(0)} cap={cap} arches={arches}")'
RUN_SCALE_CALL = 'rep=run_scale(model,pipe,spec["scale"],c,stimuli,evaluator,external);'
WARMED_RUN_SCALE_CALL = 'runpod_neutral_warmup(model,pipe); rep=run_scale(model,pipe,spec["scale"],c,stimuli,evaluator,external);'


def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def runpod_neutral_warmup(model, pipe) -> None:
    import torch
    zero = model.generate_zero_state()
    seed = [int(x) for x in pipe.encode("VELA RUNPOD SCIENCE NEUTRAL WARMUP")]
    cont = [int(x) for x in pipe.encode(" CONTINUATION CHECK")]
    _, state = model.forward(seed, [x.detach().clone() for x in zero])
    _, _ = model.forward(cont, [x.detach().clone() for x in state])
    torch.cuda.synchronize()
    del state


def main() -> None:
    root = Path(__file__).resolve().parent
    runner = root / "condition_runner.py"
    raw = runner.read_bytes()
    actual_blob = gitblob(raw)
    if actual_blob != CONDITION_RUNNER_BLOB:
        raise RuntimeError(f"condition_runner blob drift: {actual_blob}")
    source = raw.decode("utf-8")
    if source.count(P100_GUARD) != 1:
        raise RuntimeError("expected exactly one frozen P100 guard")
    if source.count(RUN_SCALE_CALL) != 1:
        raise RuntimeError("expected exactly one frozen run_scale call")
    adapted = source.replace(P100_GUARD, A40_GUARD).replace(RUN_SCALE_CALL, WARMED_RUN_SCALE_CALL)
    # No scientific file is modified on disk; the exact frozen runner is adapted in memory at two declared infrastructure seams only.
    ns = {
        "__name__": "__main__",
        "__file__": str(runner),
        "runpod_neutral_warmup": runpod_neutral_warmup,
    }
    exec(compile(adapted, str(runner) + "<runpod-adapted>", "exec"), ns, ns)


if __name__ == "__main__":
    main()
