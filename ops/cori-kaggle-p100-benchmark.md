# Cori Kaggle P100 portability benchmark

Status: PREPARED / NOT YET EXECUTED

Purpose: test whether the frozen Cori Matcha-TTS continuation recipe can run unchanged on Kaggle's free single-GPU P100 environment before using Kaggle for any research continuation.

## Frozen benchmark

- resume anchor: E280
- semantic epoch: 280
- global step: 140560
- E280 SHA-256: `081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9`
- target: E290 only
- segment length: 10 epochs
- batch size: 16, unchanged
- Matcha commit: `bd4d90d93214b37f7a159cf205ae85762c2c10aa`
- C3 text patch: unchanged
- frozen train/valid filelists and mel statistics: unchanged
- optimizer state: must resume from the exact E280 checkpoint
- no gradient accumulation substitution, batch-size change, data resampling change, or optimizer change is allowed in the first benchmark

## Kaggle assumptions

Kaggle's current notebook documentation describes free P100 and T4 x2 accelerator options. The first benchmark should use P100 because the C3 continuation recipe is a single-GPU recipe; T4 x2 would not automatically double usable memory for one process and would introduce a multi-GPU configuration change.

A Kaggle session is transient. Therefore:

1. the exact E280 checkpoint must be supplied as a private Kaggle input;
2. the original C3 handoff bundle must be supplied as a private Kaggle input;
3. the frozen Cori dataset tree must be supplied with the same relative layout expected by the handoff filelists;
4. outputs must be copied out of `/kaggle/working` after the benchmark.

Do not publish source audio, private manifests, weights, or credentials in a public Kaggle Dataset or public Notebook.

## Required private input layout

The paths may differ, but the content contract is:

```text
input/
  c3-cori-handoff/
    cori_matcha_epoch100.ckpt
    HANDOFF_MANIFEST.json
    metadata/FREEZE.json
    metadata/PREPARED.json
    metadata/STATS.json
    filelists/train.txt
    filelists/valid.txt
  c3-cori-e280/
    checkpoint_epoch=279.ckpt
  c3-cori-dataset/
    <same relative audio tree used by the frozen Cori dataset>
```

The exact E280 checkpoint is rejected unless both its SHA and checkpoint metadata match E280 / step 140560.

## Execution order

1. Enable a Kaggle P100 GPU.
2. Clone `opened-arm` at the reviewed commit/branch.
3. Install the pinned Matcha checkout and its dependencies.
4. Run `runner/kaggle_cori_segment_entry.py --preflight-only` first.
5. Confirm GPU model/VRAM, E280 SHA, handoff completeness, and dataset path existence.
6. Run exactly E280 -> E290.
7. Record wall time, peak GPU memory, final global step, checkpoint SHA, and any OOM/error.
8. Stop. Do not continue to E300 until the benchmark is reviewed.

## Decision gate

- PASS: batch 16 completes E280 -> E290 with unchanged scientific recipe and valid restart metadata. Kaggle may be used for subsequent bounded segments.
- FAIL-OOM: do not silently reduce batch size. Kaggle P100 is not a drop-in continuation platform for the primary scaling experiment.
- FAIL-ENVIRONMENT: dependency/runtime incompatibility may be fixed only if it does not change model/data/optimizer/inference semantics; rerun the same benchmark after the environment fix.
- FAIL-DATA: reconstruct or transfer the frozen dataset exactly before retrying; do not replace it with a newly filtered corpus.

## Current blocker

No Kaggle credentials/private dataset inputs are currently wired into `opened-arm`. The portability entrypoint is prepared, but the benchmark cannot be executed from GitHub alone until the user's Kaggle account has the private input bundle attached.
