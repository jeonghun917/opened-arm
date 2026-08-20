# Cori Kaggle T4 portability benchmark

Status: PREPARED / NOT YET EXECUTED

Purpose: test whether the frozen Cori Matcha-TTS continuation recipe can run unchanged on Kaggle's free single-GPU T4 environment before using Kaggle for any research continuation.

## Why T4, not P100

Kaggle's current kernel metadata documentation warns that the default Kaggle image uses a PyTorch/CUDA build that does not include Pascal (`sm_60`) kernels, so a P100 can appear CUDA-visible but fail on the first CUDA operation. Kaggle explicitly recommends T4 instead unless a Pascal-compatible torch build is installed.

For this experiment, installing a special P100-only torch stack would add an avoidable environment change. The first portability benchmark therefore targets a single T4.

Do not use T4 x2 for the first benchmark. The frozen C3 continuation recipe is single-GPU; multi-GPU execution would be a separate configuration change rather than a drop-in compute substitution.

## Frozen benchmark

- resume anchor: E280
- semantic epoch: 280
- global step: 140560
- E280 SHA-256: `081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9`
- target: E290 only
- segment length: 10 epochs
- batch size: 16, unchanged
- accelerator: one Nvidia Tesla T4
- Matcha commit: `bd4d90d93214b37f7a159cf205ae85762c2c10aa`
- C3 text patch: unchanged
- frozen train/valid filelists and mel statistics: unchanged
- optimizer state: must resume from the exact E280 checkpoint
- no gradient accumulation substitution, batch-size change, data resampling change, optimizer change, or multi-GPU substitution is allowed in the first benchmark

## Private-input contract

A Kaggle session is transient. The benchmark requires a private Kaggle Dataset containing the exact frozen assets below. Source audio, weights, manifests, and credentials must never be published as a public Kaggle Dataset or public Notebook.

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

The exact E280 checkpoint is rejected unless both its SHA and checkpoint metadata match epoch 280 / global step 140560.

## Execution order

1. Use one Kaggle T4 GPU.
2. Attach only the private C3 input Dataset.
3. Install/prepare the pinned Matcha checkout without altering the model/data/optimizer recipe.
4. Run `runner/kaggle_cori_segment_entry.py --preflight-only` first.
5. Confirm GPU model/VRAM, one visible GPU, E280 SHA, checkpoint metadata, handoff completeness, and dataset paths.
6. Run exactly E280 -> E290.
7. Record wall time, peak GPU memory, final global step, checkpoint SHA, and any OOM/error.
8. Stop. Do not continue to E300 until the benchmark is reviewed.

## Decision gate

- PASS: batch 16 completes E280 -> E290 with the unchanged scientific recipe and valid restart metadata. Kaggle T4 may then be considered for subsequent bounded segments.
- FAIL-OOM: do not silently reduce batch size. A single Kaggle T4 is not a drop-in continuation platform for the primary scaling experiment.
- FAIL-ENVIRONMENT: dependency/runtime incompatibility may be fixed only if the change does not alter model/data/optimizer semantics; rerun the same benchmark after the environment fix.
- FAIL-DATA: reconstruct or transfer the frozen dataset exactly before retrying; do not replace it with a newly filtered corpus.

## Current blocker

The code path is prepared, but the benchmark cannot start until Kaggle authentication and a private Dataset handle are available to the orchestration path. The user's token must be stored only as an encrypted repository secret or in Kaggle's own credential store, never in chat, code, logs, commits, or public artifacts.
