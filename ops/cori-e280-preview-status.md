# Cori E280 preview status — 2026-08-20

Operational note for the temporary `e280-preview-cpu` branch. This is non-canonical and contains no model weights, audio, private dataset contents, or credentials.

## Goal

Produce a descriptive listening preview of the accepted Cori Matcha E280 checkpoint without launching a paid GPU training job, while keeping the comparison vocoder fixed to the previously selected Cori-adapted BigVGAN.

## Accepted acoustic state

- epoch completed: 280
- global step: 140560
- SHA-256: `081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9`
- source kind: Lightning managed-job artifact
- recorded path: `/teamspace/jobs/c3-cori-e270-e280-b16-oa018/artifacts/c3-cori-lightning-runs/cori-e100-to-e550-b16/checkpoints/checkpoint_epoch=279.ckpt`

## Vocoder asset conclusion

The historical Modal→Lightning handoff copied the E100 acoustic checkpoint, frozen metadata, and train/valid filelists. It did **not** copy the Cori-adapted BigVGAN training weights. The continuation contract only names `Cori-adapted BigVGAN` as the fixed comparison vocoder.

The original adapted vocoder training run is therefore still referenced by its Modal volume path:

`/vol/training/cori/vocoder_adaptation/bigvgan_base_cori_22k80/20260817T022729Z`

A separate frozen Android candidate pack was previously produced with an exported BigVGAN ONNX model, but that exported runtime asset should not be silently substituted for the original PyTorch evaluation path without an explicit parity/reproducibility decision.

## What has been verified on this branch

- Public GitHub-hosted Actions runner starts normally.
- `LIGHTNING_USER_ID` and `LIGHTNING_API_KEY` are present in `opened-arm` Actions secrets.
- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` are not present in `opened-arm` Actions secrets.
- Direct `Teamspace.download_file()` against the synthesized `jobs/.../artifacts/...` path returns HTTP 404 / `NoSuchKey`.
- No paid GPU training job was launched by the preview attempts.

## Next safe technical route

1. Recover/inspect E280 through the existing Lightning CPU Studio mount, which is the same mechanism the continuation controller used when scanning job artifacts.
2. Verify the recovered checkpoint SHA-256 before use.
3. Determine whether an exact copy of the adapted BigVGAN generator/config exists anywhere in the Lightning Studio filesystem. Do not assume it does.
4. If the exact adapted vocoder is not present on Lightning, either:
   - add Modal credentials to `opened-arm` as encrypted Actions secrets and use the existing private Modal volume; or
   - intentionally establish a new parity-verified evaluation path using the frozen exported vocoder asset.
5. Synthesize E280 on CPU only, retain listening output only as a short-lived artifact, and do not promote E280 based on this descriptive preview alone.

## User action boundary

No user action is needed for repository/Drive documentation or code cleanup. User action is needed only if the exact vocoder cannot be recovered from Lightning and Modal access is required: the user must add provider credentials through GitHub repository settings rather than sending credentials in chat or committing them.

## Paid-compute gate

Training remains paused. Do not relaunch E290 or any later L4/GPU segment until a new compute budget/provider is explicitly available and the user authorizes the paid run.
