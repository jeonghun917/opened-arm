# Cori canonical asset registry

Last verified: 2026-08-20 KST

This file is a metadata-only registry. Model weights, source audio, private datasets/manifests, and credentials must never be committed to this public repository.

## Storage policy

Canonical binary anchors are kept in private provider storage, separately from transient training-job outputs. The current Lightning archive is the persistent Studio vault:

`/teamspace/studios/this_studio/C3_ASSET_VAULT`

Only accepted/formal milestone assets should be copied into the vault. Do not mirror every 10-epoch continuation checkpoint.

Every archived binary must be verified against its frozen SHA-256 before and after copying. The vault filename includes the expected SHA where practical.

## Matcha acoustic anchors

### E100

- status: archived and SHA-verified
- SHA-256: `f4409103780820e356b609ec79c425cb1cffd3059fed163e1f60bfe926438273`
- source: `/teamspace/studios/c3-cori-e100-e200/c3-migration/c3-stage-src/handoff/cori_matcha_epoch100.ckpt`
- vault: `/teamspace/studios/this_studio/C3_ASSET_VAULT/cori/matcha/E100/cori_matcha_epoch100__sha256_f4409103780820e356b609ec79c425cb1cffd3059fed163e1f60bfe926438273.ckpt`

### E200

- status: archived and SHA-verified
- SHA-256: `b3235e8bff23c6241119add85e57dccfa1e88ed2cf2ed51bed8a3c305dee5c54`
- source: `/teamspace/jobs/c3-cori-e190-e200-b16-sh010/artifacts/c3-cori-lightning-runs/cori-e100-to-e200-b16/checkpoints/checkpoint_epoch=199.ckpt`
- vault: `/teamspace/studios/this_studio/C3_ASSET_VAULT/cori/matcha/E200/checkpoint_epoch=199__sha256_b3235e8bff23c6241119add85e57dccfa1e88ed2cf2ed51bed8a3c305dee5c54.ckpt`

### E280

- status: archived and SHA-verified
- semantic epoch: 280
- global step: 140560
- SHA-256: `081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9`
- source: `/teamspace/jobs/c3-cori-e270-e280-b16-oa018/artifacts/c3-cori-lightning-runs/cori-e100-to-e550-b16/checkpoints/checkpoint_epoch=279.ckpt`
- vault: `/teamspace/studios/this_studio/C3_ASSET_VAULT/cori/matcha/E280/checkpoint_epoch=279__sha256_081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9.ckpt`

E280 is an accepted restart anchor, not an automatic perceptual-quality promotion.

## Cori-adapted BigVGAN

Historical exact run reference:

`/vol/training/cori/vocoder_adaptation/bigvgan_base_cori_22k80/20260817T022729Z`

Status as of 2026-08-20:

- exact PyTorch generator/config were recovered from the original Modal volume and mirrored into persistent Lightning Studio storage;
- Lightning compute was not started for the mirror; Studio storage transfer was used directly;
- generator: `generator_final.pt`, 56,013,388 bytes, SHA-256 `c4a76a15d367f1e291ee5c495fa41655cf7c44cb10f47a09f8cc31ef73ed16b1`;
- config: `checkpoints/config.json`, 1,216 bytes, SHA-256 `3fe116e1826849ea102eb3edf3eb0cc9c52dc0319e85feafcafdb66d89680896`;
- vault root: `C3_ASSET_VAULT/cori/bigvgan/adapted_20260817T022729Z`;
- the mirrored bytes were downloaded back from Lightning and SHA-verified after transfer;
- the exported Android/ONNX vocoder remains a separate runtime artifact and is not treated as an automatic substitute for this frozen PyTorch evaluation path.

## E280 descriptive preview

A CPU-only descriptive preview was generated on 2026-08-20 using:

- acoustic checkpoint: E280 / global_step 140560 / SHA-256 `081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9`;
- vocoder: exact Cori-adapted BigVGAN run above;
- inference: `n_timesteps=10`, `temperature=0.667`, `length_scale=1.0`;
- five fixed listening items;
- no EQ, reverb, compressor, pause surgery, or ending-duration patch;
- GPU allocated: false.

The preview passed checkpoint/vocoder loading and output verification. It is descriptive listening material only, not a formal milestone evaluation or automatic quality promotion.

## Execution and security boundary

- `opened-arm` stores orchestration and metadata only.
- Provider credentials belong only in encrypted repository secrets; never in chat, code, logs, issues, or commits.
- Listening WAVs and model binaries are transient/private artifacts, not repository content.
- Paid GPU training remains paused. Asset inspection/archive work does not authorize restarting E290 or later training.
